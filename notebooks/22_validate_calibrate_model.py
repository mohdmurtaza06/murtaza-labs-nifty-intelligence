import sys, json, warnings, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score, brier_score_loss
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'src'))
from model_features_ultimate import build_features

DATA = BASE / 'data'
MODELS = BASE / 'models'
MODELS.mkdir(exist_ok=True)

EXT = DATA / 'nifty50_external_features.csv'
LIVE = DATA / 'live_market_data.csv'
if not EXT.exists():
    raise FileNotFoundError(f'Missing {EXT}')

ext = pd.read_csv(EXT, parse_dates=['Date']).drop_duplicates('Date').set_index('Date').sort_index()
if LIVE.exists():
    live = pd.read_csv(LIVE, parse_dates=['Date']).drop_duplicates('Date').set_index('Date').sort_index()
    raw = pd.concat([ext, live[live.index > ext.index.max()]], axis=0, sort=False).sort_index()
else:
    raw = ext.copy()

feat = build_features(raw)
labeled = feat.dropna(subset=['Target']).copy()
features = [c for c in feat.columns if c not in {'Target','Next_Return'} and feat[c].notna().mean() >= .75]
X = labeled[features].replace([np.inf, -np.inf], np.nan)
y = labeled['Target'].astype(int)
print(f'Training rows: {len(X)} features: {len(features)} range: {X.index.min().date()} -> {X.index.max().date()}')

meta_path = MODELS / 'nifty50_optimized_meta.json'
if not meta_path.exists():
    raise FileNotFoundError('Run 21_optimize_ultimate_model.py first.')
meta = json.loads(meta_path.read_text(encoding='utf-8'))
best_cfg_name = meta['selected_config']
best_k = int(meta['selected_feature_count'])

configs = {
    'cat_base': {'depth':4,'learning_rate':0.025,'iterations':350,'l2_leaf_reg':12,'random_strength':1.0,'bagging_temperature':0.5},
    'cat_balanced': {'depth':5,'learning_rate':0.02,'iterations':400,'l2_leaf_reg':15,'random_strength':1.0,'bagging_temperature':0.8},
    'cat_conservative': {'depth':4,'learning_rate':0.02,'iterations':500,'l2_leaf_reg':20,'random_strength':1.5,'bagging_temperature':1.0},
    'cat_fast': {'depth':6,'learning_rate':0.02,'iterations':300,'l2_leaf_reg':18,'random_strength':1.2,'bagging_temperature':0.7},
}
cfg = configs[best_cfg_name]

initial = max(900, int(len(X)*0.62))
fold = 400
fold_starts = list(range(initial, len(X), fold))
records=[]
oos_probs=[]
oos_y=[]
oos_dates=[]
previous_probs=[]
previous_y=[]


def make_model():
    return CatBoostClassifier(
        iterations=cfg['iterations'], depth=cfg['depth'], learning_rate=cfg['learning_rate'],
        l2_leaf_reg=cfg['l2_leaf_reg'], random_strength=cfg['random_strength'],
        bagging_temperature=cfg['bagging_temperature'], loss_function='Logloss',
        verbose=False, random_seed=42, thread_count=-1, allow_writing_files=False
    )


def fit_calibrator(p, yv):
    if len(p) < 120 or len(np.unique(yv)) < 2:
        return None
    z = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6)).reshape(-1,1)
    cal = LogisticRegression(C=1.0, solver='lbfgs')
    cal.fit(z, yv)
    return cal


def calibrate(p, cal):
    if cal is None:
        return p
    z = np.log(np.clip(p, 1e-6, 1-1e-6) / np.clip(1-p, 1e-6, 1-1e-6)).reshape(-1,1)
    return cal.predict_proba(z)[:,1]

print(f'Using optimized configuration: {best_cfg_name}, {best_k} features')

for s in fold_starts:
    e=min(s+fold,len(X))
    Xt, yt = X.iloc[:s], y.iloc[:s]
    Xv, yv = X.iloc[s:e], y.iloc[s:e]
    imp=SimpleImputer(strategy='median')
    Xt_imp=pd.DataFrame(imp.fit_transform(Xt),index=Xt.index,columns=Xt.columns)
    # Feature ranking is strictly training-only for this fold.
    from sklearn.feature_selection import mutual_info_classif
    mi=mutual_info_classif(Xt_imp,yt,random_state=42)
    rank=pd.Series(mi,index=Xt.columns).sort_values(ascending=False)
    selected=rank.head(best_k).index.tolist()

    model=make_model()
    model.fit(Xt[selected],yt)
    raw_p=model.predict_proba(Xv[selected])[:,1]

    cal=fit_calibrator(np.asarray(previous_probs), np.asarray(previous_y))
    cal_p=calibrate(raw_p,cal)

    for d, yy, rp, cp in zip(Xv.index, yv.to_numpy(), raw_p, cal_p):
        records.append({'date':d,'y':int(yy),'raw_probability':float(rp),'calibrated_probability':float(cp),'fold_start':X.index[s]})
    oos_probs.extend(cal_p.tolist()); oos_y.extend(yv.tolist()); oos_dates.extend(Xv.index.tolist())
    previous_probs.extend(raw_p.tolist())
    previous_y.extend(yv.tolist())
    print(f'Fold {X.index[s].date()} -> {X.index[e-1].date()} | n={len(yv)} | raw_acc={accuracy_score(yv,raw_p>=.5):.4f} | cal_acc={accuracy_score(yv,cal_p>=.5):.4f}')

pred = pd.DataFrame(records).sort_values('date').reset_index(drop=True)

# Threshold analysis on the complete out-of-sample stream. These are fixed thresholds, not fitted to the outcomes.
threshold_rows=[]
for threshold in [0.50,0.52,0.55,0.57,0.60,0.62,0.65,0.67,0.70,0.75]:
    p=pred['calibrated_probability'].to_numpy()
    yv=pred['y'].to_numpy()
    edge=np.abs(p-0.5)
    keep=edge >= (threshold-0.5)
    if keep.sum()==0:
        continue
    direction=(p[keep]>=0.5).astype(int)
    yy=yv[keep]
    threshold_rows.append({
        'threshold':threshold,
        'predictions':int(keep.sum()),
        'coverage':float(keep.mean()),
        'accuracy':float(accuracy_score(yy,direction)),
        'balanced_accuracy':float(balanced_accuracy_score(yy,direction)),
        'avg_probability':float(np.maximum(p[keep],1-p[keep]).mean())
    })
thresholds=pd.DataFrame(threshold_rows)

# Chronological holdout: last 30% of OOS predictions is kept separate from calibration fitting.
split=max(1,int(len(pred)*0.70))
calibration_part=pred.iloc[:split]
holdout=pred.iloc[split:]
final_cal=fit_calibrator(calibration_part['raw_probability'].to_numpy(), calibration_part['y'].to_numpy())
holdout_cal=calibrate(holdout['raw_probability'].to_numpy(), final_cal)

holdout_metrics={
    'n':int(len(holdout)),
    'accuracy':float(accuracy_score(holdout['y'],holdout_cal>=.5)),
    'balanced_accuracy':float(balanced_accuracy_score(holdout['y'],holdout_cal>=.5)),
    'logloss':float(log_loss(holdout['y'],holdout_cal,labels=[0,1])),
    'auc':float(roc_auc_score(holdout['y'],holdout_cal)) if len(np.unique(holdout['y']))==2 else None,
    'brier':float(brier_score_loss(holdout['y'],holdout_cal)),
}

# Fit production calibrator on ALL OOS predictions. Those predictions are out-of-sample relative to their model fits.
production_cal=fit_calibrator(pred['raw_probability'].to_numpy(), pred['y'].to_numpy())
cal_params={}
if production_cal is not None:
    cal_params={'type':'platt_logit','coef':float(production_cal.coef_[0,0]),'intercept':float(production_cal.intercept_[0])}

pred.to_csv(DATA/'ultimate_calibrated_oos_predictions.csv',index=False)
thresholds.to_csv(DATA/'ultimate_confidence_thresholds.csv',index=False)

cal_meta={
    'base_model':'nifty50_optimized_model.pkl',
    'configuration':best_cfg_name,
    'feature_count':best_k,
    'calibration':cal_params,
    'oos_samples':int(len(pred)),
    'oos_start':str(pred.date.min().date()),
    'oos_end':str(pred.date.max().date()),
    'holdout_last_30pct_metrics':holdout_metrics,
    'thresholds':threshold_rows,
    'note':'Thresholds are fixed evaluation bands. No threshold was selected from the holdout outcomes.'
}
(MODELS/'nifty50_calibration_meta.json').write_text(json.dumps(cal_meta,indent=2,default=str),encoding='utf-8')

print('\nCALIBRATION SUMMARY')
print(f'OOS samples: {len(pred)}')
print(f'Raw OOS accuracy: {accuracy_score(pred.y,pred.raw_probability>=.5):.4f}')
print(f'Calibrated OOS accuracy: {accuracy_score(pred.y,pred.calibrated_probability>=.5):.4f}')
print(f'Calibrated OOS logloss: {log_loss(pred.y,pred.calibrated_probability,labels=[0,1]):.4f}')
print(f'Calibrated OOS Brier: {brier_score_loss(pred.y,pred.calibrated_probability):.4f}')
print('\nFIXED CONFIDENCE THRESHOLDS')
print(thresholds.to_string(index=False))
print('\nCHRONOLOGICAL HOLDOUT (LAST 30% OF OOS)')
for k,v in holdout_metrics.items(): print(f'{k}: {v}')
print(f'\nSaved: {DATA / "ultimate_calibrated_oos_predictions.csv"}')
print(f'Saved: {DATA / "ultimate_confidence_thresholds.csv"}')
print(f'Saved: {MODELS / "nifty50_calibration_meta.json"}')
