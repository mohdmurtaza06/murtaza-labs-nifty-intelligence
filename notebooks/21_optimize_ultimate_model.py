import sys, json, warnings, itertools, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'src'))
from model_features_ultimate import build_features

DATA = BASE / 'data'
MODELS = BASE / 'models'
MODELS.mkdir(exist_ok=True)

ext_path = DATA / 'nifty50_external_features.csv'
live_path = DATA / 'live_market_data.csv'
ext = pd.read_csv(ext_path, parse_dates=['Date']).drop_duplicates('Date').set_index('Date').sort_index()
live = pd.read_csv(live_path, parse_dates=['Date']).drop_duplicates('Date').set_index('Date').sort_index()
raw = pd.concat([ext, live[live.index > ext.index.max()]], axis=0, sort=False).sort_index()
feat = build_features(raw)
labeled = feat.dropna(subset=['Target']).copy()
features = [c for c in feat.columns if c not in {'Target','Next_Return'} and feat[c].notna().mean() >= .75]
X = labeled[features].replace([np.inf,-np.inf], np.nan)
y = labeled['Target'].astype(int)
print(f'Training rows: {len(X)} features: {len(features)} range: {X.index.min().date()} -> {X.index.max().date()}')

# Walk-forward folds. Feature ranking is computed ONLY on the training portion of each fold.
initial = max(900, int(len(X)*0.62))
fold = 400
fold_starts = list(range(initial, len(X), fold))

configs = [
    {'name':'cat_base','depth':4,'learning_rate':0.025,'iterations':350,'l2_leaf_reg':12,'random_strength':1.0,'bagging_temperature':0.5},
    {'name':'cat_balanced','depth':5,'learning_rate':0.02,'iterations':400,'l2_leaf_reg':15,'random_strength':1.0,'bagging_temperature':0.8},
    {'name':'cat_conservative','depth':4,'learning_rate':0.02,'iterations':500,'l2_leaf_reg':20,'random_strength':1.5,'bagging_temperature':1.0},
    {'name':'cat_fast','depth':6,'learning_rate':0.02,'iterations':300,'l2_leaf_reg':18,'random_strength':1.2,'bagging_temperature':0.7},
]
feature_counts = [30, 50, 80, 120, len(features)]
records=[]

for s in fold_starts:
    e=min(s+fold,len(X))
    Xt, yt = X.iloc[:s], y.iloc[:s]
    Xv, yv = X.iloc[s:e], y.iloc[s:e]
    # Fill for MI ranking. Ranking never sees validation rows.
    imp=SimpleImputer(strategy='median')
    Xt_imp=pd.DataFrame(imp.fit_transform(Xt),index=Xt.index,columns=Xt.columns)
    mi=mutual_info_classif(Xt_imp,yt,random_state=42)
    rank=pd.Series(mi,index=Xt.columns).sort_values(ascending=False)
    print(f'Fold {X.index[s].date()} -> {X.index[e-1].date()}')
    for k in feature_counts:
        selected=rank.head(k).index.tolist()
        for cfg in configs:
            model=CatBoostClassifier(
                iterations=cfg['iterations'], depth=cfg['depth'], learning_rate=cfg['learning_rate'],
                l2_leaf_reg=cfg['l2_leaf_reg'], random_strength=cfg['random_strength'],
                bagging_temperature=cfg['bagging_temperature'], loss_function='Logloss',
                eval_metric='Logloss', verbose=False, random_seed=42, thread_count=-1,
                allow_writing_files=False
            )
            model.fit(Xt[selected],yt)
            p=model.predict_proba(Xv[selected])[:,1]
            records.append({
                'fold':X.index[s].date(),'config':cfg['name'],'features':k,
                'accuracy':accuracy_score(yv,p>=.5),
                'balanced_accuracy':balanced_accuracy_score(yv,p>=.5),
                'logloss':log_loss(yv,p,labels=[0,1]),
                'auc':roc_auc_score(yv,p),'n':len(yv)
            })

res=pd.DataFrame(records)
summary=res.groupby(['config','features'])[['accuracy','balanced_accuracy','logloss','auc']].mean().sort_values(['logloss','balanced_accuracy'],ascending=[True,False])
print('\nOPTIMIZATION SUMMARY\n')
print(summary.to_string())
res.to_csv(DATA/'ultimate_optimization_results.csv',index=False)
summary.reset_index().to_csv(DATA/'ultimate_optimization_summary.csv',index=False)

best_cfg,best_k=summary.index[0]
best_row=summary.iloc[0]
print(f'\nBEST BY WALK-FORWARD LOGLOSS: {best_cfg} with {best_k} features')
print(best_row.to_string())

# Rank features on ALL historical training data for production fit.
imp=SimpleImputer(strategy='median')
Ximp=pd.DataFrame(imp.fit_transform(X),index=X.index,columns=X.columns)
mi=mutual_info_classif(Ximp,y,random_state=42)
rank=pd.Series(mi,index=X.columns).sort_values(ascending=False)
selected=rank.head(int(best_k)).index.tolist()

cfg=next(c for c in configs if c['name']==best_cfg)
model=CatBoostClassifier(
    iterations=cfg['iterations'], depth=cfg['depth'], learning_rate=cfg['learning_rate'],
    l2_leaf_reg=cfg['l2_leaf_reg'], random_strength=cfg['random_strength'],
    bagging_temperature=cfg['bagging_temperature'], loss_function='Logloss',
    verbose=False, random_seed=42, thread_count=-1, allow_writing_files=False
)
model.fit(X[selected],y)

meta={
    'architecture':'CATBOOST_SELECTED_FEATURES',
    'selected_config':best_cfg,
    'selected_feature_count':int(best_k),
    'features':selected,
    'trained_through':str(X.index.max().date()),
    'training_start':str(X.index.min().date()),
    'samples':int(len(X)),
    'walk_forward_summary':summary.reset_index().to_dict('records'),
    'top_mutual_information_features':rank.head(80).to_dict(),
}
joblib.dump({'model':model,'feature_cols':selected,'architecture':'CATBOOST_SELECTED_FEATURES','trained_through':str(X.index.max().date())}, MODELS/'nifty50_optimized_model.pkl')
(MODELS/'nifty50_optimized_meta.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
print(f'\nSaved optimized model: {MODELS / "nifty50_optimized_model.pkl"}')
print('Selected features:', ', '.join(selected))
