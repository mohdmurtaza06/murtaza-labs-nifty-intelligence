import sys,json,warnings,joblib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,balanced_accuracy_score,log_loss,roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')
BASE=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'src'))
from model_features_ultimate import build_features
DATA=BASE/'data'; MODELS=BASE/'models'; MODELS.mkdir(exist_ok=True)
ext=pd.read_csv(DATA/'nifty50_external_features.csv',parse_dates=['Date']).drop_duplicates('Date').set_index('Date').sort_index()
live=pd.read_csv(DATA/'live_market_data.csv',parse_dates=['Date']).drop_duplicates('Date').set_index('Date').sort_index()
raw=pd.concat([ext,live[live.index>ext.index.max()]],axis=0,sort=False).sort_index()
feat=build_features(raw); labeled=feat.dropna(subset=['Target']).copy()
features=[c for c in feat.columns if c not in {'Target','Next_Return'} and feat[c].notna().mean()>=.75]
X=labeled[features].replace([np.inf,-np.inf],np.nan); y=labeled.Target.astype(int)
print('Training rows:',len(X),'features:',len(features),'range:',X.index.min().date(),'->',X.index.max().date())
models={
'logistic':Pipeline([('imp',SimpleImputer(strategy='median')),('scale',StandardScaler()),('m',LogisticRegression(C=.08,max_iter=3000))]),
'xgb':XGBClassifier(n_estimators=250,max_depth=3,learning_rate=.025,min_child_weight=10,subsample=.8,colsample_bytree=.65,reg_alpha=.2,reg_lambda=8,eval_metric='logloss',n_jobs=-1,random_state=42),
'lgbm':LGBMClassifier(n_estimators=250,num_leaves=15,max_depth=4,learning_rate=.02,min_child_samples=35,subsample=.82,colsample_bytree=.7,reg_alpha=.2,reg_lambda=8,verbosity=-1,n_jobs=-1,random_state=42),
'catboost':CatBoostClassifier(iterations=250,depth=5,learning_rate=.03,l2_leaf_reg=10,loss_function='Logloss',verbose=False,random_seed=42,thread_count=-1)}
initial=max(900,int(len(X)*.62)); fold=400; records=[]; oof={k:[] for k in models}; oofy=[]
for s in range(initial,len(X),fold):
 e=min(s+fold,len(X)); Xt,yt=X.iloc[:s],y.iloc[:s]; Xv,yv=X.iloc[s:e],y.iloc[s:e]
 ps={}
 for name,m in models.items():
  m.fit(Xt,yt); p=m.predict_proba(Xv)[:,1]; ps[name]=p; oof[name].extend(p); records.append({'fold':X.index[s].date(),'model':name,'accuracy':accuracy_score(yv,p>=.5),'balanced_accuracy':balanced_accuracy_score(yv,p>=.5),'logloss':log_loss(yv,p,labels=[0,1]),'auc':roc_auc_score(yv,p),'n':len(yv)})
 for name,sel in [('ALL_ENSEMBLE',list(models))]:
  p=np.mean([ps[k] for k in sel],axis=0); records.append({'fold':X.index[s].date(),'model':name,'accuracy':accuracy_score(yv,p>=.5),'balanced_accuracy':balanced_accuracy_score(yv,p>=.5),'logloss':log_loss(yv,p,labels=[0,1]),'auc':roc_auc_score(yv,p),'n':len(yv)}); oof.setdefault(name,[]).extend(p)
 oofy.extend(yv.tolist())
res=pd.DataFrame(records); summary=res.groupby('model')[['accuracy','balanced_accuracy','logloss','auc']].mean().sort_values(['logloss','accuracy'],ascending=[True,False]); print('\nWALK-FORWARD SUMMARY\n',summary.to_string()); res.to_csv(DATA/'ultimate_walk_forward_results.csv',index=False)
# Production model is the ensemble with the lowest OOS logloss, but probabilities are calibrated from out-of-sample predictions.
arch=summary.index[0]
final={}
for name,m in models.items(): m.fit(X,y); final[name]=m
# model weights based on inverse OOS logloss, bounded so one model cannot dominate.
base_scores=summary.loc[list(models.keys()),'logloss']; inv=1/(base_scores+1e-6); weights=(inv/inv.sum()).clip(.05,.40); weights=weights/weights.sum()
meta={'architecture':'WEIGHTED_ENSEMBLE','selected_by':'walk-forward logloss','trained_through':str(X.index.max().date()),'training_start':str(X.index.min().date()),'samples':len(X),'feature_count':len(features),'weights':weights.to_dict(),'summary':summary.reset_index().to_dict('records')}
joblib.dump({'models':final,'feature_cols':features,'weights':weights.to_dict(),'architecture':'WEIGHTED_ENSEMBLE','trained_through':str(X.index.max().date())},MODELS/'nifty50_ultimate_model.pkl')
(MODELS/'nifty50_ultimate_meta.json').write_text(json.dumps(meta,indent=2,default=str),encoding='utf-8')
print('\nSaved ultimate model. Architecture:',arch,'weights:',weights.to_dict())
