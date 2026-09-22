import numpy as np
import pandas as pd

def _rsi(c,n):
    d=c.diff(); up=d.clip(lower=0); dn=-d.clip(upper=0)
    ag=up.ewm(alpha=1/n,adjust=False).mean(); al=dn.ewm(alpha=1/n,adjust=False).mean()
    return 100-100/(1+ag/(al+1e-12))

def _atr(h,l,c,n):
    pc=c.shift(1); tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/n,adjust=False).mean()

def build_features(raw):
    df=raw.copy().sort_index()
    for c in ['Open','High','Low','Close','Volume']:
        if c in df: df[c]=pd.to_numeric(df[c],errors='coerce')
    c=df['Close']; h=df['High']; l=df['Low']; o=df['Open']; v=df['Volume']
    f=pd.DataFrame(index=df.index)
    for n in [1,2,3,4,5,7,10,15,20,30,40,60,90,120]: f[f'RET_{n}']=c.pct_change(n)
    f['BODY']=(c-o)/c; f['BODY_ABS']=f['BODY'].abs(); f['RANGE']=(h-l)/c; f['GAP']=(o-c.shift(1))/(c.shift(1)+1e-12)
    f['UPPER_WICK']=(h-pd.concat([o,c],axis=1).max(axis=1))/c; f['LOWER_WICK']=(pd.concat([o,c],axis=1).min(axis=1)-l)/c; f['CLOSE_LOCATION']=(c-l)/(h-l+1e-12)
    for n in [5,10,20,30,50,75,100,150,200]:
        ma=c.rolling(n).mean(); f[f'PX_MA_{n}']=c/(ma+1e-12)-1; f[f'MA_SLOPE_{n}']=ma.pct_change(5); f[f'MA_SLOPE10_{n}']=ma.pct_change(10)
    f['MA5_MA20']=c.rolling(5).mean()/(c.rolling(20).mean()+1e-12)-1; f['MA20_MA50']=c.rolling(20).mean()/(c.rolling(50).mean()+1e-12)-1; f['MA50_MA200']=c.rolling(50).mean()/(c.rolling(200).mean()+1e-12)-1
    r=c.pct_change()
    for n in [5,10,15,20,30,40,60,90]: f[f'VOL_{n}']=r.rolling(n).std()
    f['VOL_REGIME']=f['VOL_20']/(f['VOL_60']+1e-12); f['VOL_REGIME_SHORT']=f['VOL_10']/(f['VOL_60']+1e-12)
    for n in [5,7,9,14,21,28]: f[f'RSI_{n}']=_rsi(c,n)
    f['RSI14_CHANGE3']=f['RSI_14'].diff(3); f['RSI14_CHANGE5']=f['RSI_14'].diff(5)
    for fast,slow,sig in [(5,13,4),(8,21,5),(12,26,9),(19,39,9)]:
        mac=c.ewm(span=fast,adjust=False).mean()-c.ewm(span=slow,adjust=False).mean(); sm=mac.ewm(span=sig,adjust=False).mean(); f[f'MACD_{fast}_{slow}']=mac/(c+1e-12); f[f'MACD_HIST_{fast}_{slow}']=(mac-sm)/(c+1e-12); f[f'MACD_SIGNAL_{fast}_{slow}']=sm/(c+1e-12)
    for n in [10,20,30,50]:
        ma=c.rolling(n).mean(); sd=c.rolling(n).std(); up=ma+2*sd; dn=ma-2*sd; f[f'BB_POS_{n}']=(c-dn)/(up-dn+1e-12); f[f'BB_WIDTH_{n}']=(up-dn)/(ma+1e-12)
    prev=c.shift(1); tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    for n in [5,10,14,21,28]: f[f'ATR_{n}']=tr.ewm(alpha=1/n,adjust=False).mean()/(c+1e-12)
    n=14; up=h.diff(); dn=-l.diff(); plus=up.where((up>dn)&(up>0),0.0); minus=dn.where((dn>up)&(dn>0),0.0); atrv=tr.ewm(alpha=1/n,adjust=False).mean(); pdi=100*plus.ewm(alpha=1/n,adjust=False).mean()/(atrv+1e-12); mdi=100*minus.ewm(alpha=1/n,adjust=False).mean()/(atrv+1e-12); dx=100*(pdi-mdi).abs()/(pdi+mdi+1e-12); f['ADX_14']=dx.ewm(alpha=1/n,adjust=False).mean(); f['PLUS_DI_14']=pdi; f['MINUS_DI_14']=mdi; f['DI_SPREAD_14']=(pdi-mdi)/100
    for n in [5,9,14,21]:
        ll=l.rolling(n).min(); hh=h.rolling(n).max(); f[f'STOCH_{n}']=100*(c-ll)/(hh-ll+1e-12); f[f'WILLIAMS_{n}']=-100*(hh-c)/(hh-ll+1e-12); tp=(h+l+c)/3; sma=tp.rolling(n).mean(); md=(tp-sma).abs().rolling(n).mean(); f[f'CCI_{n}']=(tp-sma)/(0.015*md+1e-12)
    for n in [5,10,20,30,50,60]: f[f'VOL_RATIO_{n}']=v/(v.rolling(n).mean()+1e-12)
    f['VOLUME_CHANGE']=v.pct_change(); obv=(np.sign(c.diff()).fillna(0)*v).cumsum(); f['OBV_SLOPE']=obv.pct_change(20).replace([np.inf,-np.inf],np.nan)
    for n in [10,20,50]:
        hh=h.rolling(n).max(); ll=l.rolling(n).min(); f[f'DONCHIAN_POS_{n}']=(c-ll)/(hh-ll+1e-12); f[f'HIGH_BREAK_{n}']=c/(hh+1e-12)-1; f[f'LOW_BREAK_{n}']=c/(ll+1e-12)-1
    # External / cross-market variables are shifted already by the live-data pipeline. Keep only known fields.
    for col in df.columns:
        if any(k in col.lower() for k in ['vix','bank','sp500','usd_inr','gold','crude']):
            f[col]=pd.to_numeric(df[col],errors='coerce')
            for lag in [1,2,3]: f[f'{col}_LAG{lag}']=f[col].shift(lag)
    f['TREND_SCORE']=(f['PX_MA_20']+f['PX_MA_50']+f['PX_MA_200'])/3
    f['MOMENTUM_SCORE']=(f['RET_5']+f['RET_10']+f['RET_20'])/3
    f['RISK_ON_SCORE']=0.0
    for col,sign in [('NIFTY_Bank_Return_1D',1),('SP500_Return_1D',1),('USD_INR_Return_1D',-1),('Gold_Return_1D',-1),('Crude_Oil_Return_1D',1),('India_VIX_Return_1D',-1)]:
        if col in df: f['RISK_ON_SCORE'] += sign*pd.to_numeric(df[col],errors='coerce').fillna(0)
    # Next-day target. This is not included in the feature matrix.
    nxt=c.shift(-1)/c-1; f['Next_Return']=nxt; f['Target']=(nxt>0).astype(float); f.loc[nxt.isna(),'Target']=np.nan
    return f

