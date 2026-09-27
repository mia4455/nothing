"""Deterministic, dependency-light professional crypto analytics.
All functions use only historical/current OHLCV supplied by the caller.
No LLM is involved in calculations and no future bars are read.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x=float(v); return x if np.isfinite(x) else default
    except (TypeError, ValueError): return default


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d=df.copy(); o,h,l,c,v=(d[x].astype(float) for x in ("Open","High","Low","Close","Volume"))
    for n in (9,20,50,100,200): d[f"EMA{n}"]=c.ewm(span=n,adjust=False).mean()
    for n in (20,50,200): d[f"SMA{n}"]=c.rolling(n).mean()
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    d["RSI"]=100-100/(1+gain/loss.replace(0,np.nan)); rmin=d.RSI.rolling(14).min(); rmax=d.RSI.rolling(14).max()
    d["STOCH_RSI"]=(d.RSI-rmin)/(rmax-rmin).replace(0,np.nan)*100
    d["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(); d["MACD_SIGNAL"]=d.MACD.ewm(span=9,adjust=False).mean()
    prev=c.shift(); tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1); d["ATR"]=tr.ewm(alpha=1/14,adjust=False).mean()
    up=h.diff(); down=-l.diff(); plus=up.where((up>down)&(up>0),0.0); minus=down.where((down>up)&(down>0),0.0)
    plus_di=100*plus.ewm(alpha=1/14,adjust=False).mean()/d.ATR; minus_di=100*minus.ewm(alpha=1/14,adjust=False).mean()/d.ATR
    d["ADX"]=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,np.nan)).ewm(alpha=1/14,adjust=False).mean()
    mid=c.rolling(20).mean(); sd=c.rolling(20).std(); d["BB_MID"]=mid; d["BB_UPPER"]=mid+2*sd; d["BB_LOWER"]=mid-2*sd
    d["KC_MID"]=d.EMA20; d["KC_UPPER"]=d.EMA20+1.5*d.ATR; d["KC_LOWER"]=d.EMA20-1.5*d.ATR
    typical=(h+l+c)/3; d["VWAP"]=(typical*v).cumsum()/v.cumsum().replace(0,np.nan)
    direction=np.sign(c.diff()).fillna(0); d["OBV"]=(direction*v).cumsum(); d["VOL_MA20"]=v.rolling(20).mean()
    nine_high=h.rolling(9).max(); nine_low=l.rolling(9).min(); d["TENKAN"]=(nine_high+nine_low)/2
    high26=h.rolling(26).max(); low26=l.rolling(26).min(); d["KIJUN"]=(high26+low26)/2
    d["ICHI_A"]=(d.TENKAN+d.KIJUN)/2; d["ICHI_B"]=(h.rolling(52).max()+l.rolling(52).min())/2
    # Supertrend iterative final bands.
    hl2=(h+l)/2; upper=hl2+3*d.ATR; lower=hl2-3*d.ATR; st=pd.Series(index=d.index,dtype=float); trend=pd.Series(1,index=d.index,dtype=int)
    for i in range(1,len(d)):
        if c.iloc[i]>upper.iloc[i-1]: trend.iloc[i]=1
        elif c.iloc[i]<lower.iloc[i-1]: trend.iloc[i]=-1
        else: trend.iloc[i]=trend.iloc[i-1]
        if trend.iloc[i]>0: lower.iloc[i]=max(lower.iloc[i],lower.iloc[i-1]) if np.isfinite(lower.iloc[i-1]) else lower.iloc[i]
        else: upper.iloc[i]=min(upper.iloc[i],upper.iloc[i-1]) if np.isfinite(upper.iloc[i-1]) else upper.iloc[i]
        st.iloc[i]=lower.iloc[i] if trend.iloc[i]>0 else upper.iloc[i]
    d["SUPERTREND"]=st; d["SUPERTREND_DIR"]=trend
    return d


def swings(d: pd.DataFrame, radius: int=2) -> tuple[list[tuple[int,float]],list[tuple[int,float]]]:
    highs=[]; lows=[]
    for i in range(radius,len(d)-radius):
        if d.High.iloc[i]>=d.High.iloc[i-radius:i+radius+1].max(): highs.append((i,float(d.High.iloc[i])))
        if d.Low.iloc[i]<=d.Low.iloc[i-radius:i+radius+1].min(): lows.append((i,float(d.Low.iloc[i])))
    return highs,lows


def structure(d: pd.DataFrame) -> dict[str,Any]:
    hs,ls=swings(d); state="RANGE"; labels=[]
    for points,hi_name,lo_name in ((hs,"HH","LH"),(ls,"HL","LL")):
        for j in range(1,len(points)):
            labels.append({"idx":points[j][0],"price":points[j][1],"label":hi_name if points[j][1]>points[j-1][1] else lo_name})
    if len(hs)>=2 and len(ls)>=2:
        if hs[-1][1]>hs[-2][1] and ls[-1][1]>ls[-2][1]: state="BULLISH_HH_HL"
        elif hs[-1][1]<hs[-2][1] and ls[-1][1]<ls[-2][1]: state="BEARISH_LH_LL"
    close=float(d.Close.iloc[-1]); bos=None; choch=None
    if hs and close>hs[-1][1]: bos="BULLISH_BOS"; choch="BULLISH_CHOCH" if state.startswith("BEARISH") else None
    if ls and close<ls[-1][1]: bos="BEARISH_BOS"; choch="BEARISH_CHOCH" if state.startswith("BULLISH") else None
    return {"state":state,"bos":bos,"choch":choch,"labels":labels[-12:],"swing_highs":hs[-8:],"swing_lows":ls[-8:]}


def liquidity(d: pd.DataFrame) -> dict[str,Any]:
    hs,ls=swings(d); atr=_f(d.ATR.iloc[-1]); tol=max(atr*.2,_f(d.Close.iloc[-1])*.0005); eqh=[]; eql=[]
    for a,b in zip(hs,hs[1:]):
        if abs(a[1]-b[1])<=tol: eqh.append({"indices":[a[0],b[0]],"level":(a[1]+b[1])/2})
    for a,b in zip(ls,ls[1:]):
        if abs(a[1]-b[1])<=tol: eql.append({"indices":[a[0],b[0]],"level":(a[1]+b[1])/2})
    sweeps=[]
    for i in range(1,len(d)):
        prior_high=float(d.High.iloc[max(0,i-20):i].max()); prior_low=float(d.Low.iloc[max(0,i-20):i].min())
        if d.High.iloc[i]>prior_high and d.Close.iloc[i]<prior_high: sweeps.append({"idx":i,"type":"HIGH_SWEEP","level":prior_high})
        if d.Low.iloc[i]<prior_low and d.Close.iloc[i]>prior_low: sweeps.append({"idx":i,"type":"LOW_SWEEP","level":prior_low})
    return {"equal_highs":eqh[-5:],"equal_lows":eql[-5:],"sweeps":sweeps[-8:]}


def imbalances(d: pd.DataFrame) -> dict[str,Any]:
    fvg=[]
    for i in range(2,len(d)):
        if d.Low.iloc[i]>d.High.iloc[i-2]: fvg.append({"idx":i,"type":"BULLISH_FVG","low":float(d.High.iloc[i-2]),"high":float(d.Low.iloc[i]),"mitigated":bool((d.Low.iloc[i+1:]<=d.High.iloc[i-2]).any())})
        if d.High.iloc[i]<d.Low.iloc[i-2]: fvg.append({"idx":i,"type":"BEARISH_FVG","low":float(d.High.iloc[i]),"high":float(d.Low.iloc[i-2]),"mitigated":bool((d.High.iloc[i+1:]>=d.Low.iloc[i-2]).any())})
    obs=[]; atr=d.ATR
    for i in range(1,len(d)):
        impulse=abs(d.Close.iloc[i]-d.Open.iloc[i])>1.2*atr.iloc[i]
        if impulse and d.Close.iloc[i]>d.Open.iloc[i] and d.Close.iloc[i-1]<d.Open.iloc[i-1]: obs.append({"idx":i-1,"type":"BULLISH_OB","low":float(d.Low.iloc[i-1]),"high":float(d.High.iloc[i-1])})
        if impulse and d.Close.iloc[i]<d.Open.iloc[i] and d.Close.iloc[i-1]>d.Open.iloc[i-1]: obs.append({"idx":i-1,"type":"BEARISH_OB","low":float(d.Low.iloc[i-1]),"high":float(d.High.iloc[i-1])})
    return {"fair_value_gaps":fvg[-10:],"order_blocks":obs[-8:]}


def volume_profile(d: pd.DataFrame,bins:int=24) -> dict[str,float]:
    lo,hi=float(d.Low.min()),float(d.High.max())
    if hi<=lo:return {"poc":lo,"vah":hi,"val":lo}
    edges=np.linspace(lo,hi,bins+1); hist=np.zeros(bins)
    for _,r in d.iterrows():
        idx=min(bins-1,max(0,int((float((r.High+r.Low+r.Close)/3)-lo)/(hi-lo)*bins))); hist[idx]+=float(r.Volume)
    centers=(edges[:-1]+edges[1:])/2; poc=float(centers[int(hist.argmax())]); order=np.argsort(hist)[::-1]; chosen=[]; total=hist.sum(); acc=0
    for idx in order:
        chosen.append(idx); acc+=hist[idx]
        if acc>=total*.70:break
    return {"poc":poc,"vah":float(edges[max(chosen)+1]),"val":float(edges[min(chosen)])}


def patterns(d: pd.DataFrame) -> list[dict[str,Any]]:
    out=[]; r=d.iloc[-1]; p=d.iloc[-2]; body=abs(r.Close-r.Open); rng=max(r.High-r.Low,1e-12); upper=r.High-max(r.Open,r.Close); lower=min(r.Open,r.Close)-r.Low
    if body/rng<.1: out.append({"name":"DOJI","quality":round((1-body/rng)*100)})
    if lower>2*body and upper<body: out.append({"name":"HAMMER","quality":75})
    if upper>2*body and lower<body: out.append({"name":"SHOOTING_STAR","quality":75})
    if r.Close>r.Open and p.Close<p.Open and r.Open<=p.Close and r.Close>=p.Open: out.append({"name":"BULLISH_ENGULFING","quality":80})
    if r.Close<r.Open and p.Close>p.Open and r.Open>=p.Close and r.Close<=p.Open: out.append({"name":"BEARISH_ENGULFING","quality":80})
    return out


def snapshot(df:pd.DataFrame)->tuple[pd.DataFrame,dict[str,Any]]:
    d=add_indicators(df); s=structure(d); liq=liquidity(d); imb=imbalances(d); vp=volume_profile(d.iloc[-100:]); last=d.iloc[-1]
    snap={"indicators":{"rsi":_f(last.RSI),"stoch_rsi":_f(last.STOCH_RSI),"macd":_f(last.MACD),"macd_signal":_f(last.MACD_SIGNAL),"adx":_f(last.ADX),"atr":_f(last.ATR),"obv":_f(last.OBV),"vwap":_f(last.VWAP),"supertrend":_f(last.SUPERTREND),"supertrend_direction":int(last.SUPERTREND_DIR),"ichimoku_tenkan":_f(last.TENKAN),"ichimoku_kijun":_f(last.KIJUN)},"structure":s,"liquidity":liq,"imbalances":imb,"volume_profile":vp,"patterns":patterns(d),"data":{"candles":len(d),"last_closed_at":d.index[-1].isoformat()}}
    return d,snap


def position_size(capital:float,risk_pct:float,entry:float,stop:float)->dict[str,float]:
    if min(capital,risk_pct,entry,stop)<=0 or entry==stop: raise ValueError("Invalid positive values")
    risk_cash=capital*risk_pct/100; units=risk_cash/abs(entry-stop)
    return {"risk_cash":risk_cash,"units":units,"position_value":units*entry,"stop_distance_pct":abs(entry-stop)/entry*100}
