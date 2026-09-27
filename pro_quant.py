"""Free professional features: cache, retest state, scoring and quality."""
from __future__ import annotations
import asyncio, hashlib, json, time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

class TTLCache:
    def __init__(self, maxsize:int=512): self.maxsize=maxsize; self._data={}; self._locks={}
    def get(self,key:str):
        item=self._data.get(key)
        if not item:return None
        expiry,value=item
        if expiry<=time.monotonic(): self._data.pop(key,None); return None
        return value
    def set(self,key:str,value:Any,ttl:float):
        if len(self._data)>=self.maxsize:
            for k in sorted(self._data,key=lambda x:self._data[x][0])[:max(1,self.maxsize//10)]: self._data.pop(k,None)
        self._data[key]=(time.monotonic()+ttl,value)
    async def get_or_create(self,key:str,ttl:float,factory:Callable[[],Awaitable[Any]]):
        hit=self.get(key)
        if hit is not None:return hit
        lock=self._locks.setdefault(key,asyncio.Lock())
        async with lock:
            hit=self.get(key)
            if hit is not None:return hit
            value=await factory(); self.set(key,value,ttl); return value


def retest_state(df, breakout:dict[str,Any], tolerance_atr:float=.35)->dict[str,Any]:
    """Classify latest closed candle around a previously derived range trigger."""
    if len(df)<3:return {"state":"NO_RETEST_DATA"}
    last=df.iloc[-1]; prev=df.iloc[-2]; atr=float(df.ATR.iloc[-1]); tol=max(atr*tolerance_atr,float(last.Close)*.001)
    up=float(breakout["bullish_trigger"]); down=float(breakout["bearish_trigger"]); state=breakout["state"]
    result={"state":"NOT_APPLICABLE","level":None,"distance":None}
    if state=="BREAKOUT_CONFIRMED" or prev.Close>up:
        touched=last.Low<=up+tol; held=last.Close>up
        result={"state":"RETEST_HELD" if touched and held else ("RETEST_FAILED" if last.Close<up-tol else "WAITING_FOR_RETEST"),"level":up,"distance":float(last.Close-up)}
    elif state=="BREAKDOWN_CONFIRMED" or prev.Close<down:
        touched=last.High>=down-tol; held=last.Close<down
        result={"state":"RETEST_HELD" if touched and held else ("RETEST_FAILED" if last.Close>down+tol else "WAITING_FOR_RETEST"),"level":down,"distance":float(last.Close-down)}
    return result


def explainable_score(q:dict[str,Any], higher:dict[str,Any], market:dict[str,Any], external:dict[str,Any])->dict[str,Any]:
    b=q["breakout"]; bullish={}
    bullish["structure"]=16 if q["structure"].startswith("bullish") else (5 if q["structure"]=="range" else 0)
    bullish["momentum"]=13 if q["rsi"]>=50 and q["macd"]>q["macd_signal"] else 5
    bullish["volume"]=15 if b["volume_ratio"]>=1.5 else (10 if b["volume_ratio"]>=1.1 else 4)
    bullish["volatility"]=10 if b["squeeze"] else 5
    bullish["higher_timeframe"]=15 if str(higher.get("structure","")).startswith("bullish") else 5
    bullish["market_filter"]=12 if market.get("risk_on") else 4
    funding=external.get("funding_rate"); bullish["futures"]=8 if funding is not None and abs(funding)<.001 else 4
    bullish["liquidity_risk"]=6 if not q.get("professional",{}).get("liquidity",{}).get("sweeps") else 3
    total=min(100,sum(bullish.values()))
    conflicts=[]
    if q["structure"].startswith("bullish") and q["rsi"]>70:conflicts.append("Bullish structure কিন্তু RSI overbought")
    if q["macd"]>q["macd_signal"] and b["volume_ratio"]<1:conflicts.append("Momentum bullish কিন্তু volume দুর্বল")
    if q["structure"].startswith("bullish") and not market.get("risk_on",True):conflicts.append("Coin bullish কিন্তু BTC/ETH filter দুর্বল")
    return {"bullish_total":total,"bearish_total":100-total,"components":bullish,"conflicts":conflicts}


def data_quality(frame_len:int,higher:dict,external:dict,market:dict)->dict[str,Any]:
    items={"spot_candles":frame_len>=100,"higher_timeframe":higher.get("status")!="unavailable",
           "futures":external.get("funding_rate") is not None,"news":external.get("news_status")=="live_rss",
           "btc_eth_filter":bool(market)}
    score=round(sum(items.values())/len(items)*100)
    return {"score":score,"sources":items,"label":"High" if score>=80 else ("Moderate" if score>=50 else "Low")}


def event_fingerprint(symbol:str,timeframe:str,candle:str,state:str,level:float)->str:
    raw=json.dumps([symbol,timeframe,candle,state,round(level,10)],separators=(",",":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:24]
