import numpy as np
import pandas as pd
from pro_quant import add_indicators, position_size, snapshot
from pro_features import TTLCache, data_quality, event_fingerprint, retest_state


def candles(n=240):
    idx=pd.date_range("2025-01-01",periods=n,freq="h",tz="UTC")
    close=np.linspace(100,130,n)+np.sin(np.arange(n)/5)
    return pd.DataFrame({"Open":close-.2,"High":close+.8,"Low":close-.8,"Close":close,"Volume":np.linspace(1000,1600,n)},index=idx)


def test_indicators_and_snapshot_are_finite():
    d,s= snapshot(candles())
    assert 0 <= s["indicators"]["rsi"] <= 100
    assert s["data"]["candles"] == 240
    assert "volume_profile" in s and "structure" in s


def test_position_size():
    r=position_size(1000,1,100,95)
    assert r["risk_cash"] == 10
    assert r["units"] == 2
    assert r["position_value"] == 200


def test_retest_held():
    d=add_indicators(candles())
    level=float(d.Close.iloc[-2])-0.1
    b={"state":"BREAKOUT_CONFIRMED","bullish_trigger":level,"bearish_trigger":90}
    assert retest_state(d,b)["state"] in {"RETEST_HELD","WAITING_FOR_RETEST"}


def test_quality_and_fingerprint_stable():
    q=data_quality(150,{"structure":"bullish"},{"funding_rate":0.0,"news_status":"live_rss"},{"risk_on":True})
    assert q["score"] == 100
    assert event_fingerprint("BTC/USDT","4h","x","BREAKOUT",10)==event_fingerprint("BTC/USDT","4h","x","BREAKOUT",10)
