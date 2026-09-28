import numpy as np
import pandas as pd
from bot import scan_breakout_history


def test_history_uses_close_time_and_suppresses_continuing_breakouts():
    idx=pd.date_range("2026-01-01",periods=30,freq="4h",tz="UTC")
    close=np.array([100.0]*21+[110,112,114,116,118,120,122,124,126],dtype=float)
    df=pd.DataFrame({"Open":close-2,"High":close+1,"Low":close-1,"Close":close,"Volume":[100]*21+[500]*9},index=idx)
    events=scan_breakout_history(df,timeframe="4h")
    confirmed=[e for e in events if e["state"]=="BREAKOUT_CONFIRMED"]
    assert len(confirmed)==1
    assert pd.Timestamp(confirmed[0]["close_time"])==pd.Timestamp(confirmed[0]["time"])+pd.Timedelta(hours=4)
