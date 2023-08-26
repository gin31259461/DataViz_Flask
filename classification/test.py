import sys
from pathlib import Path

print(Path.cwd())
sys.path.append(f"{Path.cwd().absolute()}/")
import pandas as pd

from setup import setup

db = setup()

df = pd.DataFrame(pd.date_range("2000-01-02", freq="1D", periods=15), columns=["Date"])

bins_dt = pd.date_range("2000-01-01", freq="3D", periods=6)
bins_str = bins_dt.astype(str).values

print(df)
print(bins_dt)
