import sys
from pathlib import Path

print(Path.cwd())
sys.path.append(f"{Path(Path.cwd()).parent.absolute()}/")
import pandas as pd
from sqlalchemy import text

from setup import setup

db = setup()
result = db.execute(text("select * from RawDB.dbo.D139"))
df = pd.DataFrame(result.fetchall(), columns=result.keys())

print(df)
