import json
from pathlib import Path

from pandas import DataFrame
from sqlalchemy import Engine, create_engine


def create_db_engine():
    f = open(f"{Path(__file__).parent.absolute()}/mssql.json")
    jos = json.load(f)
    f.close()
    conf = jos[0]["local"]
    drv = conf["driver"]
    uid = conf["uid"]
    pwd = conf["pwd"]
    srv = conf["server"]
    ins = conf["instance"]
    pno = conf["port"]
    db = conf["db"]
    str = f"mssql+pyodbc://{uid}:{pwd}@{srv}{ins}:{pno}/{db}?driver={drv}"

    return create_engine(str, fast_executemany=True)


def data_to_sql(db: Engine, data: DataFrame, name: str):
    df = DataFrame(data)
    df.to_sql(name, db, if_exists="replace", index=False, schema="dbo")
