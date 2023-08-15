import json
from pathlib import Path
from sqlalchemy import create_engine


def setup():
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
    db = create_engine(str, fast_executemany=True)
    return db
