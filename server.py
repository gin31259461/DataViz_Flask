import json
import tempfile

import pandas as pd
from flask_cors import CORS
from pandas.core.api import DataFrame
from sqlalchemy import text

from classification.decisionTree import decisionTreeHandler
from flask import Flask, request
from setup import setup

server = Flask(__name__)
CORS(server)


db = setup()


def data_to_sql(data: DataFrame, lastID="0"):
    df = pd.DataFrame(data)
    df.to_sql("D" + lastID, db, if_exists="replace", index=False, schema="dbo")


# Argument: form data
# form:
#   lastID
#   url
# files:
#   file
@server.route("/api/upload", methods=["POST"])
def upload_router():
    lastID = request.form["lastID"]
    if request.files:
        csvFile = request.files["file"]
        tempFile = tempfile.NamedTemporaryFile(delete=False)
        csvFile.save(tempFile.name)
        with open(tempFile.name, "rb") as f:
            data = pd.read_csv(f, encoding="utf-8")
            print(data)
            data_to_sql(data, lastID)
    elif request.form["url"]:
        url = request.form["url"]
        data = pd.read_csv(url, encoding="utf-8")
        print(data)
        data_to_sql(data, lastID)

    return "upload successfully"


# Argument:
# oid: string
# target: string
# features: string,string,...
@server.route("/api/decision_tree", methods=["GET"])
def decision_tree_router():
    result = db.execute(text("select * from RawDB.dbo.D" + request.args.get("oid")))
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
    result = decisionTreeHandler(df, request.args.get("target"), request.args.get("features").split(","))
    print(result)
    return json.dumps(result)


if __name__ == "__main__":
    server.run(debug=True, host="127.0.0.1", port=3090)
