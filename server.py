import ssl
import tempfile

import pandas as pd
from flask_cors import CORS
from pandas.core.api import DataFrame

from db import create_db_engine
from flask import Flask, request

server = Flask(__name__)
CORS(server)

# ! urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
# ! self-signed certificate in certificate chain (_ssl.c:xxxx)>
ssl._create_default_https_context = ssl._create_unverified_context

db = create_db_engine()


def data_to_sql(data: DataFrame, dataId):
    if not dataId:
        print("dataId is not valid")
        return

    df = pd.DataFrame(data)
    df.to_sql("D" + dataId, db, if_exists="replace", index=False, schema="dbo")


# * Form data
# * form:
#   dataId
#   url
# * files:
#   file
@server.route("/api/file_upload", methods=["POST"])
def file_upload():
    dataId = request.form["dataId"]
    if request.files:
        csvFile = request.files["file"]
        tempFile = tempfile.NamedTemporaryFile(delete=False)
        csvFile.save(tempFile.name)
        with open(tempFile.name, "rb") as f:
            data = pd.read_csv(f, encoding="utf-8", encoding_errors="ignore")
            print(data)
            data_to_sql(data, dataId)
    elif request.form["url"]:
        url = request.form["url"]
        data = pd.read_csv(url, encoding="utf-8", encoding_errors="ignore")
        print(data)
        data_to_sql(data, dataId)

    return "upload successfully"


@server.route("/api/get_column_types", methods=["GET"])
def get_column_types():
    pass


@server.route("/api/path_analysis", methods=["GET"])
def path_analysis():
    pass


@server.route("/api/pivot_analysis", methods=["GET"])
def pivot_analysis():
    pass


if __name__ == "__main__":
    server.run(debug=True, host="10.22.22.97", port=3090)
