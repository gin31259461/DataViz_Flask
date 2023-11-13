import ssl
import tempfile

import pandas as pd
from flask_cors import CORS
from pandas.core.api import DataFrame

from flask import Flask, request
from setup import setup

server = Flask(__name__)
CORS(server)

# * Solve error
# urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
# self-signed certificate in certificate chain (_ssl.c:xxxx)>
ssl._create_default_https_context = ssl._create_unverified_context

db = setup()


def data_to_sql(data: DataFrame, lastID="0"):
    df = pd.DataFrame(data)
    df.to_sql("D" + lastID, db, if_exists="replace", index=False, schema="dbo")


# * Argument: form data
# * form:
#   lastID
#   url
# * files:
#   file
@server.route("/api/upload", methods=["POST"])
def upload_router():
    lastID = request.form["lastID"]
    if request.files:
        csvFile = request.files["file"]
        tempFile = tempfile.NamedTemporaryFile(delete=False)
        csvFile.save(tempFile.name)
        with open(tempFile.name, "rb") as f:
            data = pd.read_csv(f, encoding="utf-8", encoding_errors="ignore")
            print(data)
            data_to_sql(data, lastID)
    elif request.form["url"]:
        url = request.form["url"]
        data = pd.read_csv(url, encoding="utf-8", encoding_errors="ignore")
        print(data)
        data_to_sql(data, lastID)

    return "upload successfully"


@server.route("/api/analysis", methods=["GET"])
def analysis():
    pass


if __name__ == "__main__":
    server.run(debug=True, host="10.22.22.97", port=3090)
