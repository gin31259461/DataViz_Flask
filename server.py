import json
import ssl
import tempfile

from flask import Flask, request
from flask_cors import CORS
from pandas import DataFrame, read_csv
from sqlalchemy import text

from analysis.path_analysis import PathAnalysis
from analysis.pivot_analysis import PivotAnalysis
from db import create_db_engine, data_to_sql

server = Flask(__name__)
CORS(server)

# ! urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
# ! self-signed certificate in certificate chain (_ssl.c:xxxx)>
ssl._create_default_https_context = ssl._create_unverified_context

db_engine = create_db_engine()


# * Formdata
# * form:
# dataId
# url
# * files:
# file
@server.route("/api/file_upload", methods=["POST"])
def file_upload():
    dataId = request.form["dataId"]
    if request.files:
        csvFile = request.files["file"]
        tempFile = tempfile.NamedTemporaryFile(delete=False)
        csvFile.save(tempFile.name)
        with open(tempFile.name, "rb") as f:
            data = read_csv(f, encoding="utf-8", encoding_errors="ignore")
            print(data)
            data_to_sql(data, "D" + dataId)
    elif request.form["url"]:
        url = request.form["url"]
        data = read_csv(url, encoding="utf-8", encoding_errors="ignore")
        print(data)
        data_to_sql(data, "D" + dataId)

    return "upload successfully"


# * Params
# dataId
@server.route("/api/get_data_info", methods=["GET"])
def get_data_info():
    dataId = request.args.get("dataId")
    query = text(f"SELECT * FROM [RawDB].[dbo].[D{dataId}]")
    data = None

    with db_engine.connect() as connection:
        cursor_result = connection.execute(query, {"OID": dataId})
        data = DataFrame(cursor_result.fetchall())

    data_info = {}
    for column in data.columns.tolist():
        data_info[column] = {}
        data_info[column]["type"] = data[column].dtype.name

        if data_info[column]["type"] == "float64" or data_info[column]["type"] == "int64":
            data_info[column]["values"] = len(data[column].unique())
        else:
            data_info[column]["values"] = data[column].unique().tolist()

    return data_info


# ! 關於分析 API 的參數傳遞
# * 分析的某些參數需要給 JSON string
# 1. 呼叫此 API 時需要先將 JSON 格式的參數轉成字串。
# 2. 然後再轉成 URI 格式。
# 3. 最後放到 URL 的參數裡面


# * Params
# dataId
# target
# skip_features
# skip_values
# concept_hierarchy
@server.route("/api/path_analysis", methods=["GET"])
def path_analysis():
    dataId = request.args.get("dataId")
    target = request.args.get("target")
    skip_features = request.args.get("skip_features")
    skip_values = request.args.get("skip_values")
    concept_hierarchy = request.args.get("concept_hierarchy")

    if dataId is None or target is None:
        return {}

    path = PathAnalysis(
        dataId=dataId,
        db=db_engine,
        target=target,
    )

    if skip_features is not None:
        path.skip_features = json.loads(json.loads(skip_features))

    if skip_values is not None:
        path.skip_values = json.loads(json.loads(skip_values))

    if concept_hierarchy is not None:
        path.concept_hierarchy = json.loads(json.loads(concept_hierarchy))

    path.analysis_pipeline()

    return path.result


# * Params
# dataId
# target
# process
@server.route("/api/process_pivot_analysis", methods=["GET"])
def process_pivot_analysis():
    dataId = request.args.get("dataId")
    target = request.args.get("target")
    process = request.args.get("process")

    pivot = PivotAnalysis(dataId=dataId, db=db_engine)

    if process is None:
        return {}

    pivot.process_pivot_data(json.loads(json.loads(process)), target)

    return pivot.process_result


if __name__ == "__main__":
    server.run(debug=True, host="10.22.22.97", port=3090)
