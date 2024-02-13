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
@server.route("/api/upload_file", methods=["POST"])
def upload_file():
    dataId = request.form.get("dataId")
    file = request.files.get("file")
    url = request.form.get("url")

    if dataId is None:
        return {"message": "Invalid data id"}

    if file is not None:
        tempFile = tempfile.NamedTemporaryFile(delete=False)
        file.save(tempFile.name)
        with open(tempFile.name, "rb") as f:
            data = read_csv(f, encoding="utf-8", encoding_errors="ignore")
            print(data)
            data_to_sql(db_engine, data, "D" + dataId)
    elif url is not None:
        url = request.form["url"]
        data = read_csv(url, encoding="utf-8", encoding_errors="ignore")
        print(data)
        data_to_sql(db_engine, data, "D" + dataId)

    return {"message": "upload file done"}


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

    data_info = {"columns": {}, "info": {}}

    data_info["info"]["id"] = dataId
    data_info["info"]["rows"] = data.size

    with db_engine.connect() as connection:
        query = text("SELECT CName, CDes FROM [DV].[dbo].[Object] where OID = :OID")
        object_table = DataFrame(connection.execute(query, {"OID": dataId}).fetchall())
        data_name = object_table["CName"][0]
        data_des = object_table["CDes"][0]

        data_info["info"]["name"] = data_name
        data_info["info"]["des"] = data_des

    for column in data.columns.tolist():
        data_info["columns"][column] = {}
        col_type = data[column].dtype.name

        match col_type:
            case "float64":
                data_info["columns"][column]["type"] = "float"
                data_info["columns"][column]["values"] = len(data[column].unique())
            case "int64":
                data_info["columns"][column]["type"] = "number"
                data_info["columns"][column]["values"] = len(data[column].unique())
            case "object":
                data_info["columns"][column]["type"] = "string"
                data_info["columns"][column]["values"] = data[column].unique().tolist()

    return data_info


# * Data
# dataId
# target
# skip_features
# skip_values
# concept_hierarchy
@server.route("/api/path_analysis", methods=["POST"])
def path_analysis():
    data: dict = request.get_json()

    dataId = data.get("dataId")
    target = data.get("target")
    skip_features = data.get("skip_features")
    skip_values = data.get("skip_values")
    concept_hierarchy = data.get("concept_hierarchy")

    if dataId is None or target is None:
        return {}

    path = PathAnalysis(
        dataId=dataId,
        db=db_engine,
        target=target,
    )

    if skip_features is not None:
        path.skip_features = skip_features

    if skip_values is not None:
        path.skip_values = skip_values

    if concept_hierarchy is not None:
        path.concept_hierarchy = concept_hierarchy

    path.analysis_pipeline()

    return path.result


# * Data
# dataId
# target
# process
@server.route("/api/process_pivot_analysis", methods=["POST"])
def process_pivot_analysis():
    data: dict = request.get_json()

    dataId = data.get("dataId")
    target = data.get("target")
    process = data.get("process")

    pivot = PivotAnalysis(dataId=dataId, db=db_engine)

    if process is None:
        return {}

    pivot.process_pivot_data(process, target)

    return pivot.process_result


# * Data
# dataId
# index
# values
# columns
# focus_index
# focus_columns
@server.route("/api/pivot_table", methods=["GET"])
def pivot_table():
    pass


if __name__ == "__main__":
    server.run(debug=True, host="10.22.22.97", port=3090)
