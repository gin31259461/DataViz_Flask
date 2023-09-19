# %% [markdown]
# # Decision tree analysis for data discovery
# This program is used to perform analysis on data category separation, exploring more valuable information.

import json
# %%
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(f"{Path.cwd().parent.absolute()}/")
from setup import setup

db = setup()

# %% [markdown]
# ## Required parameters
# * OID (int): data object id in database
# * target (str): target column name
# * skip_features (list[str]): exclude feature from attribute (feature) list
# * datetime_format (str): if there has datetime column, use this format to parse datetime column

# %%
OID = 139
analysis_depth = 30
skip_features = []
datetime_format = ""
# target = "教育程度類別"
# target = "信用卡交易金額[新台幣]"
# target = "性別"
target = "產業別"
# target = "信用卡交易筆數"

# %% [markdown]
# ### Fetch data from database

# %%
query = text(f"SELECT * FROM [RawDB].[dbo].[D{OID}]")
data = db.execute(query)
query = text("SELECT * FROM [DV].[dbo].[Object] where OID = :OID")
object = db.execute(query, OID=OID).fetchall()
df = pd.DataFrame(data.fetchall())
origin_df = df.copy()
data_object = pd.DataFrame(object)
print(data_object["CName"])
print(df.shape)

df

# %% [markdown]
# ## Exploratory data analysis and feature engineering
# * Clean and pre-processing data
# * Split data to training sets (70% - 80%) and test sets
# * Feature engineering: category values are encoded and other suitable changes are made to the data
# * Predictive model is ready

# %% [markdown]
# ### Show all column of this data

# %%
column_names: list[str] = df.columns.to_list()
column_names

# %% [markdown]
# ### Check if has null value

# %%
df.isnull().sum()

# %% [markdown]
# ### Datetime column quantization

# %% [markdown]
# ##### Initial variables

# %%
from typing import Tuple, Type

datetime_format_list = ["%Y-%m-%d", "%Y-%m", "%Y%m%d", "%Y%m", "%Y"]
# [column_name, earliest_time, latest_time]
datetime_column_earliest_latest_tuple: Tuple[str, Type[pd.Timestamp], Type[pd.Timestamp]] = []

if len(datetime_format) != 0:
    datetime_format_list.insert(0, datetime_format)

# %% [markdown]
# #### Find datetime column, also record min and max value

# %%
# to_datetime reference: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html

for col in column_names:
    is_datetime = df[col].dtype == "datetime64[ns]"
    if is_datetime:
        datetime_column_earliest_latest_tuple.append(
            [
                col,
                df[col].min(),
                df[col].max(),
            ]
        )
        continue
    for test_format in datetime_format_list:
        is_numeric = df[col].dtype == "int64"
        is_datetime = is_numeric and (
            True
            not in pd.to_datetime(arg=df[col].astype("str"), format=test_format, errors="coerce")
            .isna()
            .value_counts()
            .index.to_list()
        )
        if is_datetime:
            parsed_datetime = pd.to_datetime(arg=df[col], format=test_format, errors="coerce")
            df[col] = parsed_datetime
            datetime_column_earliest_latest_tuple.append(
                [
                    col,
                    parsed_datetime.min(),
                    parsed_datetime.max(),
                ]
            )
            break

datetime_column_earliest_latest_tuple

# %% [markdown]
# #### Quantize found datetime column

# %%
# Quantize datetime problem reference:
# https://stackoverflow.com/questions/43500894/pandas-pd-cut-binning-datetime-column-series
quantile_mapping = {}
quantile_columns = []

for tuple in datetime_column_earliest_latest_tuple:
    # date_range reference: https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
    # Frequency reference: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    # 季度開始 -> QS
    # 年份開始 -> YS
    # TODO: 測試時間用哪個頻率進行離散化效果最好，季度 or 年度 or 月 etc.
    # * 逐一比較不同頻率的交叉驗證平均分數 (cross validation average score)
    # * 分數越高代表頻率更適合
    col = tuple[0]
    quantile_columns.append(col)
    datetime_range = pd.date_range(start=tuple[1], end=tuple[2], freq="QS")
    datetime_range = datetime_range.union([tuple[2]])
    labels = ["({}, {}]".format(datetime_range[i - 1], datetime_range[i]) for i in range(1, len(datetime_range))]
    quantile_interval = pd.cut(df[col], bins=datetime_range, include_lowest=True)
    df[col] = pd.cut(df[col], bins=datetime_range, labels=labels, include_lowest=True)
    quantile_mapping[col] = pd.Series(data=quantile_interval.unique().tolist(), index=df[col].unique().tolist())

# %% [markdown]
# ### Numerical column

# %% [markdown]
# #### Counting min and max value

# %%
# [column_name, minimum_value, maximum_value]
numerical_column_max_min_tuple: Tuple[str, int, int] = []
for col in column_names:
    is_numeric = df[col].dtype == "int64"
    is_category_column = len(df[col].unique()) <= 10
    if is_numeric and (not is_category_column):
        numerical_column_max_min_tuple.append([col, df[col].min(), df[col].max()])

numerical_column_max_min_tuple

# %% [markdown]
# ### Handle target and features

# %%
# 1. 處理不同型別的屬性 (類別型，數值型) -> 都可當作分析目標
# 2. 排除不想要的屬性，並且不加入 features (X) 裡面

# TODO: numeric quantile mapping
for col in column_names:
    column_ratio = len(df[col].unique()) / df[col].count()
    is_categorical_column = df[col].dtype == "object" or df[col].dtype == "category" or column_ratio < 0.01
    is_numerical_column = df[col].dtype == "int64" and not is_categorical_column

    if is_numerical_column:
        quantile_labels = ["low", "middle", "high"]
        discrete_bin_num = 3
        quantile = pd.qcut(df[col], q=discrete_bin_num)
        df[col] = pd.qcut(df[col], q=discrete_bin_num, labels=quantile_labels)
        quantile_columns.append(col)
        quantile_mapping[col] = pd.Series(data=quantile.unique().tolist(), index=df[col].unique().tolist())

X: pd.DataFrame
try:
    X = df.drop([target] + skip_features, axis=1)
except KeyError:
    print("Column of target or skip features are not exist in data frame")

feature_names = X.columns.tolist()

y = df[target].astype("string")
target_values = y.unique().tolist()
target_values

# %% [markdown]
# ### Encode category column of features

# %% [markdown]
# #### Initialization

# %%
import category_encoders as ce

# ! Prior knowledge
# TODO: 類別型欄位，是否有大小關係? ⇒ 讓使用者去決定這個順序, 可以利用 Python 的 category type
category_frame = X.select_dtypes(include=["object", "category"])
encoded_df = df.copy()

category_frame

# %%
encoder = ce.OrdinalEncoder(cols=category_frame.columns)
X = pd.DataFrame(encoder.fit_transform(X))
X.head()

# %% [markdown]
# #### Encoder mapping

# %%
category_column_mapping = encoder.mapping
category_column_mapping

# %% [markdown]
# ### Split data to training and test dataset
# Purpose: find dependencies between target and feature column

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %%
X_train.shape, X_test.shape

# %% [markdown]
# ## Fitting the model, evaluating the results and visualizing the trees
# * Data totally prepared
# * Classifier is instantiated
# * Model is fit onto the data
# * Ensure the model is neither over fitting and under fitting the data
# * Evaluate classifier: confusion matrix, precision score, f1 score, recall, support scores

# %% [markdown]
# ### Initial variables

# %%
from sklearn.tree import DecisionTreeClassifier

row_counts = len(X.index)
max_depth = analysis_depth
min_samples_split = 0
min_samples_leaf = 0

is_big_data = row_counts > 10000

if is_big_data:
    # 確保葉節點有足夠的樣本進行有意義的分析，同時避免過度細分
    # 100 - 1000
    min_samples_leaf = 100
    # 確保在分割內部節點之前有足夠的樣本數
    # 10 - 50
    min_samples_split = 10
else:
    # 確保每個葉節點至少有一些樣本進行分析
    # 1 or 2
    min_samples_leaf = 1
    # 確保在內部節點的樣本數較少時也可以進行分割
    # 2 - 5
    min_samples_split = 2

# %% [markdown]
# ### Fitting training data into decision tree classifier

# %%
clf = DecisionTreeClassifier(
    criterion="entropy",
    splitter="best",
    max_depth=max_depth,
    random_state=0,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
)
decision_tree = clf.fit(X_train, y_train)

# %% [markdown]
# ### Dependencies

# %% [markdown]
# #### Feature importance

# %%
# Pair importance and feature
feature_importance = clf.feature_importances_
feature_importance_pairs = list(zip(feature_names, feature_importance))
# Sort importance
feature_importance_pairs.sort(key=lambda pair: pair[1], reverse=True)
feature_importance_pairs

# %% [markdown]
# #### Cross validation

# %%
from sklearn.model_selection import cross_val_score

# cv: k-fold, default is 5-fold
cross_validation_score = cross_val_score(clf, X, y, cv=5)
print("交叉驗證分數:", cross_validation_score)
print("平均分數:", cross_validation_score.mean())

# %%
y_predict_test = clf.predict(X_test)
y_predict_test

# %%
y_predict_train = clf.predict(X_train)
y_predict_train

# %% [markdown]
# #### Model accuracy

# %%
from sklearn.metrics import accuracy_score

print("Training set score: {:.4f}".format(accuracy_score(y_train, y_predict_train)))
print("Test set score: {0:0.4f}".format(accuracy_score(y_test, y_predict_test)))

# %% [markdown]
# ### Resolve decision tree structure to json

from graphviz import Source
# %%
from sklearn import tree

output_file_path = f"{Path.cwd().absolute()}/temp/temp.dot"

# %% [markdown]
# #### Types definitions

# %%
from dataclasses import dataclass


@dataclass
class DecisionTreeNode:
    id: int
    labels: list[str]


@dataclass
class DecisionTreeEdge:
    id: int
    label: str
    head: int
    tail: int


@dataclass
class DecisionTreeGraph:
    nodes: list[DecisionTreeNode]
    edges: dict[str, DecisionTreeEdge]  # node1_node2 as key value


@dataclass
class DecisionTreePath:
    path: list[int]
    nodeLabel: dict[int, list[str]]

# %% [markdown]
# #### Export decision tree from model and reconstruct DecisionTreeGraph

# %%
# The analysis goal is discovering data, not just training model
# ! re-fit the hole data (target and features, X and y), not splitted data

decision_tree = clf.fit(X, y)

# * Scikit-learn decision tree:
# Using optimized version of the CART algorithm
# Not support categorical variable for now, that is, categorical variable need to encode

# * Entropy range:
# From 0 to 1 for binary classification (target has only two classes, true or false)
# From 0 to log base 2 k where k is the number of classes

dotData = tree.export_graphviz(
    clf,
    out_file=output_file_path,
    feature_names=feature_names,
    class_names=target_values,
    max_depth=max_depth,
    label="all",
    rounded=True,
    filled=True,
)

with open(output_file_path, "r", encoding="utf-8") as f:
    dotData = f.read()

# Use graphviz lib to convert dot format to json format
source = Source(dotData)
json_graph = source.pipe(format="json").decode("utf-8")
dict_graph: dict = json.loads(json_graph)

# Filter needed part
nodes = list(
    map(
        lambda o: {"id": o.get("_gvid"), "labels": o.get("label").split("\\n")},
        dict_graph.get("objects"),
    )
)

edges = dict(
    map(
        lambda o: (
            str(o.get("tail")) + "_" + str(o.get("head")),
            {
                "id": o.get("_gvid"),
                "label": o.get("headlabel"),
                "head": o.get("tail"),
                "tail": o.get("head"),
            },
        ),
        dict_graph.get("edges"),
    )
)

# %% [markdown]
# #### Store information

# %%
data_information: dict[str, str or list or dict] = {}
data_information["target_name"] = target
data_information["target_values"] = target_values
data_information["feature_names"] = feature_names

# Numeric
feature_values: dict[str, dict[str, str or list]] = {}

for n in numerical_column_max_min_tuple:
    feature_values[n[0]] = {"type": "numeric", "value": [n[1], n[2]]}

# Datetime
for d in datetime_column_earliest_latest_tuple:
    format = "%Y-%m-%d %X"
    feature_values[d[0]] = {
        "type": "datetime",
        "value": [d[1].strftime(format), d[2].strftime(format)],
    }

# Category
for c in category_column_mapping:
    is_datetime_column = c["col"] in quantile_columns
    feature_values[c["col"]] = {
        "type": "datetime" if is_datetime_column else "category",
        "value": (c["mapping"].index.to_list()),
        "mapping": pd.Series(dict((v, k) for k, v in c["mapping"].items())),
    }
    feature_values[c["col"]]["value"].pop()

unstored_features = list(set(feature_names) - set(list(feature_values.keys())))

# TODO: Custom mapping => 使用者指定 1 男, 2 女
for f in unstored_features:
    split_value = df[f].astype("string").unique().tolist()
    mapping_pairs = dict((i + 1, split_value[i]) for i in range(len(split_value)))
    mapping_pairs[-2] = "nan"
    mapping = pd.Series(mapping_pairs)
    feature_values[f] = {
        "type": "category",
        "value": split_value,
        "mapping": mapping,
    }

data_information["feature_values"] = feature_values

data_information

# %% [markdown]
# ### Decision tree path parser

# %%
from math import log


def DecisionTreePathParser(graph: DecisionTreeGraph, root_id: int = 0):
    paths: list[DecisionTreePath] = []

    # DFS: Depth-First Search
    def SearchPathByDFS(current_id: int = 0, path: list[int] = []):
        if not graph:
            return

        path.append(current_id)

        edge_values = list(map(lambda edge: DecisionTreeEdge(**edge), list(graph.edges.values())))
        outgoing_edges = list(filter(lambda edge: edge.head == current_id, edge_values))

        # 如果目前節點沒有出邊（即為最底層節點），將路徑加入結果中
        if len(outgoing_edges) == 0:
            last_id = path[len(path) - 1]
            last_node = DecisionTreeNode(**(graph.nodes[last_id]))
            node_labels: dict[int, list[str]] = {}

            # ! 排除 entropy 高的 path
            entropy = float(last_node.labels[0].split(" ")[2])

            if entropy > log(len(feature_names), 2) / 2:
                path.pop()
                return
            # ! ####################

            node_labels[last_id] = last_node.labels

            for i in range(0, len(path) - 1):
                node_id = path[i]
                labels = DecisionTreeNode(**(graph.nodes[node_id])).labels

                # 如果下一個的 node id 是上一個 +1 則是 true，不然的話是 false
                # * left edge <= => true
                # * right edge > => false

                next_id = path[i + 1]

                if node_id + 1 != next_id:
                    new_labels = [*labels]
                    condition = new_labels[0]
                    split_condition = condition.split(" ")
                    split_condition[1] = ">"
                    new_labels[0] = " ".join(split_condition)
                    node_labels[node_id] = new_labels
                    continue

                node_labels[node_id] = [*labels]

            paths.append(DecisionTreePath([*path], node_labels))

        # 遍歷目前節點的所有出邊
        else:
            for edge in outgoing_edges:
                next_id = edge.tail

                # 遞迴呼叫深度優先搜索
                SearchPathByDFS(next_id, path)

        # 回溯，從路徑中移除目前節點
        path.pop()

    SearchPathByDFS(root_id)

    return paths


decision_tree_graph = DecisionTreeGraph(nodes, edges)
paths = DecisionTreePathParser(decision_tree_graph, 0)
print("Path counts = {}".format(len(paths)))

# %% [markdown]
# ### Decision tree path analyzer

# %%
from math import ceil, floor


def DecisionTreePathAnalyzer(paths: list[DecisionTreePath], target_values: list[str], feature_names: list[str]):
    path_analysis_result: dict = {}
    for split_value in target_values:
        path_analysis_result[split_value] = []

    for path in paths:
        path_analysis_result_part = {}

        for feature_name in feature_names:
            path_analysis_result_part[feature_name] = data_information["feature_values"][feature_name]["value"].copy()

        for node_id in path.path:
            labels = path.nodeLabel[node_id][0].split(" ")
            feature_name = labels[0]
            split_symbol = labels[1]
            split_value = float(labels[2])

            if node_id == path.path[len(path.path) - 1]:
                class_name = path.nodeLabel[node_id][3].split(" ")[2]

                sample_value = " ".join(path.nodeLabel[node_id][2].split(" ")[2:]).split(", ")
                sample_value[0] = sample_value[0][1:]
                sample_value[len(sample_value) - 1] = sample_value[len(sample_value) - 1][0:-1]

                path_analysis_result_part["entropy"] = float(split_value)
                path_analysis_result_part["samples"] = list(map(lambda value: int(value), sample_value))
                path_analysis_result_part["labels"] = target_values
                path_analysis_result_part["class"] = class_name

                path_analysis_result[class_name].append(path_analysis_result_part)
                break

            feature_type = data_information["feature_values"][labels[0]]["type"]
            split_situation = [split_symbol, feature_type]

            match split_situation:
                case ["<=", "category"]:
                    mapping: pd.Series = data_information["feature_values"][feature_name]["mapping"]
                    filter_values = mapping.drop(-2).loc[1 : floor(split_value)].tolist()
                    path_analysis_result_part[feature_name] = filter_values
                case ["<=", "datetime"]:
                    mapping: pd.Series = data_information["feature_values"][feature_name]["mapping"]
                    filter_values = mapping.drop(-2).loc[1 : floor(split_value)].tolist()
                    path_analysis_result_part[feature_name] = filter_values
                case ["<=", "numeric"]:
                    if split_value < path_analysis_result_part[feature_name][1]:
                        path_analysis_result_part[feature_name][1] = split_value
                case [">", "category"]:
                    mapping: pd.Series = data_information["feature_values"][feature_name]["mapping"]
                    filter_values = mapping.drop(-2).loc[ceil(split_value) :].tolist()
                    path_analysis_result_part[feature_name] = filter_values
                case [">", "datetime"]:
                    mapping: pd.Series = data_information["feature_values"][feature_name]["mapping"]
                    filter_values = mapping.drop(-2).loc[ceil(split_value) :].tolist()
                    path_analysis_result_part[feature_name] = filter_values
                case [">", "numeric"]:
                    if split_value > path_analysis_result_part[feature_name][0]:
                        path_analysis_result_part[feature_name][0] = split_value
                case _:
                    print("no match case")

    return path_analysis_result


path_analysis_result = DecisionTreePathAnalyzer(paths=paths, target_values=target_values, feature_names=feature_names)

path_analysis_result

# %% [markdown]
# ### Final result

# %%
json_str = json.dumps(path_analysis_result)
json_object = json.loads(json_str)

for k in json_object:
    print(k, len(json_object[k]))

# %%
df

# %%
X

# %%
y

# %%
data_information

# %%
print(quantile_columns)
quantile_mapping

# %%
json_object

# %% [markdown]
# ### Tests

# %%
# encoded_df.loc[
#     (df["教育程度類別"] == "博士")
#     & (df["年月"] == "(2022-10-01 00:00:00, 2023-01-01 00:00:00]")
#     & (df["信用卡交易筆數"] == "high")
#     & (df["信用卡交易金額[新台幣]"] == "high")
#     & (df["性別"] == 1)
# ]

# pd.DataFrame([origin_df.loc[56276], origin_df.loc[56318], origin_df.loc[56948], origin_df.loc[57620]])

encoded_df.loc[
    (df["年月"] == "(2020-04-01 00:00:00, 2020-07-01 00:00:00]")
    & (df["教育程度類別"] == "大學")
    & (df["信用卡交易筆數"] == "low")
    & (df["產業別"] == "百貨")
    & (df["信用卡交易金額[新台幣]"] == "low")
]


