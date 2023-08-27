# %% [markdown]
# # Decision tree analysis for data discovery
# This program is used to perform analysis on data category separation, exploring more valuable information.

import json
# %%
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

sys.path.append(f"{Path.cwd().absolute()}/")
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
skip_features = []
datetime_format = ""
target = "教育程度類別"
# target = "信用卡交易金額[新台幣]"
# target = "性別"
# target = "產業別"
# target = "信用卡交易筆數"

# %% [markdown]
# ### Fetch data from database

# %%
query = text(f"SELECT * FROM [RawDB].[dbo].[D{OID}]")
data = db.execute(query)
query = text("SELECT * FROM [DV].[dbo].[Object] where OID = :OID")
object = db.execute(query, OID=OID).fetchall()
df = pd.DataFrame(data.fetchall())
data_object = pd.DataFrame(object)
print(df.shape)
df.head()

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
datetime_format_list = ["%Y-%m-%d", "%Y-%m", "%Y%m%d", "%Y%m", "%Y"]
# [column_name, earliest_time, latest_time]
datetime_column_earliest_latest_tuple: list[str, pd.Timestamp, pd.Timestamp] = []

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
            not True
            in pd.to_datetime(arg=df[col].astype("str"), format=test_format, errors="coerce")
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
# Quantize datetime problem reference: https://stackoverflow.com/questions/43500894/pandas-pd-cut-binning-datetime-column-series
# ! this section can only execute once
for tuple in datetime_column_earliest_latest_tuple:
    # date_range reference: https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
    # Frequency reference: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
    # 季度開始 -> QS
    # 年份開始 -> YS
    datetime_range = pd.date_range(start=tuple[1], end=tuple[2], freq="QS")
    datetime_range = datetime_range.union([tuple[2]])
    labels = ["({}, {}]".format(datetime_range[i - 1], datetime_range[i]) for i in range(1, len(datetime_range))]
    # ! this line of code can only cut once
    df[tuple[0]] = pd.cut(df[tuple[0]], bins=datetime_range, labels=labels, include_lowest=True)

# %% [markdown]
# ### Numerical column

# %% [markdown]
# #### Counting min and max value

# %%
# [column_name, minimum_value, maximum_value]
numerical_column_max_min_tuple = []
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
# * 針對路徑中沒被用到的屬性
#   1. 類別型: 代表所有類別
#   2. 數值型: 代表在 MIN - MAX 區間
#   3. 時間型: 代表在最早與最晚的區間內

# Count target column ratio to determine its data type
target_column_ratio = len(df[target].unique()) / df[target].count()
# if not, it's numeric
is_category_column = df[target].dtype == "object" or target_column_ratio < 0.01
X: pd.DataFrame
try:
    X = df.drop([target] + skip_features, axis=1)
except:
    print("Column of target or skip features not exist in data frame")
feature_names = X.columns.to_list()
# If value of target column are numeric, divide it into multiple intervals (discretize)
quantization_labels = ["low", "middle", "high"]
discrete_bin_num = 3
y = (
    df[target].astype("string")
    if is_category_column
    else pd.qcut(df[target], q=discrete_bin_num, labels=quantization_labels)
)
target_class_names = y.unique().tolist()
target_class_names

# %% [markdown]
# ### Encode category column of features

# %% [markdown]
# #### Initialization

# %%
import category_encoders as ce

category_frame = X.select_dtypes(include=["object", "category"])
category_frame.head()

# %%
encoder = ce.OrdinalEncoder(cols=category_frame.columns)
X = pd.DataFrame(encoder.fit_transform(X))
X.head()

# %% [markdown]
# #### Encoder mapping

# %%
encoder.mapping

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
max_depth = 10
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

output_file_path = f"{Path.cwd().absolute()}/classification/temp/temp.dot"

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
    # TODO: What information can be record?
    # * target
    # * features
    attributes: dict[str, object]  # information of attributes


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

dotData = tree.export_graphviz(
    clf,
    out_file=output_file_path,
    feature_names=feature_names,
    class_names=target_class_names,
    max_depth=max_depth,
    label="all",
    rounded=True,
    filled=True,
)

with open(output_file_path, "r", encoding="utf-8") as f:
    dotData = f.read()

# Use graphviz lib to convert dot format to json format
source = Source(dotData)
jsonGraph = source.pipe(format="json").decode("utf-8")
dictGraph: dict = json.loads(jsonGraph)

# Filter needed part
nodes = list(
    map(
        lambda o: {"id": o.get("_gvid"), "labels": o.get("label").split("\\n")},
        dictGraph.get("objects"),
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
        dictGraph.get("edges"),
    )
)

decision_tree_graph = DecisionTreeGraph(nodes, edges, {})

# %% [markdown]
# ### Decision tree path parser


# %%
def DecisionTreePathParser(graph: DecisionTreeGraph, root_id: int = 0):
    paths: list[DecisionTreePath] = []

    # DFS: Depth-First Search
    def dfs(current_id: int = 0, path: list[int] = []):
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
            # TODO: 排除 entropy 高的 path
            node_labels[last_id] = last_node.labels

            for i in range(0, len(path) - 1):
                node_id = path[i]
                labels = DecisionTreeNode(**(graph.nodes[node_id])).labels
                # 如果下一個 node id 是上個 +1 則是 true，不然的話是 false
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
                dfs(next_id, path)

        # 回溯，從路徑中移除目前節點
        path.pop()

    dfs(root_id)

    return paths


paths = DecisionTreePathParser(decision_tree_graph, 0)
print("Path counts = {:}".format(len(paths)))

# %% [markdown]
# ### Decision tree path analyzer


# %%
# TODO: decision tree path analyzer
def DecisionTreePathAnalyzer():
    pass


# %% [markdown]
# ### Path data to JSON string

# %%
paths_json_str = json.dumps(list(map(lambda path: path.__dict__, paths)))
paths_object = json.loads(paths_json_str)

# %%
paths_object
