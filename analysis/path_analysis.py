import datetime
import json
import time
from dataclasses import dataclass
from math import ceil, floor, log, nan
from typing import Tuple, Type

import category_encoders as ce
import pandas as pd
from alive_progress import alive_bar
from graphviz import Source
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sqlalchemy import Engine, text
from tabulate import tabulate


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


class PathAnalysis:
    def __init__(
        self,
        dataId: int,
        skip_features: list[str] = [],
        skip_values: dict = {},
        db: Engine = None,
        datetime_format: str = None,
        concept_hierarchy: dict = {},
        column_types: dict = {},
        target: str = None,
    ):
        self.db = db
        self.dataId = dataId
        self.skip_features = skip_features
        self.skip_values = skip_values
        self.datetime_format = datetime_format
        self.concept_hierarchy = concept_hierarchy
        self.column_types = column_types
        self.target = target

        self.max_depth = 50
        self.entropy_threshold = 0.4
        self.datetime_columns = []
        self.quantile_labels = ["low", "middle", "high"]

    def analysis_pipeline(self):
        with alive_bar(10) as bar:
            print(f"Fetcing data from database with data id = {self.dataId}")
            self.fetch_data_from_db()
            time.sleep(0.001)
            bar()

            print("Preprocessing...")
            self.preprocessing()
            time.sleep(0.001)
            bar()

            print("Search numerical column...")
            self.search_numerical_column()
            time.sleep(0.001)
            bar()

            print("Search datetime column...")
            self.search_datetime_column()
            time.sleep(0.001)
            bar()

            print("Transform datetime column...")
            self.transform_datetime_column()
            time.sleep(0.001)
            bar()

            print("Transform concept hierarchy...")
            self.transform_concept_hierarchy()
            time.sleep(0.001)
            bar()

            print("Prepare data to fit model...")
            self.prepare_data_to_fit()
            time.sleep(0.001)
            bar()

            print("Encode columns...")
            self.encode_category_column()
            time.sleep(0.001)
            bar()

            print("Fitting decision tree model...")
            self.fit_decision_tree_model()
            time.sleep(0.001)
            bar()

            print("Saving analysis result...")
            self.save_analysis_information()
            self.save_result()
            self.save_analysis_table_to_db()
            time.sleep(0.001)
            bar()

            print("All done!")

        self.print_result_summary()
        self.print_analysis_table()

    def fetch_data_from_db(self):
        with self.db.connect() as connection:
            query = text("SELECT CName FROM [DV].[dbo].[Object] where OID = :OID")
            object_table = connection.execute(query, {"OID": self.dataId}).fetchall()

            self.data_name = pd.DataFrame(object_table)["CName"][0]

            query = text(f"SELECT * FROM [RawDB].[dbo].[D{self.dataId}]")
            data = connection.execute(query).fetchall()

            self.origin_df = pd.DataFrame(data)
            self.train_df = self.origin_df.copy()
            self.analysis_df = self.train_df.copy()
            self.column_names = self.train_df.columns.tolist()

    def preprocessing(self):
        for column in self.column_types.keys():
            self.train_df[column] = self.train_df[column].astype(self.column_types[column])
            self.analysis_df[column] = self.analysis_df[column].astype(self.column_types[column])

        # drop rows that contain values
        # https://www.statology.org/pandas-drop-rows-with-value/
        for column in self.skip_values.keys():
            self.train_df = self.train_df[~self.train_df[column].isin(self.skip_values[column])]

    # ! Warning: 數值欄位不允許有千分位 , 分隔符號，只能是純數字
    def search_numerical_column(self):
        self.quantile_mapping = {}
        numerical_column_tuple: Tuple[str, int, int] = []

        for col in self.column_names:
            column_ratio = len(self.train_df[col].unique()) / self.train_df[col].count()
            # 用 ratio 判斷 column 是否為類別型資料，數值型資料 ratio 通常會很大
            is_categorical_column = (
                self.train_df[col].dtype == "object" or self.train_df[col].dtype == "category" or column_ratio < 0.01
            )
            is_numerical_column = (
                self.train_df[col].dtype == "int64" or self.train_df[col].dtype == "float64"
            ) and not is_categorical_column
            if is_numerical_column:
                numerical_column_tuple.append([col, self.train_df[col].min(), self.train_df[col].max()])

        # 處理數值型資料 -> 歸類為多個區段，並用 label 取代
        quantile_numeric = []

        for tuple in numerical_column_tuple:
            col = tuple[0]
            quantile_numeric.append(col)

            # discrete_bin_num = 3
            # quantile = pd.qcut(df[col], q=discrete_bin_num)
            # df[col] = pd.qcut(df[col], q=discrete_bin_num, labels=quantile_labels)

            # 利用類似常態分佈的比例分類成三個等級 : 低、中、高
            quantile = pd.qcut(self.train_df[col], q=[0, 0.1575, 0.8425, 1])
            self.train_df[col] = pd.qcut(self.train_df[col], q=[0, 0.1575, 0.8425, 1], labels=self.quantile_labels)

            if col != self.target:
                self.analysis_df[col] = self.train_df[col]

            self.quantile_mapping[col] = pd.Series(
                data=quantile.unique().tolist(), index=self.train_df[col].unique().tolist()
            )

        self.numerical_column = numerical_column_tuple
        self.quantile_numeric = quantile_numeric

    # 搜尋是否有欄位可能是時間型資料，並嘗試轉換
    def search_datetime_column(self):
        datetime_format_list = ["%Y/%m/%d", "%Y-%m-%d", "%Y-%m", "%Y%m%d", "%Y%m", "%Y"]
        datetime_column_tuple: Tuple[str, Type[pd.Timestamp], Type[pd.Timestamp]] = []

        if self.datetime_format is not None:
            datetime_format_list.insert(0, self.datetime_format)

        # to_datetime reference: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html

        for col in self.column_names:
            is_datetime = self.train_df[col].dtype == "datetime64[ns]"
            if is_datetime:
                datetime_column_tuple.append(
                    [
                        col,
                        self.train_df[col].min(),
                        self.train_df[col].max(),
                    ]
                )
                continue
            for test_format in datetime_format_list:
                is_datetime = (
                    True
                    not in pd.to_datetime(arg=self.train_df[col].astype("str"), format=test_format, errors="coerce")
                    .isna()
                    .value_counts()
                    .index.to_list()
                )
                if is_datetime:
                    parsed_datetime = pd.to_datetime(arg=self.train_df[col], format=test_format, errors="coerce")
                    self.train_df[col] = parsed_datetime
                    self.analysis_df[col] = parsed_datetime
                    datetime_column_tuple.append(
                        [
                            col,
                            parsed_datetime.min(),
                            parsed_datetime.max(),
                        ]
                    )
                    break
            self.datetime_column = datetime_column_tuple

    # 轉換時間型欄位 -> 年、季、月
    def transform_datetime_column(self):
        # Quantize datetime problem reference:
        # https://stackoverflow.com/questions/43500894/pandas-pd-cut-binning-datetime-column-series

        for tuple in self.datetime_column:
            # date_range reference: https://pandas.pydata.org/docs/reference/api/pandas.date_range.html
            # Frequency reference: https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases
            datetime_columns = []

            col = tuple[0]
            datetime_columns.append(col)

            # * Compare which datetime frequency is the best
            # datetime_freq_manifest = ["year", "quarter", "month", "week", "day"]
            # freq_manifest = ["YS", "QS", "MS", "W", "D"]
            datetime_freq_manifest = ["year", "quarter", "month"]
            freq_manifest = ["YS", "QS", "MS"]
            freq_mapping = pd.Series(data=freq_manifest, index=datetime_freq_manifest)
            # ! validate entropy
            datetime_freq_entropy = []

            for datetime_freq in datetime_freq_manifest:
                datetime_manifest: Type[pd.DatetimeIndex] = pd.date_range(
                    start=tuple[1], end=tuple[2], freq=freq_mapping[datetime_freq]
                )
                datetime_manifest = datetime_manifest.union([tuple[2]])
                labels = self.fill_datetime_label(datetime_freq, datetime_manifest)
                discrete_datetime_series = pd.Series(
                    pd.cut(
                        self.train_df[col], bins=datetime_manifest, labels=labels, include_lowest=True, ordered=False
                    )
                )

            for freq in datetime_freq_manifest:
                datetime_manifest: Type[pd.DatetimeIndex] = pd.date_range(
                    start=tuple[1], end=tuple[2], freq=freq_mapping[freq]
                )
                datetime_manifest = datetime_manifest.union([tuple[2]])
                labels = self.fill_datetime_label(freq, datetime_manifest)

                quantile_interval = pd.cut(self.train_df[col], bins=datetime_manifest, include_lowest=True)
                self.quantile_mapping[col] = pd.Series(
                    data=quantile_interval.tolist(), index=self.train_df[col].tolist()
                )

                self.train_df[freq] = pd.Series(
                    pd.cut(
                        self.train_df[col], bins=datetime_manifest, labels=labels, include_lowest=True, ordered=False
                    )
                )
                self.analysis_df[freq] = self.train_df[freq]

                # ! validate entropy
                feature_entropy = self.count_feature_gain(feature_name=freq)
                datetime_freq_entropy.append(feature_entropy)

            # ! validate entropy test
            print(f"Original target entropy: {self.count_target_entropy()}")
            print(f"年月 entropy: {self.count_feature_gain('年月')}")
            print(datetime_freq_manifest)
            print(datetime_freq_entropy)

            self.skip_features.append(col)
            self.datetime_columns = datetime_columns

    def transform_concept_hierarchy(self):
        # print(f"Target origin entropy: {self.count_target_entropy()}")

        for col in self.concept_hierarchy.keys():
            # print(f"Origin {col} entropy:", self.count_feature_gain(col))
            hierarchy = self.concept_hierarchy[col]["hierarchy"]
            for concept in hierarchy:
                self.train_df[col] = self.train_df[col].replace(to_replace=hierarchy[concept], value=concept)
            if "order" in self.concept_hierarchy[col]:
                order = self.concept_hierarchy[col]["order"]
                self.train_df[col] = pd.Categorical(values=self.train_df[col], categories=order, ordered=True)
                self.analysis_df[col] = self.train_df[col]
            # print(f"After generalize {col} entropy:", self.count_feature_gain(col))

    def prepare_data_to_fit(self):
        self.train_df = self.train_df.drop_duplicates(keep="first")

        X: pd.DataFrame
        try:
            X = self.train_df.drop([self.target] + self.skip_features, axis=1)
        except KeyError:
            print("Column of target or skip features are not exist in data frame")

        self.X = X
        self.y = self.train_df[self.target].astype("string")

    def encode_category_column(self):
        category_frame = self.X.select_dtypes(include=["object", "category"])
        encoder = ce.OrdinalEncoder(cols=category_frame.columns)
        self.X = pd.DataFrame(encoder.fit_transform(self.X))

        self.category_column_mapping = encoder.mapping

    def fit_decision_tree_model(self):
        row_counts = len(self.X.index)
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

        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=self.max_depth,
            random_state=42,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )

        clf.fit(self.X, self.y)

        self.clf = clf
        self.feature_names = clf.feature_names_in_.tolist()
        self.feature_importances_ = clf.feature_importances_.tolist()

    def search_decision_tree_path(self, graph: DecisionTreeGraph, root_id: int = 0):
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

                if entropy > self.entropy_threshold:
                    path.pop()
                    return
                # ! #####################

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

    def decision_tree_path_analyzer(self, paths: list[DecisionTreePath]):
        feature_names = self.feature_names
        path_analysis_result: dict = {}

        for split_value in self.clf.classes_:
            path_analysis_result[split_value] = []

        # ! 開始合併所有路徑
        for path in paths:
            path_analysis_result_part = {"process": [], "features": {}}

            for feature_name in feature_names:
                path_analysis_result_part["features"][feature_name] = self.analysis_information["feature_values"][
                    feature_name
                ]["value"].copy()

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
                    path_analysis_result_part["samples"] = list(map(lambda value: float(value), sample_value))
                    path_analysis_result_part["labels"] = self.clf.classes_.tolist()
                    path_analysis_result_part["class"] = class_name
                    path_analysis_result_part["target"] = self.target

                    path_analysis_result[class_name].append(path_analysis_result_part)
                    break

                feature_type = self.analysis_information["feature_values"][labels[0]]["type"]
                split_situation = [split_symbol, feature_type]
                mapping: pd.Series = self.analysis_information["feature_values"][feature_name]["mapping"]

                match split_situation:
                    case ["<=", "category"]:
                        index = path_analysis_result_part["features"][feature_name].index(mapping[floor(split_value)])
                        path_analysis_result_part["features"][feature_name] = path_analysis_result_part["features"][
                            feature_name
                        ][0 : index + 1]
                    case ["<=", "datetime"]:
                        index = path_analysis_result_part["features"][feature_name].index(mapping[floor(split_value)])
                        path_analysis_result_part["features"][feature_name] = path_analysis_result_part["features"][
                            feature_name
                        ][0 : index + 1]
                    case [">", "category"]:
                        index = path_analysis_result_part["features"][feature_name].index(mapping[ceil(split_value)])
                        path_analysis_result_part["features"][feature_name] = path_analysis_result_part["features"][
                            feature_name
                        ][index:]
                    case [">", "datetime"]:
                        index = path_analysis_result_part["features"][feature_name].index(mapping[ceil(split_value)])
                        path_analysis_result_part["features"][feature_name] = path_analysis_result_part["features"][
                            feature_name
                        ][index:]
                    case _:
                        print("no match case")

                # ! 儲存解析結果到 process 裡面，以便回朔 decision tree 分割過程
                path_analysis_result_part["process"].append(
                    [feature_name, path_analysis_result_part["features"][feature_name]]
                )

        return path_analysis_result

    def save_result(self):
        # * Scikit-learn decision tree:
        # Using optimized version of the CART algorithm
        # Not support categorical variable for now, that is, categorical variable need to encode

        # * Entropy range:
        # From 0 to 1 for binary classification (target has only two classes, true or false)
        # From 0 to log base 2 k where k is the number of classes

        class_names = self.clf.classes_.tolist()

        dot_file = tree.export_graphviz(
            self.clf,
            feature_names=self.feature_names,
            class_names=class_names,
            max_depth=self.max_depth,
            label="all",
        )

        # Use graphviz lib to convert dot format to json format
        source = Source(dot_file)
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

        decision_tree_graph = DecisionTreeGraph(nodes, edges)
        paths = self.search_decision_tree_path(decision_tree_graph, 0)
        self.result = self.decision_tree_path_analyzer(paths)

        # convert remaining object columns to category columns
        # save column values
        self.column_values = {}

        for column in self.analysis_df.columns.tolist():
            self.column_values[column] = self.train_df[column].unique().tolist()
            if self.analysis_df[column].dtype == "object":
                self.analysis_df[column] = self.analysis_df[column].astype("category")

    def print_result_summary(self):
        print("")
        print("paths".ljust(20), "--> ", end="")

        for label in self.result:
            print(f"{label}: {len(self.result[label])} ", end="")

        print("")
        print("feature name in".ljust(20), "-->", self.feature_names)
        print("feature impotances".ljust(20), "-->", self.feature_importances_)
        print("")

    def save_analysis_information(self):
        analysis_information: dict = {}
        analysis_information["target_name"] = self.target
        analysis_information["target_values"] = self.clf.classes_
        analysis_information["feature_names"] = self.feature_names

        # Numeric
        feature_series: dict[str, dict[str, str or list]] = {}

        for n in self.numerical_column:
            feature_series[n[0]] = {"type": "numeric", "value": [n[1], n[2]]}

        # Datetime
        for part_df in self.datetime_column:
            format = "%Y-%m-%d %X"
            feature_series[part_df[0]] = {
                "type": "datetime",
                "value": [part_df[1].strftime(format), part_df[2].strftime(format)],
            }

        # Category
        for c in self.category_column_mapping:
            is_datetime_column = c["col"] in self.datetime_columns
            feature_series[c["col"]] = {
                "type": "datetime" if is_datetime_column else "category",
                "value": (c["mapping"].index.to_list()),
                "mapping": pd.Series(dict((v, k) for k, v in c["mapping"].items())),
            }
            feature_series[c["col"]]["value"].pop()

        unstored_features = list(set(self.feature_names) - set(list(feature_series.keys())))

        for f in unstored_features:
            split_value = self.train_df[f].unique().astype("str").tolist()
            self.train_df[f] = self.train_df[f].astype("str")

            mapping_pairs = dict((i + 1, split_value[i]) for i in range(len(split_value)))
            mapping_pairs[-2] = nan
            mapping = pd.Series(mapping_pairs)
            feature_series[f] = {
                "type": "category",
                "value": split_value,
                "mapping": mapping,
            }

        analysis_information["feature_values"] = feature_series
        self.analysis_information = analysis_information

    def fill_datetime_label(self, freq: str, datetime_manifest: Type[pd.DatetimeIndex]):
        labels: list[str] = []

        match freq:
            case "year":
                labels = [str(year) for year in list(datetime_manifest.year)]
                if len(labels) >= 0:
                    labels.pop()
            case "quarter":
                for i in range(1, len(datetime_manifest)):
                    lower_bound = datetime_manifest[i - 1].month
                    higher_bound = datetime_manifest[i].month

                    match [lower_bound, higher_bound]:
                        case [1, 4]:
                            labels.append("Q1")
                        case [4, 7]:
                            labels.append("Q2")
                        case [7, 10]:
                            labels.append("Q3")
                        case [10, 1]:
                            labels.append("Q4")
                        case _:
                            label_mapping = pd.Series(data=["Q1", "Q2", "Q3", "Q4"], index=[1, 4, 7, 10])
                            labels.append(label_mapping[lower_bound])
            case "month":
                labels = [str(month) for month in list(datetime_manifest.month)]
                if len(labels) >= 0:
                    labels.pop()
            case "week":
                labels = [
                    "{}".format(
                        datetime.strftime(datetime_manifest[i], format="%U"),
                    )
                    for i in range(0, len(datetime_manifest))
                ]
                if len(labels) >= 0:
                    labels.pop()
            case "day":
                labels = [str(day) for day in list(datetime_manifest.day)]
                if len(labels) >= 0:
                    labels.pop()
            case _:
                pass

        return labels

    def count_target_entropy(self):
        total_quantity = len(self.train_df[self.target])
        class_value_quantity = len(self.train_df[self.target].unique())
        H_s = [
            -(target_value / total_quantity) * log(target_value / total_quantity, class_value_quantity)
            for target_value in self.train_df[self.target].value_counts().values.tolist()
        ]
        entropy = sum(H_s)
        return entropy

    def count_feature_gain(self, feature_name: str):
        part_df = pd.DataFrame(columns=[feature_name, self.target])
        part_df[feature_name] = self.train_df[feature_name].astype("str")
        part_df[self.target] = self.train_df[self.target]

        # Quantity of S
        total_quantity = len(part_df)
        class_quantity = len(part_df[self.target].unique())
        gain = 0

        for value in part_df[feature_name].dropna().unique().tolist():
            # Filter target attribute correspond the value of target
            value_df = part_df.loc[part_df[feature_name] == value]
            value_quantity = len(value_df)
            target_value_counts = value_df[self.target].value_counts().values.tolist()
            H_sv = 0
            for value_count in target_value_counts:
                if value_count != 0:
                    H_sv += -(value_count / value_quantity) * log(value_count / value_quantity, class_quantity)
            gain += value_quantity / total_quantity * H_sv

        return gain

    def print_analysis_table(self):
        if self.analysis_df.size > 10:
            print(tabulate(self.analysis_df[0:10], floatfmt=",.2f", tablefmt="github", headers="keys"), end="\n\n")
        else:
            print(tabulate(self.analysis_df, floatfmt=",.2f", tablefmt="github", headers="keys"), end="\n\n")

    def save_analysis_table_to_db(self):
        self.analysis_df.to_sql("A" + str(self.dataId), self.db, if_exists="replace", index=False, schema="dbo")
