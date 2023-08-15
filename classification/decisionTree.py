import json
from pathlib import Path

import pandas as pd
from graphviz import Source
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

file_path = f"{Path(__file__).parent.absolute()}/temp/temp.dot"


def decisionTreeHandler(data: pd.DataFrame, target: str, features: list):
    mappings = {}

    for feature in features:
        original_strings = data[feature].tolist()
        data[feature] = LabelEncoder().fit_transform(data[feature])
        encoded_values = data[feature].tolist()
        mappings[feature] = {
            encoded_value: original_string for original_string, encoded_value in zip(original_strings, encoded_values)
        }

    max_depth = 5
    num_bins = 3

    feature_columns = [col for col in features if col in data.columns]
    data_feature = data[feature_columns]
    data_target = pd.qcut(data[target], q=num_bins, labels=False)

    # split data into training and test datasets
    # X_train, X_test, y_train, y_test = train_test_split(
    #     data_feature, data_target, random_state=42, test_size=0.2
    # )

    # create new decision tree model
    # gini : CART algorithm, entropy : ID3 algorithm

    # analyze all path ccp alphas
    # -------------------------------------- start
    # clf = DecisionTreeClassifier(
    #     criterion="entropy", random_state=42, max_depth=max_depth
    # )
    # path = clf.cost_complexity_pruning_path(X_train, y_train)
    # ccp_alphas = path.ccp_alphas
    # ccp_alphas = ccp_alphas[:-1]  # remove max value

    # clfs = []
    # for ccp_alpha in ccp_alphas:
    #     clf = DecisionTreeClassifier(
    #         criterion="entropy",
    #         random_state=42,
    #         ccp_alpha=ccp_alpha,
    #         max_depth=max_depth,
    #     )
    #     clf.fit(X_train, y_train)
    #     clfs.append(clf)

    # # 找到交叉驗證性能最佳的 ccp_alpha 值
    # # train_scores = [clf.score(X_train, y_train) for clf in clfs]
    # test_scores = [clf.score(X_test, y_test) for clf in clfs]
    # best_ccp_alpha = ccp_alphas[np.argmax(test_scores)]
    # # ---------------------------------------- end

    clf = DecisionTreeClassifier(
        criterion="entropy",
        random_state=0,
        max_depth=max_depth,
        # ccp_alpha=best_ccp_alpha,  # Cost-Complexity Pruning
    )
    clf.fit(data_feature, data_target)

    print(file_path)

    dotData = tree.export_graphviz(
        clf,
        out_file=file_path,
        feature_names=features,
        max_depth=max_depth,
        label="all",
        rounded=True,
        filled=True,
    )

    with open(file_path, "r", encoding="utf-8") as f:
        dotData = f.read()

    # use graphviz lib to convert dot format to json format
    source = Source(dotData)
    jsonGraph = source.pipe(format="json").decode("utf-8")
    dictGraph: dict = json.loads(jsonGraph)
    result = {"nodes": [], "edges": []}

    # filter needed part
    result["nodes"] = list(
        map(
            lambda o: {"id": o.get("_gvid"), "labels": o.get("label").split("\\n")},
            dictGraph.get("objects"),
        )
    )

    result["edges"] = dict(
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

    result["mappings"] = mappings

    return result
