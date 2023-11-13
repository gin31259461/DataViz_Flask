import json

import pandas as pd

from analysis.path_analysis import DataAnalysis

new_analysis = DataAnalysis(
    OID=139,
    skip_features=[],
    concept_hierarchy={
        "教育程度類別": {
            "hierarchy": {"高中以下": ["高中高職", "其他"], "大專": ["專科", "大學"], "研究所": ["碩士", "博士"]},
            "order": ["高中以下", "大專", "研究所"],
        },
        "地區": {
            "hierarchy": {
                "中部": ["南投縣", "雲林縣", "苗栗縣", "彰化縣"],
                "北部": ["宜蘭縣", "基隆市", "新竹縣", "新竹市"],
                "南部": ["嘉義縣", "屏東縣", "嘉義市", "澎湖縣"],
                "東部": ["花蓮縣", "台東縣"],
                "離島": ["金門縣", "連江縣"],
            }
        },
    },
    target="信用卡交易金額[新台幣]",
)


new_analysis.start_analysis()

import random

for c in new_analysis.class_names:
    paths = new_analysis.result[c]
    print(f"{c} path count: {len(paths)}")

print("\n")

target_class = random.choice(new_analysis.class_names)
path_index = random.choice(range(len(paths)))
target_path = paths[path_index]

part_df = new_analysis.df.copy()

for feature in new_analysis.feature_names:
    value_df = pd.DataFrame(columns=new_analysis.df.columns)
    for value in target_path[feature]:
        value_df = pd.concat([value_df, part_df.loc[(part_df[feature] == value)]], ignore_index=False)
    part_df = value_df

print(f"Target column of analysis: {target_path['target']}")
print(f"Most class of this path: {target_path['class']}")
print(f"Time range: {new_analysis.datetime_column_tuple[0][1]} to {new_analysis.datetime_column_tuple[0][2]}\n")
print(part_df[new_analysis.target].value_counts())

part_df.drop(["年月"], axis=1).drop_duplicates()
print(part_df)

print(json.loads(new_analysis.path_data_to_json()))
