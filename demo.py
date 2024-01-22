from analysis.path_analysis import PathAnalysis
from analysis.pivot_analysis import PivotAnalysis

# * path analysis

path = PathAnalysis(
    dataId=769,
    concept_hierarchy={
        "教育程度類別": {
            "hierarchy": {"高中以下": ["高中高職", "其他"], "大專": ["專科", "大學"], "研究所": ["碩士", "博士"]},
            "order": ["高中以下", "大專", "研究所"],
        },
    },
    target="信用卡交易金額[新台幣]",
)
path.analysis_pipeline()
print("")
path.print_result_summary()

# * pivot analysis

# pivot = PivotAnalysis(path.origin_df, index="地區", values="信用卡交易金額[新台幣]")
# pivot.start_pivot_table(agg_method=["mean"])
# pivot.print_pivot_results()

pivot = PivotAnalysis(path.analysis_df)
pivot.process_pivot_data(path.result["high"][0]["process"], path.target)

# * debugging
# 分析出來的低中高是沒有經過彙總的，也就是說是某些特定的 feature 組合會有 target 的低中高 label
# 路徑分析是初步分析及篩選合適的欄位給樞紐分析使用，主軸還是放在樞紐分析上面

# process = path.result["high"][0]["process"]
# for i in range(len(pivot.process_result)):
#     print("")
#     print(process[i])
# print(path.column_values)

# print(pivot.process_result[i][0])
# print(pivot.process_result[i][1])
# print(pivot.process_index_value_counts[i][0])
# print(pivot.process_index_value_counts[i][1])
