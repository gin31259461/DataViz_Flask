from analysis.path_analysis import PathAnalysis
from analysis.pivot_analysis import PivotAnalysis
from db import create_db_engine

db_engine = create_db_engine()

# * path analysis

path = PathAnalysis(
    dataId=769,
    db=db_engine,
    concept_hierarchy={
        "教育程度類別": {
            "hierarchy": {"高中以下": ["高中高職", "其他"], "大專": ["專科", "大學"], "研究所": ["碩士", "博士"]},
            "order": ["高中以下", "大專", "研究所"],
        },
    },
    target="信用卡交易金額[新台幣]",
)

path.analysis_pipeline()

# * pivot analysis

# TODO: 驗證所有 high 路徑的內容
process = path.result["high"][0]["process"]
print(process, end="\n\n")


pivot = PivotAnalysis(dataId=769, db=db_engine)
pivot.process_pivot_data(process, path.target)

# * debugging
# 分析出來的低中高是沒有經過彙總的，也就是說是某些特定的 feature 組合會有 target 的低中高 label
# 路徑分析是初步分析及篩選合適的欄位給樞紐分析使用，主軸還是放在樞紐分析上面

for p in pivot.process_result:
    print(p)
    print("")