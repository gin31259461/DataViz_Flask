import json
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from tabulate import tabulate

AggMethodType = Literal["sum", "mean", "count"]


@dataclass
class SingleChartInstance:
    name: str
    label: str
    value: float | int


@dataclass
class GroupedChartInstance:
    x: str
    group: dict


class PivotAnalysis:
    def __init__(
        self,
        data: pd.DataFrame,
        index=None,
        values=None,
        columns=None,
        focus_columns: list[str] = None,
        focus_index=None,
    ) -> None:
        self.data = data
        self.index = index
        self.values = values
        self.columns = columns
        self.focus_columns = focus_columns
        self.focus_index = focus_index

        self.pivoted_table = None
        self.process_result = None
        self.index_value_counts = None

    def process_pivot_data(self, process: list[list[str, list[str]]], target: str):
        # [[before_focus, after_focus, before_count, after_count], [...], [...], ...]
        self.process_result = []

        for p in process:
            column = p[0]
            values = p[1]

            self.index = column
            self.values = target
            self.focus_index = None

            self.start_pivot_table()
            before_focus_index = self.to_grouped_data()
            before_index_value_counts = self.index_value_counts
            # self.print_pivoted_table()

            self.focus_index = values

            self.start_pivot_table()
            after_focus_index = self.to_grouped_data()
            after_index_value_counts = self.index_value_counts
            # self.print_pivoted_table()

            self.process_result.append(
                [
                    before_focus_index,
                    after_focus_index,
                    [
                        {
                            "name": before_index_value_counts.index.name + " " + before_index_value_counts.name,
                            "label": v,
                            "value": before_index_value_counts[v],
                        }
                        for v in before_index_value_counts.index.tolist()
                    ],
                    [
                        {
                            "name": after_index_value_counts.index.name + " " + after_index_value_counts.name,
                            "label": v,
                            "value": after_index_value_counts[v],
                        }
                        for v in after_index_value_counts.index.tolist()
                    ],
                ]
            )

    # agg : aggregate 聚合
    def start_pivot_table(self, agg_method: list[AggMethodType] = ["sum"]):
        if self.index is None or self.values is None:
            return None

        if self.focus_index is not None:
            self.data = self.data[self.data[self.index].isin(self.focus_index)]

        # main pivot
        if self.columns is None:
            pivoted_table = self.data.pivot_table(
                index=self.index,
                values=self.values,
                aggfunc=agg_method,
                observed=True,
            )
        else:
            pivoted_table = self.data.pivot_table(
                index=self.index,
                values=self.values,
                columns=self.columns,
                aggfunc=[agg_method[0]],
                observed=True,
            )

        if self.focus_columns is not None:
            pivoted_table = pivoted_table[self.focus_columns]

        # 如果 index 是 year, month, day 則重新排序
        if (
            pivoted_table.index.name == "year"
            or pivoted_table.index.name == "month"
            or pivoted_table.index.name == "day"
        ):
            self.pivoted_table = pivoted_table.reindex(
                index=pivoted_table.index.to_series().astype(int).sort_values().index
            )
        else:
            self.pivoted_table = pivoted_table

        self.index_value_counts = self.data[self.index].value_counts()

    def print_pivoted_table(self):
        if self.pivoted_table is None:
            print("pivoted_table is None, please run start_pivot_table first.")
            return

        print(tabulate(self.pivoted_table, headers="keys", tablefmt="github", floatfmt=",.2f"), end="\n\n")

    def print_pivot_results(self):
        self.print_pivoted_table()
        print(self.index_value_counts)

    def to_json(self):
        return self.pivoted_table.to_json()

    def to_grouped_data(self):
        """as function name

        Structrue: a list contain many grouped instance
        [
            {
                "x": "column (columns)",
                "group (index)": value (values)
            },
            ...
        ]

        Returns:
            list: grouped instances
        """
        data: list[GroupedChartInstance] = []
        pivoted_dict: dict = json.loads(self.to_json())

        for k in pivoted_dict.keys():
            data.append({"x": k, "group": pivoted_dict[k]})

        return data

    def to_single_data(self):
        """as function name

        Structrue: a list contain many single instance
        [
            {
                "name": "column (columns)",
                "lable": "group (index)",
                "value": "value (values)"
            },
            ...
        ]

        Returns:
            list: single instances
        """
        data: list[SingleChartInstance] = []
        pivoted_dict: dict = json.loads(self.to_json())

        for ki in pivoted_dict.keys():
            for kj in pivoted_dict[ki].keys():
                part = {}
                part["name"] = ki
                part["label"] = kj
                part["value"] = pivoted_dict[ki][kj]
                data.append(part)

        return data
