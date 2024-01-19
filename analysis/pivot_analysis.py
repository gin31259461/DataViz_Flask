import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

AggMethodType = Literal["sum", "mean"]
agg_func = {"sum": np.sum, "mean": np.mean}


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
        self.proccess_result = None

    def process_pivot_data(self, process: list[list[str, list[str]]], target: str):
        # [[before, after], [...], [...], ...]
        self.proccess_result = []

        for p in process:
            column = p[0]
            values = p[1]

            self.index = column
            self.values = target
            self.focus_index = None

            self.start_pivot_table()
            before_focus_index = self.to_single_data()
            # self.print_pivoted_table()

            self.focus_index = values

            self.start_pivot_table()
            after_focus_index = self.to_single_data()
            # self.print_pivoted_table()

            self.proccess_result.append([before_focus_index, after_focus_index])

    # agg : aggregate 聚合
    def start_pivot_table(self, agg_method: list[AggMethodType] = ["sum"]):
        if self.index is None or self.values is None:
            return None

        if self.focus_index is not None:
            self.data = self.data[self.data[self.index].isin(self.focus_index)]

        if self.columns is None:
            pivoted_table = self.data.pivot_table(
                index=self.index, values=self.values, aggfunc=list(map(lambda m: agg_func[m], agg_method))
            )
        else:
            pivoted_table = self.data.pivot_table(
                index=self.index, values=self.values, columns=self.columns, aggfunc=agg_func[agg_method[0]]
            )

        if self.focus_columns is not None:
            pivoted_table = pivoted_table[self.focus_columns]

        if self.focus_index is not None:
            pivoted_table = pivoted_table.drop(
                list(filter(lambda index: index not in self.focus_index, pivoted_table.index)), axis=0
            )

        self.pivoted_table = pivoted_table

    def print_pivoted_table(self):
        if self.pivoted_table is None:
            print("pivoted_table is None, please run start_pivot_table first.")
            return

        with pd.option_context(
            "display.max_rows", None, "display.max_columns", None, "display.float_format", "{:2f}".format
        ):  # more options can be specified also
            print(self.pivoted_table.to_markdown(floatfmt=".2f"), end="\n\n")

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
