import time
from typing import Dict, Text, List
from collections import defaultdict

import numpy as np
import pandas as pd
import ast

from .logger import RcLogger


def percentage_difference(a, b):
    return abs(a - b) / ((a + b) / 2) if a + b > 0 else 0


def dicts_append(*dicts):
    out_dict = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            out_dict[k] += [v]
    return dict(out_dict)


def dicts_sum(*dicts):
    out_dict = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            out_dict[k] += v
    return dict(out_dict)


def dicts_mean(*dicts):
    out_dict = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            out_dict[k] += [v]
    for k in out_dict:
        out_dict[k] = np.mean(out_dict[k])

    return dict(out_dict)


def dicts_np_concatenate(*dicts):
    out_dict = defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            out_dict[k] = np.concatenate([out_dict[k], v])
    return dict(out_dict)


def balance_groups_by_ratio(groups: Dict[Text, int], ratio: Dict[Text, float]):
    # formula (a - removed_values) = new_value => removed_values = a - new_value --> new_values = (b - 0) * ra / rb
    def new_ratio_formula(a, b, ra, rb):
        return a - (b * ra / rb)

    out_groups = groups.copy()
    min_gr = min(groups.items(), key=lambda x: x[1])[0]
    max_gr = (out_groups.keys() - {min_gr}).pop()

    new_ratio = new_ratio_formula(groups[max_gr], groups[min_gr], ratio[max_gr], ratio[min_gr])
    if new_ratio >= 0:
        out_groups[max_gr] = new_ratio
    else:
        new_ratio = new_ratio_formula(groups[min_gr], groups[max_gr], ratio[min_gr], ratio[max_gr])
        out_groups[min_gr] = new_ratio

    return out_groups


def convert_dataframe_str_to_bytestr_cols_index(df: pd.DataFrame):
    df.index = [ast.literal_eval(_str) for _str in df.index]
    df.columns = [ast.literal_eval(_str) for _str in df.columns]

    return df


def check_multiple(arr: List):
    assert len(arr) == 2, "Cannot check multiple on list longer than 2"


def measure_time(func_to_measure):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func_to_measure(*args, **kwargs)
        RcLogger.get().info(f"{func_to_measure.__name__} time execution: "
                            f"{time.strftime('%H:%M:%S', time.gmtime(time.perf_counter() - start))} seconds")

        return res

    return wrapper
