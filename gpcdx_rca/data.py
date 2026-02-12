from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np


class AERCAStyleData:
    """Thin wrapper that loads dataset via AERCA's dataset classes and exposes a common dict.

    Expects the AERCA/ package in the workspace with datasets implemented.
    """

    def __init__(self, dataset_name: str, options: Dict[str, Any]):
        self.dataset_name = dataset_name.lower()
        self.options = options
        self.data = None

    def _dataset_and_defaults(self):
        """Return (DatasetClass, default_options_dict) using AERCA arg parsers for the dataset."""
        if self.dataset_name == "msds":
            from AERCA.datasets.msds import MSDS as DS
            from AERCA.args.msds_args import create_arg_parser as parser_fn
        elif self.dataset_name == "swat":
            from AERCA.datasets.swat import SWaT as DS
            from AERCA.args.swat_args import create_arg_parser as parser_fn
        elif self.dataset_name == "linear":
            from AERCA.datasets.linear import Linear as DS
            from AERCA.args.linear_args import create_arg_parser as parser_fn
        elif self.dataset_name == "nonlinear":
            from AERCA.datasets.nonlinear import Nonlinear as DS
            from AERCA.args.nonlinear_args import create_arg_parser as parser_fn
        elif self.dataset_name == "lorenz96":
            from AERCA.datasets.lorenz96 import Lorenz96 as DS
            from AERCA.args.lorenz96_args import create_arg_parser as parser_fn
        else:
            # Fallback to linear
            from AERCA.datasets.linear import Linear as DS
            from AERCA.args.linear_args import create_arg_parser as parser_fn
        defaults = vars(parser_fn() .parse_args([]))
        return DS, defaults

    def load(self):
        DS, defaults = self._dataset_and_defaults()
        # Always use AERCA/datasets/<name> as data_dir
        defaults["data_dir"] = os.path.join(
            os.getcwd(), "AERCA", "datasets", self.dataset_name)
        # merge defaults with user-provided options (user overrides defaults)
        merged = {**defaults, **self.options}
        ds = DS(merged)
        if self.options.get("preprocessing_data", 0) == 1:
            ds.generate_example()
            ds.save_data()
        else:
            ds.load_data()
        self.data = ds.data_dict
        # also remember resolved options (for shape hints)
        self.options = merged
        return self.data

    @staticmethod
    def extract_normal_windows(data_dict, window_size: int):
        """Build a list of normal sequences (T, N) from x_n_list entries.

        AERCA provides x_n_list as large chunks; we'll further split into windows of length >= window_size.
        """
        xs = []
        for arr in data_dict["x_n_list"]:
            # arr: (T, N)
            T, N = arr.shape
            if T <= window_size:
                continue
            xs.append(arr)
        return xs

    @staticmethod
    def extract_abnormal_windows(data_dict):
        return data_dict["x_ab_list"], data_dict.get("label_list", None)
