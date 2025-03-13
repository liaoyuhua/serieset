from typing import List, Union, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):

    _dtypes = {
        "float32": torch.float32,
        "float64": torch.float64,
    }

    _modes = ["train", "val"]

    def __init__(
        self,
        data: pd.DataFrame,
        inp_len: int,
        pred_len: int,
        target_col: str,
        date_col: str,
        group_id: Union[str, List[str]],
        features: Optional[Union[str, List[str]]] = None,
        train_val_split_date: str = None,
        dtype: str = "float32",
        mode: str = "train",
    ):
        """
        Pytorch dataset for single or multiple time series data.

        Args:
            data (pd.DataFrame): dataframe with single or multiple time series, `group_id`, `date_col`, `target_col`
                columns are expected, `features` columns are optional.
            inp_len (int): Input sequence length. i.e. 96.
            pred_len (int): Prediction sequence length. i.e. 14.
            target_col (str): Target column name. i.e. "sales".
            date_col (str): Date column name. i.e. "date".
            group_id (Union[str, List[str]]): Group id column name. i.e. "store_name" or ["store_name", "product_id"].
            features (Optional[Union[str, List[str]]]): Feature column name. i.e. "volume" or ["volume", "price"].
                Note: All features should be numeric.
            train_val_split_date (str): Date for train-validation split. i.e. "2019-01-01". Default is None.
                If 'last', the last inp_len + pred_len data will be used for validation.
            dtype (str): Data type of torch data tensor. Default is "float32".
            mode (str): Train or validation.
        """

        super().__init__()
        self.dtype = dtype
        self.mode = mode
        self._validate_variable()

        self.target_col = target_col
        features = [] if features is None else features
        assert isinstance(features, (str, list)), "features must be a string or a list"
        assert isinstance(group_id, (str, list)), "group_id must be a string or a list"
        self.group_id = group_id if isinstance(group_id, list) else [group_id]
        self.features = features if isinstance(features, list) else [features]

        self.date_col = date_col
        self.inp_len = inp_len
        self.pred_len = pred_len
        self.train_val_split_date = train_val_split_date

        self.group_id_map = {}

        self.data = self._preprocess(data)
        self.train_index, self.val_index = self._construct_index(self.data)
        self.index = self.train_index if self.mode == "train" else self.val_index

    def _validate_variable(self):
        assert (
            self.dtype in self._dtypes
        ), f"dtype must be one of {list(self._dtypes.keys())}"
        assert self.mode in self._modes, f"mode must be one of {self._modes}"

    def _preprocess(self, data):
        data = data.sort_values(by=[*self.group_id, self.date_col], ignore_index=True)
        assert "__idx__" not in data.columns, "Column name '__idx__' is reserved."
        data["__idx__"] = np.arange(len(data))  # add incremental index

        return data

    def _construct_index(self, data):
        g = data.groupby(self.group_id)
        col_idx = g["__idx__"].apply(list).values.tolist()
        col_date = g[self.date_col].apply(list).values.tolist()
        col_group = g[self.group_id].first().values

        index = []
        predict_date = []
        groups = []
        for i, idx in enumerate(col_idx):
            for j in range(
                0,
                len(idx) - self.inp_len - self.pred_len + 1,
            ):
                index.append(
                    [
                        idx[j],
                        idx[j + self.inp_len + self.pred_len - 1],
                    ]
                )
                predict_date.append(col_date[i][j + self.inp_len])
                groups.append(i)
            self.group_id_map[i] = col_group[i][0]
        # convert index to dataframe
        index = pd.DataFrame(index, columns=["index_start", "index_end"])
        # add group ids and sample ids
        index["group_id"] = groups  # start from 0
        index["predict_start_date"] = predict_date

        # split train and validation
        if self.train_val_split_date is not None:
            if self.train_val_split_date != "last":
                train_index = index[
                    index["predict_start_date"] < self.train_val_split_date
                ].reset_index(drop=True)
                val_index = index[
                    index["predict_start_date"] >= self.train_val_split_date
                ].reset_index(drop=True)
            else:
                # drop last sample in each group
                train_index = (
                    index.groupby("group_id")
                    .apply(lambda x: x.iloc[:-1])
                    .reset_index(drop=True)
                )
                val_index = (
                    index.groupby("group_id")
                    .apply(lambda x: x.iloc[-1])
                    .reset_index(drop=True)
                )
            val_index["sample_id"] = np.arange(len(val_index))
        else:
            train_index = index
            val_index = None

        train_index["sample_id"] = np.arange(len(train_index))
        return train_index, val_index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        start_idx = row["index_start"]
        end_idx = row["index_end"]
        group_id = row["group_id"]
        sample_id = row["sample_id"]

        data = self.data.iloc[start_idx : end_idx + 1]

        x = torch.tensor(
            data[self.target_col][: self.inp_len].values, dtype=self._dtypes[self.dtype]
        )
        x_feats = torch.tensor(
            data[self.features][: self.inp_len].values, dtype=self._dtypes[self.dtype]
        )

        y = torch.tensor(
            data[self.target_col][-self.pred_len :].values,
            dtype=self._dtypes[self.dtype],
        )

        return {
            "x": x,
            "x_feats": x_feats,
            "y": y,
            "group_id": group_id,
            "sample_id": sample_id,
        }
