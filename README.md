<div align="center">

# SerieSet: Easy Time Series Dataset with PyTorch

</div>

## 💡 What's SerieSet?

We need an easy and (relatively) general dataset builder for time series model training. This project only relies on `PyTorch`, `Pandas`, and `Numpy`. 


## ⬇️ Installation

```shell
pip install serieset
```

## </> API details

> **data** (pandas.DataFrame): dataframe with single or multiple time series, `group_id`, `date_col`, `target_col` columns are expected, `features` columns are optional.

> **inp_len** (int): input sequence length. i.e. 96.

> **pred_len** (int): prediction sequence length. i.e. 14.

> **target_col** (str): target time series column name (i.e., airport volume, store sales).

> **date_col** (str): date column name. i.e. "date".

> **group_id** (Union[str, List[str]]): group id column name. i.e. "store_name" or ["store_name", "product_id"].

> **features** (Optional[Union[str, List[str]]]): feature column name. i.e. "volume" or ["volume", "price"]. All features should be numeric.

> **train_val_split_date** (str): date for train-validation split. i.e. "2019-01-01". Default is None. If 'last', the last inp_len + pred_len data will be used for validation.

> **dtype** (str): data type of torch data tensor. Default is "float32".

> **mode** (str): train or validation.

## 💡 Example

```python
import pandas as pd
from serieset import TimeSeriesDataset

data = pd.read_csv("./data/ETTh1.csv")
data["group_id"] = "ETTh1"

print(data.head())
print(f"minimum date: {data['date'].min()}")
print(f"maximum date: {data['date'].max()}")

params = {
    'target_col': 'OT',
    'features': ["HUFL", "HULL"],
    'group_id': 'group_id',
    'date_col': 'date',
    'inp_len': 36,
    'pred_len': 12,
    'train_val_split_date': '2018-01-01 00:00:00',
    'mode': 'train',
}

torch_dataset = TimeSeriesDataset(
    data=data,
    **params
)
```