import torch
import polars as pl
from ..errors import RequestRejected


class ApplyBins(torch.nn.Module):
    """BastionLab internal class used to serialize user-defined functions (UDF) in TorchScript.
    It uses `torch.nn.Module` and stores the `bin_size`, which is the aggregation count of the query.
    """

    def __init__(self, bin_size: int) -> None:
        super().__init__()
        #: The aggregation size of the query.
        self.bin_size = torch.Tensor([bin_size])

    def forward(self, x):
        bins = self.bin_size * torch.ones_like(x)
        return round(x // bins) * bins


class Palettes:
    dict = {
        "standard": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
        "light": ["#add8e6", "#ffb6c1", "#cbc3e3", "#ff8520", "#7ded7f", "#92f7e4"],
        "mithril": ["#f0ba2d", "#0b2440", "#030e1a", "#ffffff", "#F74C00"],
        "ocean": ["#006A94", "#2999BC", "#3EBDC8", "#69D1CB", "#83DEF1", "#01BFFF"],
    }


class ApplyAbs(torch.nn.Module):
    """BastionLab internal class used to serialize user-defined functions (UDF) in TorchScript.
    It uses `torch.nn.Module` and applies abs() to the input value.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return torch.abs(x)


class VisTools:
    def _get_estimator_dict(agg: str):
        return {
            "mean": pl.col(agg).mean(),
            "count": pl.col(agg).count(),
            "max": pl.col(agg).max(),
            "min": pl.col(agg).min(),
            "std": pl.col(agg).std(),
            "sum": pl.col(agg).sum(),
            "median": pl.col(agg).median(),
        }

    def _get_unique_values(
        self,
        col: str,
    ):
        tmp = (
            self.groupby(pl.col(col))
            .agg(pl.count())
            .sort(pl.col(col))
            .collect()
            .fetch()
        )
        RequestRejected._check_valid_df(tmp)
        return tmp.to_pandas()[col].tolist()

    def _bar_get_x_position(points, index, total, width):
        scale = lambda x: x - (index - total / 2 + 0.5) * width
        return scale(points)

    def _get_all_cols(
        rdf,
        x: str,
        y: str,
        hue: str = None,
    ):
        if x == None and y == None:
            raise ValueError("Please provide a x or y column name")
        if x != None and y != None:
            selects = [x, y] if x != y else [x]
        else:
            selects = [x] if x != None else [y]
        if hue:
            selects.append(hue)
        for col in selects:
            if not col in rdf.columns:
                raise ValueError("Column ", col, " not found in dataframe")
        return selects


all = ["ApplyBins", "Palettes", "ApplyAbs"]
