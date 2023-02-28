from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar, Sequence, Union, Dict
import polars as pl
from polars.internals.sql.context import SQLContext
import json
import torch
from ..pb.bastionlab_conversion_pb2 import (
    RemoteArray as PbRemoteArray,
    RemoteDataFrame as PbRemoteDataFrame,
)
from ..pb.bastionlab_polars_pb2 import ReferenceResponse, SplitRequest, ReferenceRequest
from .client import BastionLabPolars
from .utils import ApplyBins, Palettes, ApplyAbs, VisTools
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TYPE_CHECKING
from ..errors import RequestRejected
import numpy as np
from serde.json import to_json
from ._utils import (
    Metadata,
    PolarsPlanSegment,
    EntryPointPlanSegment,
    UdfPlanSegment,
    StackPlanSegment,
    PlanSegments,
    RowCountSegment,
)
from .._utils import delegate, delegate_properties


LDF = TypeVar("LDF", bound="pl.LazyFrame")

if TYPE_CHECKING:
    from ..client import Client
    from ..torch import RemoteTensor
    import matplotlib as mat


# TODO
# collect
# cleared
# Map
# Joins
@delegate_properties("columns", "dtypes", "schema", target_attr="_inner")
@delegate(
    target_cls=pl.LazyFrame,
    target_attr="_inner",
    f_names=[
        "__bool__",
        "__contains__",
        "__copy__",
        "__deepcopy__",
        "__getitem__",
    ],
)
@delegate(
    target_cls=pl.LazyFrame,
    target_attr="_inner",
    f_names=[
        "sort",
        "cache",
        "filter",
        "select",
        "with_columns",
        "with_context",
        "with_column",
        "drop",
        "rename",
        "reverse",
        "shift",
        "shift_and_fill",
        "slice",
        "limit",
        "head",
        "tail",
        "last",
        "first",
        "take_every",
        "fill_null",
        "fill_nan",
        "std",
        "var",
        "max",
        "min",
        "sum",
        "mean",
        "median",
        "quantile",
        "explode",
        "unique",
        "drop_nulls",
        "melt",
        "interpolate",
        "unnest",
    ],
    wrap=True,
    make_docstring=lambda fname: f"""
        See the polars documentation for
        [pl.LazyFrame.{fname}](https://pola-rs.github.io/polars/py-polars/html/reference/lazyframe/api/polars.LazyFrame.{fname}.html).
    """,
)
@delegate(
    target_cls=pl.LazyFrame,
    target_attr="_inner",
    f_names=[
        "groupby",
        "groupby_rolling",
        "groupby_dynamic",
    ],
    wrap=True,
    wrap_fn=lambda rlf, res: RemoteLazyGroupBy(res, rlf._meta),
    make_docstring=lambda fname: f"""
        See the polars documentation for
        [pl.LazyFrame.{fname}](https://pola-rs.github.io/polars/py-polars/html/reference/lazyframe/api/polars.LazyFrame.{fname}.html).
    """,
)
@dataclass
class RemoteLazyFrame:
    """
    A class to represent a RemoteLazyFrame.

    Delegated attributes
    --------------------
    dtypes : dict[str, pl.DataType]
        Get dtypes of columns in LazyFrame.<br>
    schema : dict[str, pl.DataType]
        The dataframe's schema."""

    _inner: pl.LazyFrame
    _meta: Metadata

    def __str__(self: LDF) -> str:
        return f"RemoteLazyFrame"

    def __repr__(self: LDF) -> str:
        return str(self)

    def clone(self: LDF) -> LDF:
        """clones RemoteLazyFrame
        Returns:
            RemoteLazyFrame: clone of current RemoteLazyFrame
        """
        return RemoteLazyFrame(self._inner.clone(), self._meta)

    @property
    def composite_plan(self: LDF) -> str:
        """Gets composite_plan
        Returns:
            Composite_plan as str
        """
        return to_json(
            PlanSegments(
                segments=[*self._meta._prev_segments, PolarsPlanSegment(self._inner)]
            )
        )

    def collect(self: LDF) -> LDF:
        """runs any pending queries/actions on RemoteLazyFrame that have not yet been performed.
        Returns:
            FetchableLazyFrame: FetchableLazyFrame of datarame after any queries have been performed
        """
        return self._meta._polars_client._run_query(self.composite_plan)

    @staticmethod
    def sql(query: str, *rdfs: LDF) -> LDF:
        """Parses given SQL query and interpolates {} placeholders with given RemoteLazyFrames.
        Args:
            query (str): the SQL query
            rdfs (RemoteLazyFrame): DataFrames used in the SQL query
        Returns:
            RemoteLazyFrame: The resulting RemoteLazyFrame
        """
        if len(rdfs) == 0:
            raise Exception("The SQL query must at least use one RemoteLazyFrame")
        if any(
            [
                rdf._meta._polars_client is not rdfs[0]._meta._polars_client
                for rdf in rdfs
            ]
        ):
            raise Exception(
                "Cannot use remote data frames from two different servers in an SQL query"
            )

        unique_rdfs = []
        rdfs_refs = []
        for rdf in rdfs:
            try:
                rdfs_refs.append(unique_rdfs.index(rdf))
            except ValueError:
                rdfs_refs.append(len(unique_rdfs))
                unique_rdfs.append(rdf)

        query = query.format(*(f"__{i}" for i in rdfs_refs))

        ctx = SQLContext()
        for i, rdf in enumerate(unique_rdfs):
            ctx.register(f"__{i}", rdf._inner)
        res = ctx.execute(query)

        return RemoteLazyFrame(
            res,
            Metadata(
                rdfs[0]._meta._polars_client,
                [
                    seg
                    for segs in reversed(unique_rdfs)
                    for seg in segs._meta._prev_segments
                ],
            ),
        )

    def apply_udf(self: LDF, columns: List[str], udf: Callable) -> LDF:
        """Applied user-defined function to selected columns of RemoteLazyFrame and returns result
        Args:
            columns (List[str]): List of columns that user-defined function should be applied to
            udf (Callable): user-defined function to be applied to columns, must be a compatible input for the `torch.jit.script` function.
        Returns:
            RemoteLazyFrame: An updated RemoteLazyFrame after udf applied
        """
        ts_udf = torch.jit.script(udf)
        df = pl.DataFrame(
            [pl.Series(k, dtype=v) for k, v in self._inner.schema.items()]
        )
        return RemoteLazyFrame(
            df.lazy(),
            Metadata(
                self._meta._polars_client,
                [
                    *self._meta._prev_segments,
                    PolarsPlanSegment(self._inner),
                    UdfPlanSegment(columns=columns, udf=ts_udf),
                ],
            ),
        )

    def vstack(self: LDF, df2: LDF) -> LDF:
        """appends df2 to df1 provided columns have the same name/type
        Args:
            df2 (RemoteLazyFrame): The RemoteLazyFrame you wish to append to your current RemoteLazyFrame.
        Returns:
            RemoteLazyFrame: The combined RemoteLazyFrame as result of vstack
        """
        df = pl.DataFrame(
            [pl.Series(k, dtype=v) for k, v in self._inner.schema.items()]
        )
        return RemoteLazyFrame(
            df.lazy(),
            Metadata(
                self._meta._polars_client,
                [
                    *df2._meta._prev_segments,
                    PolarsPlanSegment(df2._inner),
                    *self._meta._prev_segments,
                    PolarsPlanSegment(self._inner),
                    StackPlanSegment(),
                ],
            ),
        )

    def with_row_count(self: LDF, name: str = "index") -> LDF:
        """adds new column with row count
        Args:
            name (String): The name of the new index column.
        Returns:
            RemoteLazyFrame: The RemoteLazyFrame with new row count/index column
        """
        df = pl.DataFrame(
            [pl.Series(k, dtype=v) for k, v in self._inner.schema.items()]
        )
        ret = RemoteLazyFrame(
            df.lazy(),
            Metadata(
                self._meta._polars_client,
                [
                    *self._meta._prev_segments,
                    PolarsPlanSegment(self._inner),
                    RowCountSegment(name),
                ],
            ),
        )
        # Either we need to collect before returning OR we need to make it clear to users they need to call collect() with this function
        # because if not this leads to panics etc. when we follow this with other operations that use the new column before next using collect()
        return ret.collect()

    def describe(self: LDF) -> pl.DataFrame:
        """
        Provides the following summary statistics for our RemoteLazyFrame:
        - count
        - null count
        - mean
        - std
        - min
        - max
        - median
         Raises:
            Exception: Where necessary queries to get statistical information for the operation are rejected by the data owner
        Returns:
            A Polars DataFrame containing statistical information
        """
        ret = self.select(
            [
                pl.col("*").count().suffix("_count"),
                pl.col("*").null_count().suffix("_null_count"),
                pl.col("*").mean().suffix("_mean"),
                pl.col("*").std().suffix("_std"),
                pl.col("*").min().suffix("_min"),
                pl.col("*").max().suffix("_max"),
                pl.col("*").median().suffix("_median"),
            ]
        )
        stats = ret.collect().fetch()
        RequestRejected._check_valid_df(stats)
        description = pl.DataFrame(
            {
                "describe": [
                    "count",
                    "null_count",
                    "mean",
                    "std",
                    "min",
                    "max",
                    "median",
                ],
                **{
                    x: [
                        stats.select(f"{x}_count")[0, 0],
                        stats.select(f"{x}_null_count")[0, 0],
                        stats.select(f"{x}_mean")[0, 0],
                        stats.select(f"{x}_std")[0, 0],
                        stats.select(f"{x}_min")[0, 0],
                        stats.select(f"{x}_max")[0, 0],
                        stats.select(f"{x}_median")[0, 0],
                    ]
                    for x in self.columns
                },
            }
        )
        return description

    def join(
        self: LDF,
        other: LDF,
        left_on: Union[str, pl.Expr, Sequence[Union[str, pl.Expr]], None] = None,
        right_on: Union[str, pl.Expr, Sequence[Union[str, pl.Expr]], None] = None,
        on: Union[str, pl.Expr, Sequence[Union[str, pl.Expr]], None] = None,
        how: pl.internals.type_aliases.JoinStrategy = "inner",
        suffix: str = "_right",
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> LDF:
        """Joins columns of another DataFrame.
        Args:
            other (RemoteLazyFrame): The other RemoteLazyFrame you want to join your current dataframe with.
            left_on (Union[str, pl.Expr, Sequence[Union[str, pl.Expr]], None] = None): Name(s) of the left join column(s).
            right_on (Union[str, pl.Expr, Sequence[Union[str, pl.Expr]], None] = None): Name(s) of the right join column(s).
            on (Union[str, pl.Expr, Sequence[Union[str, pl.Expr]], None] = None): Name(s) of the join columns in both DataFrames.
            how (str = 'inner'): Join strategy, which can be either `'inner'`, `'left'`, `'outer'`, `'semi'`, `'anti'` or `'cross'`.
            suffix (str = '_right'): Suffix to append to columns with a duplicate name.
            allow_parallel (bool = True): Boolean value for allowing the physical plan to evaluate the computation of both RemoteLazyFrames up to the join in parallel.
            force_parallel (bool = False): Boolean value for forcing parallel the physical plan to evaluate the computation of both RemoteLazyFrames up to the join in parallel.
        Raises:
            Exception: Where remote dataframes are from two different servers.
        Returns:
            RemoteLazyFrame: An updated RemoteLazyFrame after join performed
        """
        if self._meta._polars_client is not other._meta._polars_client:
            raise Exception("Cannot join remote data frames from two different servers")
        res = self._inner.join(
            other._inner,
            left_on,
            right_on,
            on,
            how,
            suffix,
            allow_parallel,
            force_parallel,
        )
        return RemoteLazyFrame(
            res,
            Metadata(
                self._meta._polars_client,
                [*other._meta._prev_segments, *self._meta._prev_segments],
            ),
        )

    def join_asof(
        self: LDF,
        other: LDF,
        left_on: Union[str, None] = None,
        right_on: Union[str, None] = None,
        on: Union[str, None] = None,
        by_left: Union[str, Sequence[str], None] = None,
        by_right: Union[str, Sequence[str], None] = None,
        by: Union[str, Sequence[str], None] = None,
        strategy: pl.internals.type_aliases.AsofJoinStrategy = "backward",
        suffix: str = "_right",
        tolerance: Union[str, int, float, None] = None,
        allow_parallel: bool = True,
        force_parallel: bool = False,
    ) -> LDF:
        """Performs an asof join, which is similar to a left-join but matches on nearest key rather than equal keys.

        Args:
            other (RemoteLazyFrame): The other RemoteLazyFrame you want to join your current dataframe with.
            left_on (Union[str, None] = None): Name(s) of the left join column(s).
            right_on (Union[str, None] = None): Name(s) of the right join column(s).
            on (Union[str, None] = None): Name(s) of the join columns in both DataFrames.
            by_left (Union[str, Sequence[str], None] = None): Join on these columns before doing asof join
            by_right (Union[str, Sequence[str], None] = None): Join on these columns before doing asof join
            by (Union[str, Sequence[str], None] = None): Join on these columns before doing asof join
            strategy (str = "backward"): Join strategy: can be either `'backward'` or `'forward'`.
            suffix (str = "_right"): Suffix to append to columns with a duplicate name.
            tolerance (Union[str, int, float, None] = None): Numeric tolerance. By setting this the join will only be done if the near keys are within this distance.
            suffix (str): Suffix to append to columns with a duplicate name.
            allow_parallel (bool = True): Boolean value for allowing the physical plan to evaluate the computation of both RemoteLazyFrames up to the join in parallel.
            force_parallel (bool = False): Boolean value for forcing parallel the physical plan to evaluate the computation of both RemoteLazyFrames up to the join in parallel.
        Raises:
            Exception: Where remote dataframes are from two different servers.
        Returns:
            RemoteLazyFrame: An updated RemoteLazyFrame after join performed
        """
        if self._meta._polars_client is not other._meta._polars_client:
            raise Exception("Cannot join remote data frames from two different servers")
        res = self._inner.join_asof(
            other._inner,
            left_on,
            right_on,
            on,
            by_left,
            by_right,
            by,
            strategy,
            suffix,
            tolerance,
            allow_parallel,
            force_parallel,
        )
        return RemoteLazyFrame(
            res,
            Metadata(
                self._meta._polars_client,
                [*other._meta._prev_segments, *self._meta._prev_segments],
            ),
        )

    def pieplot(
        self: LDF,
        parts: str,
        title: str = None,
        labels: Union[str, list[str]] = None,
        ax: List[str] = None,
        fig_kwargs: dict = None,
        pie_labels: bool = True,
        key: bool = True,
        key_loc: str = "center left",
        key_title: Optional[str] = None,
        key_bbox=(1, 0, 0.5, 1),
    ) -> None:
        """Draws a pie chart based on values within single column.
        pieplot collects necessary data only and calculates percentage values before calling matplotlib pyplot's pie function to create a pie chart.
        Args:
            parts (str): The name of the column containing bar chart segment values.
            title (str = None): Title to be displayed with the bar chart.
            labels (Union[str, list[str]] = None): The labels of segments in pie charts. Either a list of string labels following the same order as the values
                in your `parts` column or the name of a column containing the labels.
            ax (List[str]): Here you can send your own matplotlib axis if required. Note- if you do this, the fig_kwargs arguments will not be used.
            fig_kwargs (dict = None): A dictionary argument where you can add any kwargs you wish to be forwarded onto matplotlib.pyplot.subplots()
                when creating the figure that the pie chart will be displayed on.
            pie_labels (bool = True): You can modify this boolean value if you do not with to label the segments of your pie chart.
            key (bool = True): This key value specifies whether you want a color map key placed to the side of your pie chart.
            key_loc (str = "center left"): A string argument where you can modify the location of your segment color key on your pie chart to be forward to matplotlib's legend function.
            key_title (Optional[str] = None): A string argument where you can specify a title for this segment color key to be forward to matplotlib's legend function.
            key_bbox (tuple = 1, 0, 0.5, 1): bbox_to_anchor argument to be forward to matplotlib's legend function.
        Raises:
            ValueError: Incorrect column name given as parts or labels argument.
            various exceptions: Note that exceptions may be raised from matplotlib pyplot's pie or subplots functions, for example if fig_kwargs keywords are not valid.
        """

        tmp = self
        import matplotlib.pyplot as plt

        if parts not in self.columns:
            raise ValueError("Parts column not found in dataframe")
        if type(labels) == str and labels not in self.columns:
            raise ValueError("Labels column not found in dataframe")

        # run previous operations to ensure order of columns are as expected
        if type(labels) == str and type(parts) == str:
            tmp = tmp.collect()
        # get list of values in parts column
        parts_list = (
            tmp.select(pl.col(parts))
            .collect()
            .fetch()
            .select(parts)
            .to_series(0)
            .to_list()
        )

        # get total for calculating percentages
        total = sum(parts_list)

        # get percentages
        pie_data = list(map(lambda x: x * 100 / total, parts_list))

        # get labels list
        if type(labels) == str:
            labels_list1 = (
                tmp.select(labels)
                .collect()
                .fetch()
                .select(labels)
                .to_series(0)
                .to_list()
            )
            labels_list = ["null" if v is None else v for v in labels_list1]
        else:
            labels_list = labels

        # add these to figkwargs and go
        if ax == None:
            if fig_kwargs == None:
                fig, ax = plt.subplots(figsize=(7, 4), subplot_kw=dict(aspect="equal"))
            else:
                if "figsize" not in fig_kwargs:
                    fig_kwargs["figsize"] = (7, 4)
                fig, ax = plt.subplots(**fig_kwargs)
            if pie_labels == True:
                wedges, autotexts = plt.pie(pie_data, labels=labels_list)
            else:
                wedges, autotexts = plt.pie(pie_data)

        elif pie_labels == True:
            wedges, autotexts = ax.pie(pie_data, labels=labels_list)
        else:
            wedges, autotexts = ax.pie(pie_data)

        if key == True:
            ax.legend(
                wedges,
                labels_list,
                title=key_title,
                loc=key_loc,
                bbox_to_anchor=key_bbox,
            )
        ax.set_title(title)

    def barplot(
        self: LDF,
        x: str = None,
        y: str = None,
        hue: str = None,
        ax: mat.axes = None,
        estimator: str = "mean",
        vertical: bool = True,
        title: str = None,
        auto_label: bool = True,
        x_label: str = None,
        y_label: str = None,
        colors: Union[str, list[str]] = Palettes.dict["standard"],
        width: float = 0.75,
        **kwargs,
    ) -> mat.axes:
        """Draws a barchart
        barplot calculates bar's data using aggregated queries and then plots using Matplotlib's bar()/barh() function.
        Args:
            x (str) = None: The name of column to be used for x axes.
            y (str) = None: The name of column to be used for y axes.
            hue (str) = None: The name of column to be used for grouped barplot
            ax (matplotlib.axes) = None: matplotlib axes to be used for plot- a new axes is generated if not supplied
            estimator (str) = "mean": string representation of estimator to be used in aggregated query. Options are: "mean", "median", "count", "max", "min", "std" and "sum"
            vertical (bool) = True: option for vertical (True) or horizontal barplot (False)
            title (str) = None: string title for plot
            auto_label (bool) = True: If True, labels for axes will be derived from x/y columns automatically. If false, x_label and y_label arguments used
            x_label (str) = None: label for x axes if auto_label set to false
            y_label (str) = None: label for y axes if auto_label set to false
            colors (Union[str, list[str]]) = Palettes.dict["standard"]: colors for bars
            **kwargs: Other keyword arguments that will be passed to Matplotlib's bar/barh() function.
        Raises:
            ValueError: Incorrect column name given, no x or y values provided, estimator function not recognized
            RequestRejected: Could not continue in function as data owner rejected a required access request
            various exceptions: Note that exceptions may be raised from Seaborn when the barplot function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.
        Returns:
            Returns the Matplotlib Axes object with the plot drawn onto it.
        """
        # Set everything up
        selects = VisTools._get_all_cols(
            self, x, y, hue
        )  # get columns and no do error checking

        allowed_fns = [
            "mean",
            "count",
            "max",
            "min",
            "std",
            "sum",
            "median",
        ]  # check estimator input is valid and get correct agg function
        if estimator not in allowed_fns:
            raise ValueError(
                "Estimator ", estimator, " not in list of accepted estimators"
            )
        agg = y if y != None else x
        agg_dict = VisTools._get_estimator_dict(agg)

        if (
            isinstance(colors, str) and colors in Palettes.dict
        ):  # check color input is valid
            colors = Palettes.dict[colors]

        if ax == None:
            fig, ax = plt.subplots()  # setup ax

        labels = VisTools._get_unique_values(self, x if x else y)  # set up labels
        if None in labels:
            labels.remove(None)
        hues = (
            VisTools._get_unique_values(self, hue) if hue else ["default"]
        )  # set up hues

        # iterate over hue values (1 by default)
        for val, index in zip(hues, range(len(hues))):
            # filter data by hue if hue provided
            tmp = self.filter(pl.col(hue) == val) if hue else self

            # X or Y only plot
            if x == None or y == None:
                # run aggregated query for bar plot
                c = x if x != None else y
                tmp = (
                    tmp.filter(pl.col(c) != None)
                    .select(agg_dict[estimator])
                    .collect()
                    .fetch()
                )
                RequestRejected._check_valid_df(tmp)
                df = tmp.to_pandas()
                values = (
                    list(map(lambda x: x - (index - len(hues) / 2 + 0.5) * width, [1]))
                    if hue
                    else [1]
                )
                height = df[(x if x else y)].to_list()
                vertical = vertical if y else False
                if vertical is True:
                    ax.bar(
                        x=values,
                        height=height,
                        color=colors[index] if hue else colors,
                        width=width,
                        **kwargs,
                        label=val,
                    )
                    ax.tick_params(labelbottom=False)
                else:
                    ax.barh(
                        y=values,
                        width=height,
                        color=colors[index] if hue else colors,
                        height=width,
                        **kwargs,
                        label=val,
                    )
                    ax.tick_params(labelleft=False)
            else:
                tmp = (
                    tmp.select(pl.col(col) for col in selects)
                    .groupby(pl.col(x))
                    .agg(agg_dict[estimator])
                    .sort(pl.col(x))
                    .collect()
                    .fetch()
                )
                RequestRejected._check_valid_df(tmp)
                df = tmp.to_pandas()
                x_values = np.arange(len(df[x].to_list()))
                values = (
                    VisTools._bar_get_x_position(x_values, index, len(hues), width)
                    if hue
                    else x_values
                )
                height = df[y].to_list()
                if vertical is True:
                    ax.bar(
                        x=values if hue else values,
                        height=height,
                        width=width,
                        color=colors[index] if hue else colors,
                        label=val if hue else None,
                        **kwargs,
                    )
                else:
                    ax.barh(
                        y=values,
                        width=height,
                        height=width,
                        color=colors[index] if hue else colors,
                        label=val if hue else None,
                        **kwargs,
                    )

        # label axes and add title and color key
        if vertical is True:
            if x:
                ax.set_xlabel(x if auto_label else x_label)
            else:
                ax.set_xlabel(y if auto_label else y_label)
            if x and y:
                ax.set_ylabel(y if auto_label else y_label)
                ax.set_xticks(np.arange(len(labels)), labels=labels)
        else:
            if x and y:
                ax.set_xlabel(y if auto_label else x_label)
            if x:
                ax.set_ylabel(x if auto_label else y_label)
            else:
                ax.set_ylabel(y if auto_label else y_label)
                ax.set_yticks(np.arange(len(labels)), labels=labels)
        if title:
            ax.set_title(title)
        if hue:
            ax.legend(loc="best")
        return ax

    def histplot(
        self: LDF,
        x: str = "count",
        y: str = "count",
        ax: mat.axes = None,
        bins: int = 10,
        colors: Union[str, list[str]] = ["lightblue"],
        **kwargs,
    ) -> mat.axes:
        """Histplot plots a univariate histogram, where one x or y axes is provided or a bivariate histogram, where both x and y axes values are supplied.

        Histplot filters down a RemoteLazyFrame to necessary columns only, groups x axes into bins
        and performs aggregated queries before calling either matplotlib.pyplot's bar (for univaritate histograms) or imshow function (for bivariate histograms),
        which helps us to limit data retrieved from the server to a minimum.
        Args:
            x (str): The name of column to be used for x axes. Default value is "count", which trigger pl.count() to be used on this axes.
            y (str): The name of column to be used for y axes. Default value is "count", which trigger pl.count() to be used on this axes.
            ax (matplotlib.axes) = None: matplotlib axes to be used for plot- a new axes is generated if not supplied
            bins (int): An integer bin value which x axes will be grouped by. Default value is 10.
            colors (Union[str, list[str]]) = ["lightblue"]: colors to be used for barplot
            **kwargs: Other keyword arguments that will be passed to Matplotlib's bar function, in the case of one column being supplied, or imshow function, where both x and y columns are supplied.

        Raises:
            ValueError: Incorrect column name given
            RequestRejected: Could not continue in function as data owner rejected a required access request
            various exceptions: Note that exceptions may be raised from Matplotlib when the bar or imshow function is called. See Matplotlib's documentation for further details.
        Returns:
            Returns the Matplotlib Axes object with the plot drawn onto it.
        """
        if x == "count" and y == "count":
            raise ValueError("Please provide an at least an x or y value")

        if x != "count" and x not in self.columns:
            raise ValueError("X column not found in dataframe")
        if y != "count" and y not in self.columns:
            raise ValueError("Y column not found in dataframe")

        model = ApplyBins(bins)

        # if we have only X or Y
        if x == "count" or y == "count":
            col_x = x if x != "count" else y
            col_y = "count"
            q_x = pl.col(col_x) if col_x != "count" else pl.col(col_y)
            q_y = pl.count()

            if not col_x in self.columns and not col_y in self.columns:
                raise ValueError("Please supply a valid column for x or y axes")

            tmp = (
                self.filter(q_x != None)
                .select(q_x)
                .apply_udf([col_x if col_x != "count" else col_y], model)
                .groupby(q_x)
                .agg(q_y)
                .sort(q_x)
            )
            RequestRejected._check_valid_df(tmp)
            # horizontal barplot where x axis is count
            if "edgecolor" not in kwargs:
                kwargs["edgecolor"] = "black"

            if x == "count" and not "orient" in kwargs:
                return tmp.barplot(
                    x=col_x,
                    y=col_y,
                    vertical=False,
                    colors=colors,
                    width=1,
                    ax=ax,
                    **kwargs,
                )
            else:
                return tmp.barplot(
                    x=col_x,
                    y=col_y,
                    vertical=True,
                    colors=colors,
                    width=1,
                    ax=ax,
                    **kwargs,
                )

        # If we have X and Y
        else:
            for col in [x, y]:
                if not col in self.columns:
                    raise ValueError("Column {col} not found in dataframe")
            tmp = (
                self.filter(pl.col(x) != None)
                .filter(pl.col(y) != None)
                .select([pl.col(y), pl.col(x)])
                .apply_udf([x], model)
                .groupby([pl.col(x), pl.col(y)])
                .agg(pl.count())
                .sort(pl.col(x))
                .collect()
                .fetch()
            )
            RequestRejected._check_valid_df(tmp)
            df = tmp.to_pandas()
            my_cmap = sns.color_palette("Blues", as_cmap=True)
            pivot = df.pivot(index=y, columns=x, values="count")
            if "cmap" not in kwargs:
                kwargs["cmap"] = my_cmap
            if ax == None:
                ax = plt.gca()
            cbar_kw = {}
            im = ax.imshow(pivot, **kwargs)
            cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
            x_labels = df[x].unique().tolist()
            y_labels = df[y].unique().tolist()
            ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
            ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
            ax.set_ylabel(y)
            ax.set_xlabel(x)
            ax.invert_yaxis()

        return ax

    def lineplot(
        self: LDF,
        x: str,
        y: str,
        hue: str = None,
        size: str = None,
        style: str = None,
        units: str = None,
        **kwargs,
    ):
        """Draws a lineplot based on x and y values.

        Lineplot filters data down to necessary columns only and then calls Seaborn's lineplot function with this scaled down dataframe.

        Lineplot accepts any additional options supported by Seaborn's lineplot as kwargs, which can be viewed in Seaborn's documentation.

        Args:
            x (str): The name of column to be used for x axes.
            y (str): The name of column to be used for y axes.
            hue (str = None): The name of the column to be used as a grouping variable that will produce lines with different colors.
            size (str = None): The name of the column to be used as a grouping variable that will produce lines with different widths.
            style (str = None): The name of the column to be used as a grouping variable that will produce lines with different dashes and/or markers.
            units (str = None): The name of the column to be used as a grouping variable identifying sampling units.
            **kwargs: Other keyword arguments that will be passed to Seaborn's lineplot function.
        Raises:
            ValueError: Incorrect column name given
            RequestRejected: Could not continue in function as data owner rejected a required access request
            various exceptions: Note that exceptions may be raised from Seaborn when the lineplot function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.
        """
        selects = [x, y] if x != y else [x]

        for op in [hue, size, style, units]:
            if op not in selects and op != None:
                selects.append(op)

        if hue != None:
            kwargs["hue"] = hue
        if size != None:
            kwargs["size"] = size
        if style != None:
            kwargs["style"] = style
        if units != None:
            kwargs["units"] = units

        for col in selects:
            if not col in self.columns:
                raise ValueError(f"Column `{col}` not found in dataframe")

        # get df with necessary columns
        tmp = self.select([pl.col(x) for x in selects]).collect().fetch()
        RequestRejected._check_valid_df(tmp)
        df = tmp.to_pandas()
        sns.lineplot(data=df, x=x, y=y, **kwargs)

    def scatterplot(
        self: LDF,
        x: str,
        y: str,
        hue: str = None,
        ax: mat.axes = None,
        colors: Union[str, list[str]] = Palettes.dict["standard"],
        **kwargs,
    ) -> mat.axes:
        """Draws a scatter plot
        Scatterplot filters data down to necessary columns only and then calls Seaborn's scatterplot function.
        Args:
            x (str): The name of column to be used for x axes.
            y (str): The name of column to be used for y axes.
            hue (str) = None: The name of column to be used for grouped scatterplots
            ax (matplotlib.axes) = None: matplotlib axes to be used for plot- a new axes is generated if not supplied
            colors (Union[str, list[str]]) = Palettes.dict["standard"]: color(s) for plot(s)
            **kwargs: Other keyword arguments that will be passed to Matplotlib.pyplot's scatter function.
        Raises:
            ValueError: Incorrect column name given
            RequestRejected: Could not continue in function as data owner rejected a required access request
            various exceptions: Note that exceptions may be raised from Matplotlib.pyplot when the scatter function is called. See Matplotlib's documentation for further details.
        Returns:
            Returns the Matplotlib Axes object with the plot drawn onto it.
        """
        # if there is a hue or style argument add them to cols
        # Set everything up
        selects = VisTools._get_all_cols(
            self, x, y, hue
        )  # get columns and no do error checking

        hues = (
            VisTools._get_unique_values(self, hue) if hue else ["default"]
        )  # set up hues

        if ax == None:
            ax = plt.gca()
        # iterate over hue values (1 by default)
        for val, index in zip(hues, range(len(hues))):
            # filter data by hue if hue provided
            tmp = self.filter(pl.col(hue) == val) if hue else self
            tmp = tmp.select([pl.col(x) for x in selects]).collect().fetch()
            RequestRejected.check_valid_df(tmp)
            x_vals = tmp.select(x).to_numpy()
            y_vals = tmp.select(y).to_numpy()
            ax.scatter(
                x=x_vals,
                y=y_vals,
                color=colors[index % len(colors)],
                label=val,
                **kwargs,
            )
        if hue:
            ax.legend(loc="best")
        ax.set_ylabel(y)
        ax.set_xlabel(x)
        return ax

    def _calculate_boxes(
        self: LDF,
        x: str = None,
        y: str = None,
        ax=None,
    ):
        # todo check error handling if data owner rejects
        boxes = []
        if x == None or y == None:
            if y == None:
                col_x = pl.col(x)
            else:
                col_x = pl.col(y)
            ax.set_ylabel(x if x is not None else y)
            q1 = self.select(col_x.quantile(0.25)).collect().fetch().to_numpy()[0]
            q3 = self.select(col_x.quantile(0.75)).collect().fetch().to_numpy()[0]
            boxes.append(
                {
                    "label": x,
                    "whislo": self.select(col_x.min()).collect().fetch().to_numpy()[0],
                    "q1": q1,
                    "med": self.select(col_x.median()).collect().fetch().to_numpy()[0],
                    "q3": q3,
                    "iqr": q3 - q1,
                    "whishi": self.select(col_x.max()).collect().fetch().to_numpy()[0],
                }
            )

        else:
            col_x = pl.col(x)
            col_y = pl.col(y)
            ax.set_ylabel(y)
            ax.set_xlabel(x)
            mins = self.groupby(col_x).agg(col_y.min()).sort(col_x)
            maxes = self.groupby(col_x).agg(col_y.max()).sort(col_x)
            meds = self.groupby(col_x).agg(col_y.median()).sort(col_x)
            q1s = self.groupby(col_x).agg(col_y.quantile(0.25)).sort(col_x)
            q3s = self.groupby(col_x).agg(col_y.quantile(0.75)).sort(col_x)
            iqrs = self.groupby(col_x).agg(col_y.quantile(0.25)).sort(col_x)
            labels = mins.collect().fetch().to_numpy()[:, 0]

            for x in range(len(labels)):
                boxes.append(
                    {
                        "label": labels[x],
                        "whislo": mins.collect().fetch().to_numpy()[x, 1],
                        "q1": q1s.collect().fetch().to_numpy()[x, 1],
                        "med": meds.collect().fetch().to_numpy()[x, 1],
                        "q3": q3s.collect().fetch().to_numpy()[x, 1],
                        "iqr": iqrs.collect().fetch().to_numpy()[x, 1],
                        "whishi": maxes.collect().fetch().to_numpy()[x, 1],
                    }
                )
        return boxes

    def boxplot(
        self: LDF,
        x: str = None,
        y: str = None,
        colors: Union[str, list[str]] = Palettes.dict["standard"],
        vertical: bool = True,
        ax: "mat.axes" = None,
        widths: float = 0.75,
        median_linestyle: str = "-",
        median_color: str = "black",
        median_linewidth: float = 0.75,
        **kwargs,
    ):
        """Draws a boxplot based on x and y values.

        boxplot uses aggregated queries to get data necessary to create a boxplot using matplotlib's boxplot

        kwargs arguments are fowarded to matplotlib's Axes.bxp boxplot function

        Args:
            x (str): The name of column to be used for x axes.
            y (str): The name of column to be used for y axes.
            colors (Union[str, list[str]]): The color(s) or name of builtin BastionLab color palette to be used for boxes
            vertical (bool): Option for vertical or horizontal orientation
            ax (matplotlib.axes): axes to plot on. A new axes is created if set to None.
            widths (float): boxes' widths
            median_linestyle (str): linestyle for median line
            median_color (str): color for median line
            median_linewidth (float): boxes' widths
            **kwargs: keyword arguments that will be passed to Matplolib's bxp function
        Raises:
            ValueError: Incorrect column name given
            various exceptions: Note that exceptions may be raised from Seaborn when the lineplot function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.
        """

        if isinstance(colors, str):
            c = colors
            if c in Palettes.dict:
                colors = Palettes.dict[c]
            elif c in mat.colors.cnames:
                colors = [mat.colors.cnames[c]]
            else:
                raise ValueError("Color not found")
        if ax == None:
            ax = plt.gca()
        selects = []
        for col in [x, y]:
            if col != None:
                if col not in self.columns:
                    raise ValueError(f"Column `{col}` not found in dataframe")
                else:
                    selects.append(col)
        if selects == []:
            raise ValueError("Please specify at least an X or Y value")
        boxes = self._calculate_boxes(x, y, ax)
        medianprops = dict(
            linestyle=median_linestyle, color=median_color, linewidth=median_linewidth
        )
        boxprops = dict(facecolor="#1f77b4")
        bplot = ax.bxp(
            boxes,
            showfliers=False,
            widths=widths,
            medianprops=medianprops,
            boxprops=boxprops,
            patch_artist=True,
            vert=vertical,
        )
        i = -1
        for patch in bplot["boxes"]:
            i = i + 1
            patch.set_facecolor(colors[i % len(colors)])
        if vertical is False:
            tmp = ax.get_xlabel()
            ax.set_xlabel(ax.get_ylabel())
            ax.set_ylabel(tmp)
        plt.show()

    def facet(
        self: LDF, col: Optional[str] = None, row: Optional[str] = None, **kwargs
    ) -> any:
        """Creates a multi-plot grid for plotting conditional relationships.
        Args:
            col (Optional[str] = None): column value for grid
            row (Optional[str] = None): row value for grid
            **kwargs: Any additional keywords to be sent to Facet class to be applied to matplotlib pyplot's subplot function

        Returns:
            Facet instance created based on arguments given

        Raises:
            ValueError: Incorrect col/row argument provided
        """
        for x in [col, row]:
            if x != None:
                if not x in self.columns:
                    raise ValueError(f"Column `{x}` not found in dataframe")
        return Facet(inner_rdf=self, col=col, row=row, kwargs=kwargs)

    def minmax_scale(self: LDF, cols: Union[str, List[str]]) -> LDF:
        """Rescales data using the Min/Max or normalization method to a range of [0,1]
        by subtracting the overall minimum value of the data and then dividing the result by the difference between the minimum and maximum values.

        Args:
            cols (Union[str, List[str]]): The name of the column(s) which scaling should be applied to.
        Returns:
            Copy of original RemoteLazyFrame with scaling applied to specified column(s)
        Raises:
            ValueError: Column with a name provided as the cols argument not found in dataset.
        """
        columns = []
        # set up columns for single string argument
        if isinstance(cols, str):
            if cols not in self.columns:
                raise ValueError(f"Column `{cols}` not found in dataframe")
            columns.append(cols)
        else:  # set up columns for list
            for x in cols:
                if x not in self.columns:
                    raise ValueError(f"Column `{x}` not found in dataframe")
                columns.append(x)
        rdf = self.with_columns(
            [
                (pl.col(x) - pl.col(x).min())
                / (pl.col(x).max() - pl.col(x).min()).alias(x)
                for x in columns
            ]
        )
        return rdf

    def mean_scale(self: LDF, cols: Union[str, List[str]]) -> LDF:
        """Similar to the Min/Max scaling method, but subtracts the overall mean value of data instead of the min value.

        Args:
            cols (Union[str, List[str]]): The name of the column(s) which scaling should be applied to.
        Returns:
            Copy of original RemoteLazyFrame with scaling applied to specified column(s)
        Raises:
            ValueError: Column with a name provided as the cols argument not found in dataset.
        """
        columns = []
        # set up columns for single string argument
        if isinstance(cols, str):
            if cols not in self.columns:
                raise ValueError(f"Column `{cols}` not found in dataframe")
            columns.append(cols)
        else:  # set up columns for list
            for x in cols:
                if x not in self.columns:
                    raise ValueError(f"Column `{x}` not found in dataframe")
                columns.append(x)
        rdf = self.with_columns(
            [
                (pl.col(x) - pl.col(x).mean())
                / (pl.col(x).max() - pl.col(x).min()).alias(x)
                for x in columns
            ]
        )
        return rdf

    def max_abs_scale(self: LDF, cols: Union[str, List[str]]) -> LDF:
        """Rescales each data point between -1 and 1 by dividing each data point by its maximum absolute value.

        Args:
            cols (Union[str, List[str]]): The name of the column(s) which scaling should be applied to.
        Returns:
            Copy of original RemoteLazyFrame with scaling applied to specified column(s)
        Raises:
            ValueError: Column with a name provided as the cols argument not found in dataset.
        """
        model = ApplyAbs()
        columns = []
        # set up columns for single string argument
        if isinstance(cols, str):
            if cols not in self.columns:
                raise ValueError(f"Column `{cols}` not found in dataframe")
            columns.append(cols)
        else:  # set up columns for list
            for x in cols:
                if x not in self.columns:
                    raise ValueError(f"Column `{x}` not found in dataframe")
                columns.append(x)
        rdf = self.select([pl.col(x) for x in self.columns]).apply_udf(
            [x for x in columns], model
        )
        rdf = rdf.with_columns(
            [pl.col(x) / (pl.col(x).max()).alias(x) for x in columns]
        )
        return rdf

    def zscore_scale(self: LDF, cols: Union[str, List[str]]) -> LDF:
        """Rescales data by subtracting the mean from data poiints and then dividing the result by the standard deviation of the data.

        Args:
            cols (Union[str, List[str]]): The name of the column(s) which scaling should be applied to.
        Returns:
            Copy of original RemoteLazyFrame with scaling applied to specified column(s)
        Raises:
            ValueError: Column with a name provided as the cols argument not found in dataset.
        """
        columns = []
        # set up columns for single string argument
        if isinstance(cols, str):
            if cols not in self.columns:
                raise ValueError(f"Column `{cols}` not found in dataframe")
            columns.append(cols)
        else:  # set up columns for list
            for x in cols:
                if x not in self.columns:
                    raise ValueError(f"Column `{x}` not found in dataframe")
                columns.append(x)
        rdf = self.with_columns(
            [(pl.col(x) - pl.col(x).mean()) / pl.col(x).std().alias(x) for x in columns]
        )
        return rdf

    def median_quantile_scale(self: LDF, cols: Union[str, List[str]]) -> LDF:
        """Rescales data by subtracting the median value from data points and dividing the result by the IQR (inter-quartile range).

        Args:
            cols (Union[str, List[str]]): The name of the column(s) which scaling should be applied to.
        Returns:
            Copy of original RemoteLazyFrame with scaling applied to specified column(s)
        Raises:
            ValueError: Column with a name provided as the cols argument not found in dataset.
        """
        columns = []
        # set up columns for single string argument
        if isinstance(cols, str):
            if cols not in self.columns:
                raise ValueError(f"Column `{cols}` not found in dataframe")
            columns.append(cols)
        else:  # set up columns for list
            for x in cols:
                if x not in self.columns:
                    raise ValueError(f"Column `{x}` not found in dataframe")
                columns.append(x)
        rdf = self.with_columns(
            [
                (pl.col(x) - pl.col(x).median())
                / (pl.col(x).quantile(0.75) - pl.col(x).quantile(0.25)).alias(x)
                for x in columns
            ]
        )
        return rdf


@dataclass
class FetchableLazyFrame(RemoteLazyFrame):
    """
    A class to represent a FetchableLazyFrame, which can then be accessed as a Polar's dataframe via the fetch() method.
    """

    _identifier: str

    def to_array(self: "FetchableLazyFrame") -> "RemoteArray":
        """
        Converts a FetchableLazyFrame into a RemoteArray

        Returns:
            RemoteArray
        """
        client = self._meta._polars_client.client
        res = client._converter._stub.ConvToArray(
            PbRemoteDataFrame(identifier=self._identifier)
        )
        return RemoteArray(client, res.identifier)

    @property
    def identifier(self) -> str:
        """
        Gets identifier

        Return:
            returns identifier
        """
        return self._identifier

    @staticmethod
    def _from_reference(client: BastionLabPolars, ref: ReferenceResponse) -> LDF:
        header = json.loads(ref.header)["inner"]

        def get_dtype(v: Union[str, Dict]):
            if isinstance(v, str):
                return [None], getattr(pl, v)()
            else:
                k, v = list(v.items())[0]
                values, v = get_dtype(v)
                return [values], getattr(pl, k)(v)

        def get_series(name, dtype):
            values, dtype = get_dtype(dtype)
            return pl.Series(name, values=values, dtype=dtype)

        df = pl.DataFrame([get_series(k, v) for k, v in header.items()])

        return FetchableLazyFrame(
            _identifier=ref.identifier,
            _inner=df.lazy(),
            _meta=Metadata(client, [EntryPointPlanSegment(ref.identifier)]),
        )

    def __str__(self) -> str:
        return f"FetchableLazyFrame(identifier={self._identifier})"

    def __repr__(self) -> str:
        return str(self)

    def fetch(self) -> pl.DataFrame:
        """Fetches your FetchableLazyFrame and returns it as a Polars DataFrame
        Returns:
            Polars.DataFrame: returns a Polars DataFrame instance of your FetchableLazyFrame
        """
        return self._meta._polars_client._fetch_df(self._identifier)

    def save(self):
        return self._meta._polars_client._persist_df(self._identifier)

    def delete(self):
        return self._meta._polars_client._delete_df(self._identifier)


@dataclass
class Facet:
    """Namespace for matplotlib functions"""

    inner_rdf: RemoteLazyFrame
    col: Optional[str] = None
    row: Optional[str] = None
    kwargs: dict = None

    def __str__(self: LDF) -> str:
        return f"FacetGrid"

    def scatterplot(
        self: Facet,
        x: str = None,
        y: str = None,
        hue: str = None,
        ax: mat.axes = None,
        colors: Union[str, list[str]] = Palettes.dict["standard"],
        **kwargs,
    ) -> None:
        """Draws a scatter plot for each subset in row/column facet grid.
        Scatterplot filters data down to necessary columns only before calling Seaborn's scatterplot function on rows of dataset
        where values match with each combination of row/grid values.

        Draws a scatter plot
        Scatterplot filters data down to necessary columns only and then calls Seaborn's scatterplot function.
        Args:
            x (str): The name of column to be used for x axes.
            y (str): The name of column to be used for y axes.
            hue (str) = None: The name of column to be used for grouped scatterplots
            colors (Union[str, list[str]]) = Palettes.dict["standard"]: colors for bars
            ax (matplotlib.axes) = None: matplotlib axes to be used for plot- a new axes is generated if not supplied
            **kwargs: Other keyword arguments that will be passed to Matplotlib.pyplot's scatter function.
        Raises:
            ValueError: Incorrect column name given
            RequestRejected: Could not continue in function as data owner rejected a required access request
            various exceptions: Note that exceptions may be raised from Matplotlib.pyplot when the scatter function is called. See Matplotlib's documentation for further details.
        Returns:
            Returns the Matplotlib Axes object with the plot drawn onto it.
        """
        return self.__bastion_map("scatterplot", x, y, hue, ax, colors, **kwargs)

    def lineplot(
        self: LDF,
        x: str,
        y: str,
        **kwargs,
    ) -> None:
        """Draws a lineplot based on x and y values for each subset in row/column facet grid.
         Lineplot filters data down to necessary columns only and then calls Seaborn's lineplot function on rows of dataset
        where values match with each combination of row/grid values.

        Args:
            x (str): The name of column to be used for x axes.
            y (str): The name of column to be used for y axes.
            **kwargs: Other keyword arguments that will be passed to Seaborn's lineplot function.
        Raises:
            ValueError: Incorrect column name given
            various exceptions: Note that exceptions may be raised from Seaborn when the lineplot function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.
        """
        self.__map(sns.lineplot, x=x, y=y, **kwargs)

    def histplot(
        self: Facet,
        x: str = "count",
        y: str = "count",
        bins: int = 10,
        colors: Union[str, list[str]] = ["lightblue"],
        **kwargs,
    ) -> mat.axes:
        """Draws a histplot for each subset in row/column facet grid.

        Facet's histplot iterates over each possible combination of row/column values in the dataset, filters the dataset to rows where the values match this
        combination of row/column values and applies histplot to this dataset.

        Args:
            x (str): The name of column to be used for x axes. Default value is "count", which trigger pl.count() to be used on this axes.
            y (str): The name of column to be used for y axes. Default value is "count", which trigger pl.count() to be used on this axes.
            bins (int): An integer bin value which x axes will be grouped by. Default value is 10.
            colors (Union[str, list[str]]) = ["lightblue"]: colors to be used for barplot
            **kwargs: Other keyword arguments that will be passed to Matplotlib's bar function, in the case of one column being supplied, or imshow function, where both x and y columns are supplied.

        Raises:
            ValueError: Incorrect column name given
            RequestRejected: Could not continue in function as data owner rejected a required access request
            various exceptions: Note that exceptions may be raised from Matplotlib when the bar or imshow function is called. See Matplotlib's documentation for further details.
        Returns:
            Returns the Matplotlib Axes object with the plot drawn onto it.
        """
        return self.__bastion_map("histplot", x, y, None, None, bins, colors, **kwargs)

    def barplot(
        self: LDF,
        x: Optional[str] = None,
        y: Optional[str] = None,
        hue: Optional[str] = None,
        ax: mat.axes = None,
        estimator: str = "mean",
        vertical: bool = True,
        title: str = None,
        auto_label: bool = True,
        x_label: str = None,
        y_label: str = None,
        colors: Union[str, list[str]] = Palettes.dict["standard"],
        width: float = 0.75,
        **kwargs,
    ) -> mat.axes:
        """Draws a bar chart for each subset in row/column facet grid.

        barplot filters data down to necessary columns only and then calls Seaborn's barplot function.
         Args:
            x (str) = None: The name of column to be used for x axes.
            y (str) = None: The name of column to be used for y axes.
            hue (str) = None: The name of column to be used for grouped barplot
            ax (matplotlib.axes) = None: matplotlib axes to be used for plot- a new axes is generated if not supplied
            estimator (str) = "mean": string representation of estimator to be used in aggregated query. Options are: "mean", "median", "count", "max", "min", "std" and "sum"
            vertical (bool) = True: option for vertical (True) or horizontal barplot (False)
            title (str) = None: string title for plot
            auto_label (bool) = True: If True, labels for axes will be derived from x/y columns automatically. If false, x_label and y_label arguments used
            x_label (str) = None: label for x axes if auto_label set to false
            y_label (str) = None: label for y axes if auto_label set to false
            colors (Union[str, list[str]]) = Palettes.dict["standard"]: colors for bars
            **kwargs: Other keyword arguments that will be passed to Matplotlib's bar/barh() function.
        Raises:
            ValueError: Incorrect column name given, no x or y values provided, estimator function not recognized
            RequestRejected: Could not continue in function as data owner rejected a required access request
            various exceptions: Note that exceptions may be raised from Seaborn when the barplot function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.
        Returns:
            Returns the Matplotlib Axes object with the plot drawn onto it.
        """
        return self.__bastion_map(
            "barplot",
            x,
            y,
            hue,
            ax,
            estimator,
            vertical,
            title,
            auto_label,
            x_label,
            y_label,
            colors,
            width,
            **kwargs,
        )

    def __bastion_map(
        self,
        fn,
        x: str,
        y: str,
        hue: str,
        axes: mat.axes,
        *args,
    ) -> mat.axes:
        #    create list of all columns needed for query
        selects = []
        for to_add in [x, y, self.col, self.row, hue]:
            if to_add != None and to_add != "count":
                selects.append(to_add)
                if to_add not in self.inner_rdf.columns:
                    raise ValueError("Column ", to_add, " not found in dataframe")

        #    get unique row and col values
        cols = VisTools._get_unique_values(self.inner_rdf, self.col) if self.col else []
        rows = VisTools._get_unique_values(self.inner_rdf, self.row) if self.row else []

        # mapping
        r_len = len(rows) if len(rows) != 0 else 1
        c_len = len(cols) if len(cols) != 0 else 1
        if axes == None:
            figsize = ((5 * c_len), (5 * r_len))
            fig, axes = plt.subplots(r_len, c_len, figsize=figsize)
        cols_len = len(cols)
        rows_len = len(rows)
        if (cols_len != 0) and (rows_len != 0):
            for col_count in range(cols_len):
                for row_count in range(rows_len):
                    df = self.inner_rdf.clone().filter(
                        (pl.col(self.col) == cols[col_count])
                        & (pl.col(self.row) == rows[row_count])
                    )
                    t1 = (
                        self.row
                        + ": "
                        + str(rows[row_count])
                        + " | "
                        + self.col
                        + ": "
                        + str(cols[col_count])
                    )
                    if fn == "histplot":
                        df.select([pl.col(col) for col in selects]).histplot(
                            x,
                            y,
                            axes[row_count, col_count],
                            *args,
                        )
                    elif fn == "barplot":
                        df.select([pl.col(col) for col in selects]).barplot(
                            x, y, hue, axes[row_count, col_count], *args
                        )
                    else:
                        df.select([pl.col(col) for col in selects]).scatterplot(
                            x, y, hue, axes[row_count, col_count], *args
                        )
                    axes[row_count, col_count].set_title(t1)
        else:
            col_check = True if cols_len != 0 else False
            max_len = cols_len if col_check else rows_len
            my_list = cols if col_check else rows
            t = self.col if col_check else self.row
            for count in range(max_len):
                df = self.inner_rdf.clone().filter((pl.col(t) == my_list[count]))
                t1 = t + ": " + str(my_list[count])
                if fn == "histplot":
                    df.select([pl.col(col) for col in selects]).histplot(
                        x, y, axes[count], *args
                    )
                elif fn == "barplot":
                    df.select([pl.col(col) for col in selects]).barplot(
                        x, y, hue, axes[count], *args
                    )
                else:
                    df.select([pl.col(col) for col in selects]).scatterplot(
                        x,
                        y,
                        hue,
                        axes[count],
                        *args,
                    )
                axes[count].set_title(t1)
        return axes

    def __map(self: LDF, func, **kwargs) -> None:
        # create list of all columns needed for query
        selects = [self.col, self.row]
        if "x" in kwargs and not kwargs["x"] in selects:
            selects.append(kwargs["x"])
        if "y" in kwargs and not kwargs["y"] in selects:
            selects.append(kwargs["y"])

        for col in selects:
            if col not in self.inner_rdf.columns:
                raise ValueError(f"Column `{col}` not found in dataframe")

        # get unique row and col values
        cols = []
        rows = []
        if self.col != None:
            tmp = (
                self.inner_rdf.groupby(pl.col(self.col))
                .agg(pl.count())
                .sort(pl.col(self.col))
                .collect()
                .fetch()
            )
            RequestRejected._check_valid_df(tmp)
            cols = tmp.to_pandas()[self.col].tolist()

        if self.row != None:
            tmp = (
                self.inner_rdf.groupby(pl.col(self.row))
                .agg(pl.count())
                .sort(pl.col(self.row))
                .collect()
                .fetch()
            )
            RequestRejected._check_valid_df(tmp)
            rows = tmp.to_pandas()[self.row].tolist()

        # mapping
        r_len = len(rows) if len(rows) > 0 else 1
        c_len = len(cols) if len(cols) > 0 else 1
        if self.kwargs == None:
            fig, axes = plt.subplots(r_len, c_len, figsize=((5 * c_len), (5 * r_len)))
        else:
            if "figsize" not in kwargs:
                self.kwargs["figsize"] = ((5 * c_len), (5 * r_len))
            fig, axes = plt.subplots(r_len, c_len, **self.kwargs)
        cols_len = len(cols)
        rows_len = len(rows)
        if (cols_len != 0) & (rows_len != 0):
            for col_count in range(cols_len):
                for row_count in range(rows_len):
                    df = self.inner_rdf.clone().filter(
                        (pl.col(self.col) == cols[col_count])
                        & (pl.col(self.row) == rows[row_count])
                    )
                    t1 = (
                        self.row
                        + ": "
                        + str(rows[row_count])
                        + " | "
                        + self.col
                        + ": "
                        + str(cols[col_count])
                    )
                    tmp = df.select([pl.col(x) for x in selects]).collect().fetch()
                    RequestRejected._check_valid_df(tmp)
                    sea_df = tmp.to_pandas()
                    func(data=sea_df, ax=axes[row_count, col_count], **kwargs)
                    axes[row_count, col_count].set_title(t1)
        else:
            col_check = True if cols_len != 0 else False
            max_len = cols_len if col_check else rows_len
            my_list = cols if col_check else rows
            t = self.col if col_check else self.row
            for count in range(max_len):
                df = self.inner_rdf.clone().filter((pl.col(t) == my_list[count]))
                t1 = t + ": " + str(my_list[count])
                tmp = df.select([pl.col(x) for x in selects]).collect().fetch()
                RequestRejected._check_valid_df(tmp)
                sea_df = tmp.to_pandas()
                func(data=sea_df, ax=axes[row_count, col_count], **kwargs)
                axes[count].set_title(t1)


# TODO: implement apply method
@delegate(
    target_cls=pl.internals.lazyframe.groupby.LazyGroupBy,
    target_attr="_inner",
    f_names=["agg", "head", "tail"],
    wrap=True,
    wrap_fn=lambda rlg, res: RemoteLazyFrame(res, rlg._meta),
)
@dataclass
class RemoteLazyGroupBy(Generic[LDF]):
    """Builder for a GroupBy operation on a `RemoteLazyFrame`."""

    _inner: pl.internals.lazyframe.groupby.LazyGroupBy[LDF]
    _meta: Metadata


def train_test_split(
    *arrays: List["RemoteArray"],
    train_size: Optional[float] = None,
    test_size: Optional[float] = 0.25,
    shuffle: Optional[bool] = False,
    random_state: Optional[int] = None,
) -> List["RemoteArray"]:
    """
    Split RemoteArrays into train and test subsets.

    Args:
        train_size (Optional[float], optional): It should
            be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
            If None, the value is automatically set to the complement of the test size.
        test_size (Optional[float], optional): It should
            be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
            If None, the value is set to the complement of the train size.
            If train_size is also None, it will be set to 0.25. Defaults to 0.25.
        shuffle (Optional[bool], optional): Whether or
            not to shuffle the data before splitting.
        random_state (Optional[int], optional): Controls the
            shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls.
    """

    if len(arrays) == 0:
        raise ValueError("At least one RemoteDataFrame required as input")

    _train_remote_array: RemoteArray = arrays[0]

    train_size = 1 - test_size if train_size is None else train_size
    test_size = 1 - train_size if test_size is None else test_size

    if test_size < 0.0 or train_size < 0.0:
        raise ValueError("Neither train_size nor test_size can be a negative value")

    arrays: List[ReferenceRequest] = [
        ReferenceRequest(identifier=rdf.identifier) for rdf in arrays
    ]

    res = _train_remote_array._client.polars.stub.Split(
        SplitRequest(
            arrays=arrays,
            train_size=train_size,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state,
        )
    )
    res = [RemoteArray(_train_remote_array._client, ref.identifier) for ref in res.list]
    return res


class RemoteArray(RemoteLazyFrame):
    """Intermediate representation for conversion between Tensor and Dataframes."""

    def __init__(self, client: "Client", identifier: str) -> None:
        self._client = client
        self.identifier = identifier

    def to_tensor(self) -> "RemoteTensor":
        """
        Converts `RemoteArray` to `RemoteTensor`

        `RemoteArray` is BastionLab's internal intermediate representation which is akin to
        numpy arrays but are essentially pointers to a `DataFrame` on the server which when `to_tensor`
        is called converts the `DataFrame` to `Tensor` on the server.

        Returns:
            RemoteTensor
        """
        from ..torch.data import RemoteTensor

        res = self._client._converter._stub.ConvToTensor(
            PbRemoteArray(identifier=self.identifier)
        )
        return RemoteTensor._from_reference(res, self._client)

    def __str__(self) -> str:
        return f"RemoteArray(identifier={self.identifier})"


__pdoc__ = {}
__all__ = [
    "RemoteLazyFrame",
    "RemoteLazyGroupBy",
    "FetchableLazyFrame",
    "train_test_split",
    "Facet",
    "RemoteArray",
]
__pdoc__["RemoteLazyFrame.__init__"] = False
__pdoc__["RemoteLazyGroupBy.__init__"] = False
__pdoc__["FetchableLazyFrame.__init__"] = False
__pdoc__["Facet.__init__"] = False
__pdoc__["RemoteArray.__init__"] = False
