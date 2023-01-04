from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Generic, List, Optional, TypeVar, Sequence, Union
import seaborn as sns
import polars as pl
from torch.jit import ScriptFunction
import base64
import json
import torch
from ..pb.bastionlab_polars_pb2 import ReferenceResponse
from .client import BastionLabPolars
from .utils import ApplyBins
import matplotlib.pyplot as plt

LDF = TypeVar("LDF", bound="pl.LazyFrame")


def delegate(
    target_cls: Callable,
    target_attr: str,
    f_names: List[str],
    wrap: bool = False,
    wrap_fn: Optional[Callable] = None,
) -> Callable[[Callable], Callable]:
    def inner(cls: Callable) -> Callable:
        delegates = {f_name: getattr(target_cls, f_name) for f_name in f_names}

        def delegated_fn(f_name: str) -> Callable:
            def f(_self, *args, **kwargs):
                res = (delegates[f_name])(getattr(_self, target_attr), *args, **kwargs)
                if wrap:
                    if wrap_fn is not None:
                        return wrap_fn(_self, res)
                    else:
                        wrapped_res = _self.clone()
                        setattr(wrapped_res, target_attr, res)
                        return wrapped_res
                else:
                    return res

            return f

        for f_name in f_names:
            setattr(cls, f_name, delegated_fn(f_name))

        return cls

    return inner


def delegate_properties(*names: str) -> Callable[[Callable], Callable]:
    def inner(cls: Callable) -> Callable:
        def prop(name):
            def f(_self):
                return getattr(_self._inner, name)

            return property(f)

        for name in names:
            setattr(cls, name, prop(name))

        return cls

    return inner


class CompositePlanSegment:
    """
    Composite plan segment class which handles segment plans that have not been implemented
    """

    def serialize(self) -> str:
        """
        will raise NotImplementedError

        raises:
            NotImplementedError
        """
        raise NotImplementedError()


@dataclass
class EntryPointPlanSegment(CompositePlanSegment):
    """
    Composite plan segment class responsible for new entry points
    """

    _inner: str

    def serialize(self) -> str:
        """
        returns serialized string of this plan segment

        Returns:
            str: serialized string of this plan segment
        """
        return f'{{"EntryPointPlanSegment":"{self._inner}"}}'


@dataclass
class PolarsPlanSegment(CompositePlanSegment):
    """
    Composite plan segment class responsible for Polars queries
    """

    _inner: LDF

    def serialize(self) -> str:
        """
        returns serialized string of this plan segment

        Returns:
            str: serialized string of this plan segment
        """
        return f'{{"PolarsPlanSegment":{self._inner.write_json()}}}'


@dataclass
class UdfPlanSegment(CompositePlanSegment):
    """
    Composite plan segment class responsible for user defined functions
    """

    _inner: ScriptFunction
    _columns: List[str]

    def serialize(self) -> str:
        """
        returns serialized string of this plan segment

        Returns:
            str: serialized string of this plan segment
        """
        columns = ",".join([f'"{c}"' for c in self._columns])
        b64str = base64.b64encode(self._inner.save_to_buffer()).decode("ascii")
        return f'{{"UdfPlanSegment":{{"columns":[{columns}],"udf":"{b64str}"}}}}'


@dataclass
class StackPlanSegment(CompositePlanSegment):
    """
    Composite plan segment class responsible for vstack function
    """

    def serialize(self) -> str:
        return '"StackPlanSegment"'


@dataclass
class Metadata:
    """
    A class containing metadata related to your dataframe
    """

    _client: BastionLabPolars
    _prev_segments: List[CompositePlanSegment] = field(default_factory=list)


# TODO
# collect
# cleared
# Map
# Joins
@delegate_properties(
    "columns",
    "dtypes",
    "schema",
)
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
        "with_row_count",
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
)
@dataclass
class RemoteLazyFrame:
    """
    A class to represent a RemoteLazyFrame.

    Delegate attributes:
        columns (str): Get column names.
        dtypes: Get dtypes of columns in LazyFrame.
        schema (dict[column name, DataType]): Get dataframe's schema

    Delegate methods:
    As well as the methods that will be later described, we also support the following Polars methods which are defined in detail
    in Polar's documentation:

    "sort", "cache", "filter", "select", "with_columns", "with_context", "with_column", "drop", "rename", "reverse",
    "shift", "shift_and_fill", "slice", "limit", "head", "tail", "last", "first", "with_row_count", "take_every", "fill_null",
    "fill_nan", "std", "var", "max", "min", "sum", "mean", "median", "quantile", "explode", "unique", "drop_nulls", "melt",
    "interpolate", "unnest",
    """

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
        segments = ",".join(
            [
                seg.serialize()
                for seg in [*self._meta._prev_segments, PolarsPlanSegment(self._inner)]
            ]
        )
        return f"[{segments}]"

    def collect(self: LDF) -> LDF:
        """runs any pending queries/actions on RemoteLazyFrame that have not yet been performed.
        Returns:
            FetchableLazyFrame: FetchableLazyFrame of datarame after any queries have been performed
        """
        return self._meta._client._run_query(self.composite_plan)

    def apply_udf(self: LDF, columns: List[str], udf: Callable) -> LDF:
        """Applied user-defined function to selected columns of RemoteLazyFrame and returns result
        Args:
            columns (List[str]): List of columns that user-defined function should be applied to
            udf (Callable): user-defined function to be applied to columns, must be a compatible input for torch.jit.script() function.
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
                self._meta._client,
                [
                    *self._meta._prev_segments,
                    PolarsPlanSegment(self._inner),
                    UdfPlanSegment(ts_udf, columns),
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
                self._meta._client,
                [
                    *df2._meta._prev_segments,
                    PolarsPlanSegment(df2._inner),
                    *self._meta._prev_segments,
                    PolarsPlanSegment(self._inner),
                    StackPlanSegment(),
                ],
            ),
        )

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
            how (pl.internals.type_aliases.JoinStrategy = "inner"): Join strategy {'inner', 'left', 'outer', 'semi', 'anti', 'cross'}
            suffix (str = "_right"): Suffix to append to columns with a duplicate name.
            allow_parallel (bool = True): Boolean value for allowing the physical plan to evaluate the computation of both RemoteLazyFrames up to the join in parallel.
            force_parallel (bool = False): Boolean value for forcing parallel the physical plan to evaluate the computation of both RemoteLazyFrames up to the join in parallel.
        Raises:
            Exception: Where remote dataframes are from two different servers.
        Returns:
            RemoteLazyFrame: An updated RemoteLazyFrame after join performed
        """
        if self._meta._client is not other._meta._client:
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
                self._meta._client,
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
            strategy (pl.internals.type_aliases.AsofJoinStrategy = "backward"): Join strategy: {'backward', 'forward'}.
            suffix (str  = "_right"): Suffix to append to columns with a duplicate name.
            tolerance (Union[str, int, float, None] = None): Numeric tolerance. By setting this the join will only be done if the near keys are within this distance.
            suffix (str): Suffix to append to columns with a duplicate name.
            allow_parallel (bool = True): Boolean value for allowing the physical plan to evaluate the computation of both RemoteLazyFrames up to the join in parallel.
            force_parallel (bool = False): Boolean value for forcing parallel the physical plan to evaluate the computation of both RemoteLazyFrames up to the join in parallel.
        Raises:
            Exception: Where remote dataframes are from two different servers.
        Returns:
            RemoteLazyFrame: An updated RemoteLazyFrame after join performed
        """
        if self._meta._client is not other._meta._client:
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
                self._meta._client,
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
        key_title: str = None,
        key_bbox=(1, 0, 0.5, 1),
    ) -> None:
        """Draws a pie chart based on values within single column.
        pieplot collects necessary data only and calculates percentage values before calling matplotlib pyplot's pie function to create a pie chart.
        Args:
            parts (str): The name of the column containing bar chart segment values.
            title (str = None): Title to be displayed with the bar chart.
            labels (Union[str, list[str]] = None) = The labels of segments in pie charts. Either a list of string labels following the same order as the values
            in your `parts` column or the name of a column containing the labels.
            ax (List(str)): Here you can send your own matplotlib axis if required. Note- if you do this, the fig_kwargs arguments will not be used.
            fig_kwargs (dict = None): A dictionary argument where you can add any kwargs you wish to be forwarded onto matplotlib.pyplot.subplots()
            when creating the figure that the pie chart will be displayed on.
            pie_labels (bool = True): You can modify this boolean value if you do not with to label the segments of your pie chart.
            key (bool = True): This key value specifies whether you want a color map key placed to the side of your pie chart.
            key_loc (str = "center left"): A string argument where you can modify the location of your segment color key on your pie chart to be forward to matplotlib's legend function.
            key_title (str = None): A string argument where you can specify a title for this segment color key to be forward to matplotlib's legend function.
            key_bbox (tuple = 1, 0, 0.5, 1): bbox_to_anchor argument to be forward to matplotlib's legend function.
        Raises:
            ValueError: Incorrect column name given as parts or labels argument.
            various exceptions: Note that exceptions may be raised from matplotlib pyplot's pie or subplots functions, for example if fig_kwargs keywords are not valid.
        """

        if parts not in self.columns:
            raise ValueError("Parts column not found in dataframe")
        if type(labels) == str and labels not in self.columns:
            raise ValueError("Labels column not found in dataframe")

        # get list of values in parts column
        parts_tmp = self.select(pl.col(parts)).collect().fetch().to_numpy()
        parts_list = [x[0] for x in parts_tmp]

        # get total for calculating percentages
        total = sum(parts_list)

        # get percentages
        pie_data = list(map(lambda x: x * 100 / total, parts_list))

        # get labels list
        if type(labels) == str:
            labels_tmp = self.select(pl.col(labels)).collect().fetch().to_numpy()
            labels_list = [x[0] for x in labels_tmp]
        else:
            labels_list = labels

        # add these to figkwargs and go
        if ax == None:
            if fig_kwargs == None:
                fig, ax = plt.subplots(figsize=(7, 4), subplot_kw=dict(aspect="equal"))
            else:
                if "figsize" not in self.kwargs:
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
        estimator: str = "mean",
        hue: str = None,
        **kwargs,
    ):
        """Draws a barchart
        barplot filters data down to necessary columns only and then calls Seaborn's barplot function.
        Args:
            x (str) = None: The name of column to be used for x axes.
            y (str) = None: The name of column to be used for y axes.
            estimator (str) = "mean": string represenation of estimator to be used in aggregated query. Options are: "mean", "median", "count", "max", "min", "std" and "sum"
            hue (str) = None: The name of column to be used for colour encoding.
            **kwargs: Other keyword arguments that will be passed to Seaborn's barplot function.
        Raises:
            ValueError: Incorrect column name given, no x or y values provided, estimator function not recognised
            various exceptions: Note that exceptions may be raised from Seaborn when the barplot function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.
        """
        # if there is a hue argument add them to cols and no duplicates
        if x == None and y == None:
            raise ValueError("Please provide a x or y column name")

        allowed_fns = ["mean", "count", "max", "min", "std", "sum", "median"]

        if estimator not in allowed_fns:
            raise ValueError("Column ", col, " not found in dataframe")
        if x != None and y != None:
            selects = [x, y] if x != y else [x]
        else:
            selects = [x] if x != None else [y]
        groups = [x]
        if hue != None:
            kwargs["hue"] = hue
            if hue != x:
                groups.append(hue)
            if hue != x and hue != y:
                selects.append(hue)

        for col in selects:
            if not col in self.columns:
                raise ValueError("Column ", col, " not found in dataframe")

        agg = y if y != None else x
        agg_dict = {
            "mean": pl.col(agg).mean(),
            "count": pl.col(agg).count(),
            "max": pl.col(agg).max(),
            "min": pl.col(agg).min(),
            "std": pl.col(agg).std(),
            "sum": pl.col(agg).sum(),
            "median": pl.col(agg).median(),
        }
        if x == None or y == None:
            c = x if x != None else y
            df = (
                self.filter(pl.col(c) != None)
                .select(agg_dict[estimator])
                .collect()
                .fetch()
                .to_pandas()
            )
        else:
            agg_fn = pl.col(y).mean()
            df = (
                self.filter(pl.col(x) != None)
                .select(pl.col(y) for y in selects)
                .groupby(pl.col(y) for y in groups)
                .agg(agg_dict[estimator])
                .sort(pl.col(x))
                .collect()
                .fetch()
                .to_pandas()
            )
        # run query
        if x == None:
            sns.barplot(data=df, y=y, **kwargs)
        elif y == None:
            sns.barplot(data=df, x=x, **kwargs)
        else:
            sns.barplot(data=df, x=x, y=y, **kwargs)

    def histplot(
        self: LDF, x: str = "count", y: str = "count", bins: int = 10, **kwargs
    ):
        """Histplot plots a univariate histogram, where one x or y axes is provided or a bivariate histogram, where both x and y axes values are supplied.

        Histplot filters down a RemoteLazyFrame to necessary columns only, groups x axes into bins
        and performs aggregated queries before calling either Seaborn's barplot (for univaritate histograms) or heatmap function (for bivariate histograms),
        which helps us to limit data retrieved from the server to a minimum.
        Args:
            x (str): The name of column to be used for x axes. Default value is "count", which trigger pl.count() to be used on this axes.
            y (str): The name of column to be used for y axes. Default value is "count", which trigger pl.count() to be used on this axes.
            bins (int): An integer bin value which x axes will be grouped by. Default value is 10.
            **kwargs: Other keyword arguments that will be passed to Seaborn's barplot function, in the case of one column being supplied, or heatmap function, where both x and y columns are supplied.

        Raises:
            ValueError: Incorrect column name given
            various exceptions: Note that exceptions may be raised from Seaborn when the barplot or heatmap function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.
        """

        col_x = x if x != None else "count"
        col_y = y if y != None else "count"

        if col_x == "count" and col_y == "count":
            print("Please provide an 'x' or 'y' value")
            return

        model = ApplyBins(bins)

        # if we have only X or Y
        if col_x == "count" or col_y == "count":
            q_x = pl.col(col_x) if col_x != "count" else pl.col(col_y)
            q_y = pl.count()

            if not col_x in self.columns and not col_y in self.columns:
                raise ValueError("Please supply a valid column for x or y axes")

            df = (
                self.filter(q_x != None)
                .select(q_x)
                .apply_udf([col_x if col_x != "count" else col_y], model)
                .groupby(q_x)
                .agg(q_y)
                .sort(q_x)
                .collect()
                .fetch()
                .to_pandas()
            )

            # horizontal barplot where x axis is count
            if "color" not in kwargs:
                kwargs["color"] = "lightblue"
            if "edgecolor" not in kwargs:
                kwargs["edgecolor"] = "black"
            if "width" not in kwargs:
                kwargs["width"] = 1

            if col_x == "count" and not "orient" in kwargs:
                sns.barplot(
                    data=df,
                    x=df[col_x],
                    y=df[col_y],
                    orient="h",
                    **kwargs,
                )
            else:
                sns.barplot(data=df, x=df[col_x], y=df[col_y], **kwargs)

        # If we have X and Y
        else:
            for col in [col_x, col_y]:
                if not col in self.columns:
                    raise ValueError("Column name not found in dataframe")
            df = (
                self.filter(pl.col(col_x) != None)
                .filter(pl.col(col_y) != None)
                .select([pl.col(col_y), pl.col(col_x)])
                .apply_udf([col_x], model)
                .groupby([pl.col(col_x), pl.col(col_y)])
                .agg(pl.count())
                .sort(pl.col(col_x))
                .collect()
                .fetch()
                .to_pandas()
            )
            my_cmap = sns.color_palette("Blues", as_cmap=True)
            pivot = df.pivot(index=col_y, columns=col_x, values="count")
            if "cmap" not in kwargs:
                kwargs["cmap"] = my_cmap
            ax = sns.heatmap(pivot, **kwargs)
            ax.invert_yaxis()

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
                raise ValueError("Column ", col, " not found in dataframe")

        # get df with necessary columns
        df = self.select([pl.col(x) for x in selects]).collect().fetch().to_pandas()
        sns.lineplot(data=df, x=x, y=y, **kwargs)

    def scatterplot(self: LDF, x: str, y: str, **kwargs):
        """Draws a scatter plot
        Scatterplot filters data down to necessary columns only and then calls Seaborn's scatterplot function.
        Args:
            x (str): The name of column to be used for x axes.
            y (str): The name of column to be used for y axes.
            **kwargs: Other keyword arguments that will be passed to Seaborn's scatterplot function.
        Raises:
            ValueError: Incorrect column name given
            various exceptions: Note that exceptions may be raised from Seaborn when the scatterplot function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.
        """
        # if there is a hue or style argument add them to cols
        cols = [x, y]
        if "hue" in kwargs:
            if not kwargs["hue"] in cols:
                cols.append(kwargs["hue"])
        if "style" in kwargs:
            if not kwargs["style"] in cols:
                cols.append(kwargs["style"])

        for col in cols:
            if not col in self.columns:
                raise ValueError("Column ", col, " not found in dataframe")

        # get df with necessary columns
        df = self.select([pl.col(x) for x in cols]).collect().fetch().to_pandas()
        # run query
        sns.scatterplot(data=df, x=x, y=y, **kwargs)

    def barplot(
        self: LDF,
        x: str = None,
        y: str = None,
        estimator: str = "mean",
        hue: str = None,
        **kwargs,
    ):
        """Draws a barchart
        barplot filters data down to necessary columns only and then calls Seaborn's barplot function.
        Args:
            x (str) = None: The name of column to be used for x axes.
            y (str) = None: The name of column to be used for y axes.
            estimator (str) = "mean": string represenation of estimator to be used in aggregated query. Options are: "mean", "median", "count", "max", "min", "std" and "sum"
            hue (str) = None: The name of column to be used for colour encoding.
            **kwargs: Other keyword arguments that will be passed to Seaborn's barplot function.
        Raises:
            ValueError: Incorrect column name given, no x or y values provided, estimator function not recognised
            various exceptions: Note that exceptions may be raised from Seaborn when the barplot function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.
        """
        # if there is a hue argument add them to cols and no duplicates
        if x == None and y == None:
            raise ValueError("Please provide a x or y column name")

        allowed_fns = ["mean", "count", "max", "min", "std", "sum", "median"]

        if estimator not in allowed_fns:
            raise ValueError("Column ", col, " not found in dataframe")
        if x != None and y != None:
            selects = [x, y] if x != y else [x]
        else:
            selects = [x] if x != None else [y]
        groups = [x]
        if hue != None:
            kwargs["hue"] = hue
            if hue != x:
                groups.append(hue)
            if hue != x and hue != y:
                selects.append(hue)

        for col in selects:
            if not col in self.columns:
                raise ValueError("Column ", col, " not found in dataframe")

        agg = y if y != None else x
        agg_dict = {
            "mean": pl.col(agg).mean(),
            "count": pl.col(agg).count(),
            "max": pl.col(agg).max(),
            "min": pl.col(agg).min(),
            "std": pl.col(agg).std(),
            "sum": pl.col(agg).sum(),
            "median": pl.col(agg).median(),
        }
        if x == None or y == None:
            c = x if x != None else y
            df = (
                self.filter(pl.col(c) != None)
                .select(agg_dict[estimator])
                .collect()
                .fetch()
                .to_pandas()
            )
        else:
            agg_fn = pl.col(y).mean()
            df = (
                self.filter(pl.col(x) != None)
                .select(pl.col(y) for y in selects)
                .groupby(pl.col(y) for y in groups)
                .agg(agg_dict[estimator])
                .sort(pl.col(x))
                .collect()
                .fetch()
                .to_pandas()
            )
        # run query
        if x == None:
            sns.barplot(data=df, y=y, **kwargs)
        elif y == None:
            sns.barplot(data=df, x=x, **kwargs)
        else:
            sns.barplot(data=df, x=x, y=y, **kwargs)

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
                    raise ValueError("Column ", x, " not found in dataframe")
        return Facet(inner_rdf=self, col=col, row=row, kwargs=kwargs)


@dataclass
class FetchableLazyFrame(RemoteLazyFrame):
    """
    A class to represent a FetchableLazyFrame, which can then be accessed as a Polar's dataframe via the fetch() method.
    """

    _identifier: str

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
        df = pl.DataFrame(
            [pl.Series(k, dtype=getattr(pl, v)()) for k, v in header.items()]
        )
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
        return self._meta._client._fetch_df(self._identifier)

    def save(self):
        return self._meta._client.persist_df(self._identifier)


@dataclass
class Facet:
    inner_rdf: RemoteLazyFrame
    col: Optional[str] = None
    row: Optional[str] = None
    kwargs: dict = None

    def __str__(self: LDF) -> str:
        return f"FacetGrid"

    def scatterplot(
        self: LDF,
        *args: list[str],
        **kwargs,
    ) -> None:
        """Draws a scatter plot for each subset in row/column facet grid.
        Scatterplot filters data down to necessary columns only before calling Seaborn's scatterplot function on rows of dataset
        where values match with each combination of row/grid values.

        Args:
            x (str): The name of column to be used for x axes.
            y (str): The name of column to be used for y axes.
            *args: (list[str]): Arguments to be passed to Seaborn's scatterplot function.
            **kwargs: Other keyword arguments that will be passed to Seaborn's scatterplot function.

        Raises:
            ValueError: Incorrect column name given
            various exceptions: Note that exceptions may be raised from internal Seaborn (scatterplot) or Matplotlib.pyplot functions (subplots, set_title),
            for example, if kwargs keywords are not expected. See Seaborn/Matplotlib documentation for further details.
        """
        self.__map(sns.scatterplot, *args, **kwargs)

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
        self: LDF,
        x: str = None,
        y: str = None,
        bins: int = 10,
        **kwargs,
    ) -> None:
        """Draws a histplot for each subset in row/column facet grid.

        Facet's histplot iterates over each possible combination of row/column values in the dataset, filters the dataset to rows where the values match this
        combination of row/column values and applies histplot to this dataset.

        Args:
            x (str) = None: The name of column to be used for x axes.
            y (str) = None: The name of column to be used for y axes.
            bins (int) = 10: An integer bin value which x axes will be grouped by.
            **kwargs: Other keyword arguments that will be passed to Seaborn's barplot function, in the case of one column being supplied, or heatmap function, where both x and y columns are supplied.
        Raises:
            ValueError: Incorrect column name given
            various exceptions: Note that exceptions may be raised from internal Seaborn (scatterplot) or Matplotlib.pyplot functions (subplots, set_title),
            for example, if kwargs keywords are not expected. See Seaborn/Matplotlib documentation for further details.
        """
        kwargs["bins"] = bins
        self.__bastion_map("histplot", x=x, y=y, **kwargs)

    def barplot(
        self: LDF,
        x: str = None,
        y: str = None,
        hue: str = None,
        estimator: str = "mean",
        **kwargs,
    ) -> None:
        """Draws a bar chart for each subset in row/column facet grid.

         barplot filters data down to necessary columns only and then calls Seaborn's barplot function.
        Args:
            x (str) = None: The name of column to be used for x axes.
            y (str) = None: The name of column to be used for y axes.
            estimator (str) = "mean": string represenation of estimator to be used in aggregated query. Options are: "mean", "median", "count", "max", "min", "std" and "sum"
            hue (str) = None: The name of column to be used for colour encoding.
            **kwargs: Other keyword arguments that will be passed to Seaborn's barplot function.
        Raises:
            ValueError: Incorrect column name given, no x or y values provided, estimator function not recognised
            various exceptions: Note that exceptions may be raised from Seaborn when the barplot function is called,
            for example, where kwargs keywords are not expected. See Seaborn documentation for further details.

        """
        kwargs["estimator"] = estimator
        kwargs["hue"] = hue
        self.__bastion_map("barplot", x=x, y=y, **kwargs)

    def __bastion_map(self, fn: str, x: str = None, y: str = None, **kwargs):
        # create list of all columns needed for query
        hue = kwargs["hue"] if "hue" in kwargs else None
        selects = []
        for to_add in [x, y, self.col, self.row, hue]:
            if to_add != None:
                selects.append(to_add)

        for col in selects:
            if col not in self.inner_rdf.columns:
                raise ValueError("Column ", col, " not found in dataframe")

        # get unique row and col values
        cols = []
        rows = []
        if self.col != None:
            cols = (
                self.inner_rdf.select(pl.col(self.col))
                .unique()
                .sort(pl.col(self.col))
                .collect()
                .fetch()
                .to_pandas()[self.col]
                .tolist()
            )
        if self.row != None:
            rows = (
                self.inner_rdf.select(pl.col(self.row))
                .unique()
                .sort(pl.col(self.row))
                .collect()
                .fetch()
                .to_pandas()[self.row]
                .tolist()
            )

        if fn == "histplot":
            bins = kwargs["bins"] if "bins" in kwargs else 10
            del kwargs["bins"]
        if fn == "barplot":
            del kwargs["hue"]
            estimator = kwargs["estimator"] if "estimator" in kwargs else None
            del kwargs["estimator"]

        # mapping
        r_len = len(rows) if len(rows) != 0 else 1
        c_len = len(cols) if len(cols) != 0 else 1
        if self.kwargs == None:
            fig, axes = plt.subplots(r_len, c_len, figsize=((5 * c_len), (5 * r_len)))
        else:
            if "figsize" not in self.kwargs:
                self.kwargs["figsize"] = ((5 * c_len), (5 * r_len))
            fig, axes = plt.subplots(r_len, c_len, **self.kwargs)
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
                        df.select([pl.col(x) for x in selects]).histplot(
                            x, y, bins, ax=axes[row_count, col_count], **kwargs
                        )
                    else:
                        df.select([pl.col(x) for x in selects]).barplot(
                            x,
                            y,
                            hue=hue,
                            estimator=estimator,
                            ax=axes[row_count, col_count],
                            **kwargs,
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
                    df.select([pl.col(x) for x in selects]).histplot(
                        x, y, bins, ax=axes[count], **kwargs
                    )
                else:
                    df.select([pl.col(x) for x in selects]).barplot(
                        x, y, hue=hue, estimator=estimator, ax=axes[count], **kwargs
                    )
                axes[count].set_title(t1)

    def __map(self: LDF, func, **kwargs) -> None:
        # create list of all columns needed for query
        selects = [self.col, self.row]
        if "x" in kwargs and not kwargs["x"] in selects:
            selects.append(kwargs["x"])
        if "y" in kwargs and not kwargs["y"] in selects:
            selects.append(kwargs["y"])

        for col in selects:
            if col not in self.inner_rdf.columns:
                raise ValueError("Column ", col, " not found in dataframe")

        # get unique row and col values
        cols = []
        rows = []
        if self.col != None:
            cols = (
                self.inner_rdf.select(pl.col(self.col))
                .unique()
                .sort(pl.col(self.col))
                .collect()
                .fetch()
                .to_pandas()[self.col]
                .tolist()
            )

        if self.row != None:
            rows = (
                self.inner_rdf.select(pl.col(self.row))
                .unique()
                .sort(pl.col(self.row))
                .collect()
                .fetch()
                .to_pandas()[self.row]
                .tolist()
            )

        # mapping
        r_len = len(rows) if len(rows) > 0 else 1
        c_len = len(cols) if len(cols) > 0 else 1
        if self.kwargs == None:
            fig, axes = plt.subplots(r_len, c_len, figsize=((5 * c_len), (5 * r_len)))
        else:
            if "figsize" not in self.kwargs:
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
                    sea_df = (
                        df.select([pl.col(x) for x in selects])
                        .collect()
                        .fetch()
                        .to_pandas()
                    )
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
                sea_df = (
                    df.select([pl.col(x) for x in selects])
                    .collect()
                    .fetch()
                    .to_pandas()
                )
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
    _inner: pl.internals.lazyframe.groupby.LazyGroupBy[LDF]
    _meta: Metadata
