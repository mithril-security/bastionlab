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
    def serialize(self) -> str:
        raise NotImplementedError()


@dataclass
class EntryPointPlanSegment(CompositePlanSegment):
    _inner: str

    def serialize(self) -> str:
        return f'{{"EntryPointPlanSegment":"{self._inner}"}}'


@dataclass
class PolarsPlanSegment(CompositePlanSegment):
    _inner: LDF

    def serialize(self) -> str:
        return f'{{"PolarsPlanSegment":{self._inner.write_json()}}}'


@dataclass
class UdfPlanSegment(CompositePlanSegment):
    _inner: ScriptFunction
    _columns: List[str]

    def serialize(self) -> str:
        columns = ",".join([f'"{c}"' for c in self._columns])
        b64str = base64.b64encode(self._inner.save_to_buffer()).decode("ascii")
        return f'{{"UdfPlanSegment":{{"columns":[{columns}],"udf":"{b64str}"}}}}'


@dataclass
class Metadata:
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
    _inner: pl.LazyFrame
    _meta: Metadata

    def __str__(self: LDF) -> str:
        return f"RemoteLazyFrame"

    def __repr__(self: LDF) -> str:
        return str(self)

    def clone(self: LDF) -> LDF:
        return RemoteLazyFrame(self._inner.clone(), self._meta)

    @property
    def composite_plan(self: LDF) -> str:
        segments = ",".join(
            [
                seg.serialize()
                for seg in [*self._meta._prev_segments, PolarsPlanSegment(self._inner)]
            ]
        )
        return f"[{segments}]"

    def collect(self: LDF) -> LDF:
        return self._meta._client._run_query(self.composite_plan)

    def apply_udf(self: LDF, columns: List[str], udf: Callable) -> LDF:
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
                [*self._meta._prev_segments, *other._meta._prev_segments],
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
                [*self._meta._prev_segments, *other._meta._prev_segments],
            ),
        )

    def histplot(self: LDF, **kwargs):
        # if x or y is empty, replace with "count" but if no x or y, return
        col_x = "count" if not "x" in kwargs else kwargs["x"]
        col_y = "count" if not "y" in kwargs else kwargs["y"]
        if col_x == "count" and col_y == "count":
            print("Please provide an 'x' or 'y' value")
            return

        # avoid duplicated in Seaborn calls
        for x in ["x", "data", "y"]:
            if x in kwargs:
                del kwargs[x]

        model = ApplyBins(kwargs["bins"])
        # we are done with bins now and don't want to foward this to the barplot() function
        del kwargs["bins"]
        # if we have only X or Y
        if col_x == "count" or col_y == "count":
            q_x = pl.col(col_x) if col_x != "count" else pl.col(col_y)
            q_y = pl.count()

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

    def curveplot(
        self: LDF,
        x: str,
        y: str,
        order: int = 3,
        ci: Union[int, None] = None,
        scatter: bool = False,
        **kwargs,
    ):
        # get df with necessary columns
        df = self.select([pl.col(x), pl.col(y)]).collect().fetch().to_pandas()
        sns.regplot(data=df, x=x, y=y, order=order, ci=ci, scatter=scatter, **kwargs)

    def scatterplot(self: LDF, x: str, y: str, **kwargs):
        # if there is a hue or style argument add them to cols
        cols = [x, y]
        if "hue" in kwargs:
            if not kwargs["hue"] in cols:
                cols.append(kwargs["hue"])
        if "style" in kwargs:
            if not kwargs["style"] in cols:
                cols.append(kwargs["style"])
        # get df with necessary columns
        df = self.select([pl.col(x) for x in cols]).collect().fetch().to_pandas()
        # run query
        sns.scatterplot(data=df, x=x, y=y, **kwargs)

    def facet(
        self: LDF, col: Optional[str] = None, row: Optional[str] = None, *args, **kwargs
    ) -> any:
        return Facet(inner_rdf=self, col=col, row=row, kwargs=kwargs)


@dataclass
class FetchableLazyFrame(RemoteLazyFrame):
    _identifier: str

    @property
    def identifier(self) -> str:
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
        return self._meta._client._fetch_df(self._identifier)


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
        self.__map(sns.scatterplot, *args, **kwargs)

    def curveplot(
        self: LDF,
        *args: list[str],
        order: int = 3,
        ci: int | None = None,
        scatter: bool = False,
        **kwargs,
    ) -> None:
        self.__map(sns.regplot, *args, order=order, ci=ci, scatter=scatter, **kwargs)

    def histplot(
        self: LDF,
        bins: int = 10,
        *args: list[str],
        **kwargs,
    ) -> None:
        self.__map(self.inner_rdf.histplot, bins=bins, *args, **kwargs)

    def __map(self: LDF, func, **kwargs) -> None:
        # create list of all columns needed for query
        selects = [self.col, self.row]
        if "x" in kwargs and not kwargs["x"] in selects:
            selects.append(kwargs["x"])
        if "y" in kwargs and not kwargs["y"] in selects:
            selects.append(kwargs["y"])

        # get unique row and col values
        cols = (
            self.inner_rdf.select(pl.col(self.col))
            .unique()
            .sort(pl.col(self.col))
            .collect()
            .fetch()
            .to_pandas()[self.col]
            .tolist()
        )
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
        fig, axes = plt.subplots(len(rows), len(cols), figsize=(15, 20))
        col_count = 0
        for col in cols:
            row_count = 0
            for row in rows:
                df = self.inner_rdf.clone().filter(
                    (pl.col(self.col) == col) & (pl.col(self.row) == row)
                )
                t1 = self.row + ": " + str(row) + " | " + self.col + ": " + str(col)

                # decipher if function is a Seaborn or BastionLab function to forward pandas or RDF as data
                func_module = str(getattr(func, "__module__", ""))
                if func_module.startswith("seaborn"):
                    sea_df = (
                        df.select([pl.col(x) for x in selects])
                        .collect()
                        .fetch()
                        .to_pandas()
                    )
                    func(data=sea_df, ax=axes[row_count, col_count], **kwargs)
                else:
                    df.select([pl.col(x) for x in selects]).histplot(
                        ax=axes[row_count, col_count], **kwargs
                    )
                axes[row_count, col_count].set_title(t1)
                row_count = row_count + 1
            col_count = col_count + 1


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
