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
class StackPlanSegment(CompositePlanSegment):
    def serialize(self) -> str:
        return '"StackPlanSegment"'


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

    def vstack(self: LDF, df2: LDF) -> LDF:
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

    def histplot(self: LDF, col_x: str, col_y: str, bins: int = 10, **kwargs):
        model = ApplyBins(bins)
        df = (
            self.filter(pl.col(col_x) != None)
            .filter(pl.col(col_y) != None)
            .select([pl.col(col_y), pl.col(col_x)])
            .apply_udf([col_x], model)
            .groupby(pl.col(col_x))
            .agg(pl.col(col_y).count())
            .sort(col_x)
            .collect()
            .fetch()
            .to_pandas()
        )
        sns.barplot(x=df[col_x], y=df[col_y], **kwargs)

    def curveplot(
        self: LDF,
        col_x: str,
        col_y: str,
        bins: int = 10,
        order: int = 3,
        ci: Union[int, None] = None,
        scatter: bool = False,
        **kwargs,
    ):
        model = ApplyBins(bins)
        df = (
            self.filter(pl.col(col_x) != None)
            .filter(pl.col(col_y) != None)
            .select([pl.col(col_y), pl.col(col_x)])
            .apply_udf([col_x], model)
            .groupby(pl.col(col_x))
            .agg(pl.col(col_y).count())
            .sort(col_x)
            .collect()
            .fetch()
            .to_pandas()
        )
        sns.regplot(
            x=df[col_x], y=df[col_y], order=order, ci=ci, scatter=scatter, **kwargs
        )

    def scatterplot(self: LDF, col_x: str, col_y: str, bins=5, **kwargs):
        model = ApplyBins(bins)
        df = (
            self.filter(pl.col(col_x) != None)
            .filter(pl.col(col_y) != None)
            .select([pl.col(col_y), pl.col(col_x)])
            .apply_udf([col_x], model)
            .groupby(pl.col(col_x))
            .agg(pl.col(col_y).count())
            .sort(col_x)
            .collect()
            .fetch()
            .to_pandas()
        )
        sns.scatterplot(x=df[col_x], y=df[col_y], **kwargs)


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
