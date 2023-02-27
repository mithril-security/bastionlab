from ..polars.remote_polars import RemoteArray

from ..polars.remote_polars import FetchableLazyFrame
from ..pb.bastionlab_linfa_pb2 import SimpleValidationRequest


def mean_squared_error(
    y_true: "RemoteArray", y_pred: "RemoteArray"
) -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            scoring="mean_squared_error",
            metric_type="regression",
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def mean_squared_log_error(
    y_true: "RemoteArray", y_pred: "RemoteArray"
) -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            scoring="mean_squared_log_error",
            metric_type="regression",
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def r2_score(y_true: "RemoteArray", y_pred: "RemoteArray") -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            scoring="r2_score",
            metric_type="regression",
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def mean_absolute_error(
    y_true: "RemoteArray", y_pred: "RemoteArray"
) -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            scoring="mean_absolute_error",
            metric_type="regression",
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def median_absolute_error(
    y_true: "RemoteArray", y_pred: "RemoteArray"
) -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            scoring="median_absolute_error",
            metric_type="regression",
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def max_error(y_true: "RemoteArray", y_pred: "RemoteArray") -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            scoring="max_error",
            metric_type="regression",
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def explained_variance(
    y_true: "RemoteArray", y_pred: "RemoteArray"
) -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            scoring="explained_variance",
            metric_type="regression",
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def accuracy_score(
    y_true: "RemoteArray", y_pred: "RemoteArray"
) -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            scoring="accuracy",
            metric_type="classification",
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def f1_score(y_true: "RemoteArray", y_pred: "RemoteArray") -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            scoring="f1_score",
            metric_type="classification",
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)
