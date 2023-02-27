from ..polars.remote_polars import RemoteArray

from ..polars.remote_polars import FetchableLazyFrame
from ..pb.bastionlab_linfa_pb2 import (
    SimpleValidationRequest,
    R2Score,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    MedianAbsoluteError,
    MaxError,
    ExplainedVariance,
    Accuracy,
    F1Score,
    Mcc,
    ClassificationMetric,
    RegressionMetric,
)


def mean_squared_error(
    y_true: "RemoteArray", y_pred: "RemoteArray"
) -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            regression_metric=RegressionMetric(
                mean_squared_error=MeanSquaredError(),
            ),
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
            regression_metric=RegressionMetric(
                mean_squared_log_error=MeanSquaredLogError()
            ),
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def r2_score(y_true: "RemoteArray", y_pred: "RemoteArray") -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            regression_metric=RegressionMetric(r2_score=R2Score()),
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
            regression_metric=RegressionMetric(mean_absolute_error=MeanAbsoluteError()),
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
            regression_metric=RegressionMetric(
                median_absolute_error=MedianAbsoluteError()
            ),
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def max_error(y_true: "RemoteArray", y_pred: "RemoteArray") -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            regression_metric=RegressionMetric(
                max_error=MaxError(),
            ),
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
            regression_metric=RegressionMetric(
                explained_variance=ExplainedVariance(),
            ),
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
            classification_metric=ClassificationMetric(accuracy=Accuracy()),
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def f1_score(y_true: "RemoteArray", y_pred: "RemoteArray") -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            classification_metric=ClassificationMetric(f1_score=F1Score()),
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)


def mcc(y_true: "RemoteArray", y_pred: "RemoteArray") -> "FetchableLazyFrame":
    from ..polars.remote_polars import FetchableLazyFrame

    res = y_true._client.linfa.stub.Validate(
        SimpleValidationRequest(
            truth=y_true.identifier,
            prediction=y_pred.identifier,
            classification_metric=ClassificationMetric(mcc=Mcc()),
        )
    )

    return FetchableLazyFrame._from_reference(y_true._client.polars, res)
