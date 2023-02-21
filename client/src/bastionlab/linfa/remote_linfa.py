from dataclasses import dataclass, field
from ..pb.bastionlab_linfa_pb2 import (
    ModelResponse,
    LinearRegression as ProtoLinReg,
    TweedieRegressor as ProtoTweedieRegressor,
    BinomialLogisticRegression as ProtoBinLogReg,
    MultinomialLogisticRegression as ProtoMultiLogReg,
    KMeans as ProtoKMeans,
    DecisionTree as ProtoDecisionTree,
    ElasticNet as ProtoElas,
    GaussianNb as ProtoGaussian,
    SVM,
    _SVM_KERNELPARAMS_N as N,
)
from ..polars.remote_polars import RemoteArray
from typing import Dict, Optional, List


@dataclass
class FittedModel:
    _identifier: str
    _model_type: str

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def type(self) -> str:
        return self._model_type

    @staticmethod
    def _from_reference(ref: ModelResponse, model_type: str) -> "FittedModel":
        return FittedModel(_identifier=ref.identifier, _model_type=model_type)

    def __repr__(self) -> str:
        return f"FittedModel(identifier={self._identifier})\n  └── {self._model_type}"


@dataclass
class Trainer:
    _fitted_model: "FittedModel" = None
    _name: str = None

    def to_msg_dict():
        raise NotImplementedError

    def set_name(self, name: str):
        self._name = name

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        client = train_set._client.linfa
        model = client._train(train_set, target_set, self)
        self._fitted_model = model

    def predict(self, prediction_input: "RemoteArray"):
        client = prediction_input._client.linfa
        return client._predict(self._fitted_model, prediction_input)

    def predict_proba(self, prediction_input: "RemoteArray"):
        client = prediction_input._client.linfa
        return client._predict_proba(self._fitted_model, prediction_input)

    def __str__(self) -> str:
        return f"{self._fitted_model}"


class GaussianNb(Trainer):
    def __init__(self):
        self.var_smoothing: float = 1e-9

    def to_msg_dict(self):
        return {"gaussian_nb": ProtoGaussian(var_smoothing=self.var_smoothing)}

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        return super().fit(train_set, target_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)


class LinearRegression(Trainer):
    def __init__(self):
        super().set_name(__class__.__name__)
        self.fit_intercept: bool = True

    def to_msg_dict(self):
        return {"linear_regression": ProtoLinReg(fit_intercept=self.fit_intercept)}

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        super().fit(train_set, target_set)

    def predict(self, prediction_input: "RemoteArray"):
        return super().predict(prediction_input)


class Link(Trainer):
    def to_msg_dict():
        raise NotImplementedError()


class Identity(Link):
    def to_msg_dict(self):
        {"identity": ProtoTweedieRegressor.Identity()}


class Log(Link):
    def to_msg_dict(self):
        {"log": ProtoTweedieRegressor.Log()}


class Logit(Link):
    def to_msg_dict(self):
        {"logit": ProtoTweedieRegressor.Logit()}


@dataclass
class TweedieRegressor(Trainer):
    fit_intercept: bool = False
    alpha: float = (1.0,)
    max_iter: int = 100
    link: Optional[Link] = None
    tol: float = (1e-4,)
    power: float = 1.0

    def to_msg_dict(self):
        link = (
            Identity() if self.power <= 0 else Log() if self.link is None else self.link
        )
        link = link.to_msg_dict()
        return {
            "tweedie_regressor": {
                ProtoTweedieRegressor(
                    fit_intercept=self.fit_intercept,
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    power=self.power,
                    **link,
                )
            }
        }


@dataclass
class LogisticRegression(Trainer):
    alpha: float = 1.0
    gradient_tolerance: float = 1e-4
    fit_intercept: bool = True
    max_iterations: int = 100
    initial_params: List[float] = field(default_factory=list)
    shape: List[float] = field(default_factory=list)
    multi_class: str = "binomial"

    def to_msg_dict(self):
        if self.multi_class not in ["binomial", "multinomial"]:
            raise TypeError(f"Unsupported Class Type: {self.multi_class}")
        return (
            dict(
                binomial_logistic_regression=ProtoBinLogReg(
                    alpha=self.alpha,
                    gradient_tolerance=self.gradient_tolerance,
                    fit_intercept=self.fit_intercept,
                    max_iterations=self.max_iterations,
                    initial_params=self.initial_params,
                )
            )
            if self.multi_class == "binomial"
            else dict(
                multinomial_logistic_regression=ProtoMultiLogReg(
                    alpha=self.alpha,
                    gradient_tolerance=self.gradient_tolerance,
                    fit_intercept=self.fit_intercept,
                    max_iterations=self.max_iterations,
                    initial_params=self.initial_params,
                    shape=self.shape,
                )
            )
        )

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        return super().fit(train_set, target_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)

    def predict_proba(self, prediction_input: "RemoteArray"):
        return super().predict_proba(prediction_input)


@dataclass
class ElasticNet(Trainer):
    penalty: float = 0.1
    l1_ratio: float = 0.1
    with_intercept: bool = False
    max_iterations: int = 1000
    tolerance: float = 1e-4

    def to_msg_dict(self):
        return {
            "elastic_net": ProtoElas(
                penalty=self.penalty,
                l1_ratio=self.l1_ratio,
                with_intercept=self.with_intercept,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )
        }

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        return super().fit(train_set, target_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)


@dataclass
class DecisionTree(Trainer):
    class SplitQuality(Trainer):
        def to_msg_dict():
            raise NotImplementedError()

    class Gini(SplitQuality):
        def to_msg_dict():
            return {"gini": ProtoDecisionTree.Gini}

    class Entropy(SplitQuality):
        def to_msg_dict():
            return {"entropy": ProtoDecisionTree.Entropy}

    split_quality: Optional[SplitQuality] = Gini
    max_depth: Optional[int] = None
    min_weight_split: float = (2.0,)
    min_weight_leaf: float = (1.0,)
    min_impurity_decrease: float = 0.00001

    def to_msg_dict(self):
        return {
            "decision_tree": ProtoDecisionTree(
                max_depth=self.max_depth,
                min_weight_split=self.min_weight_split,
                min_weight_leaf=self.min_weight_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                split_quality=self.split_quality.to_msg_dict(),
            )
        }

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        return super().fit(train_set, target_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)


@dataclass
class KMeans(Trainer):
    class InitMethod(Trainer):
        def to_msg_dict():
            raise NotImplementedError()

    class Random(InitMethod):
        def to_msg_dict():
            return {"random": ProtoKMeans.Random()}

    class KMeanPara(InitMethod):
        def to_msg_dict():
            return {"kmeans_para": ProtoKMeans.KMeansPara()}

    class KMeansPlusPlus(InitMethod):
        def to_msg_dict():
            return {"kmeans_plus_plus": ProtoKMeans.KMeansPlusPlus()}

    n_runs: int = 10
    n_clusters: int = 0
    tolerance: float = 1e-4
    max_n_iterations: int = 300
    init_method: Optional[InitMethod] = None

    def get_init_method(self) -> Dict:
        return (
            self.Random.to_msg_dict()
            if not self.init_method
            else self.init_method.to_msg_dict()
        )

    def to_msg_dict(self):
        return {
            "kmeans": ProtoKMeans(
                n_runs=self.n_runs,
                n_clusters=self.n_clusters,
                tolerance=self.tolerance,
                max_n_iterations=self.max_n_iterations,
                **self.get_init_method(),
            )
        }

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        return super().fit(train_set, target_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)


@dataclass
class SVM(Trainer):
    class PlattParams(Trainer):
        maxiter: int = 100
        ministep: float = 1e-10
        sigma: float = 1e-12

        def to_msg_dict(self):
            return {
                "platt_params": SVM.PlattParams(
                    maxiter=self.maxiter, ministep=self.ministep, sigma=self.sigma
                )
            }

    class KernelParams(Trainer):
        class KernelMethod(Trainer):
            pass

        class KernelType(Trainer):
            pass

        class Dense(KernelType):
            def to_msg_dict():
                return {"dense": SVM.KernelParams.Dense()}

        class Sparse(KernelType):
            sparsity: int = 1

            def to_msg_dict(self):
                return {"sparse": SVM.KernelParams.Sparse(sparsity=self.sparsity)}

        class Gaussian(KernelMethod):
            eps: float = 1.0

            def to_msg_dict(self):
                return {"gaussian": SVM.KernelParams.Guassian(eps=self.eps)}

        class Linear(KernelMethod):
            def to_msg_dict():
                return {"linear": SVM.KernelParams.Linear()}

        class Polynomial(KernelMethod):
            constant: float = 1.0
            degree: float = 1.0

            def to_msg_dict(self):
                return {
                    "poly": SVM.KernelParams.Polynomial(
                        constant=self.constant, degree=self.degree
                    )
                }

        kernel_method: Optional[KernelMethod] = Linear.to_msg_dict()
        kernel_type: Optional[KernelType] = Dense.to_msg_dict()
        n: N = SVM.KernelParams.LinearSearch

        def to_msg_dict(self):
            return {
                "kernel_params": SVM.KernelParams(
                    **self.kernel_method, **self.kernel_type, n=self.n
                )
            }

    c: List[float] = field(default_factory=list)
    eps: Optional[float] = None
    nu: Optional[float] = None
    shrinking: bool = False
    platt_params: Optional[PlattParams] = None
    kernel_params: Optional[KernelParams] = None

    def get_kernel_params(self):
        return (
            self.KernelParams().to_msg_dict()
            if not self.kernel_params
            else self.kernel_params.to_msg_dict()
        )

    def get_platt_params(self):
        return (
            self.PlattParams().to_msg_dict()
            if not self.platt_params
            else self.platt_params.to_msg_dict()
        )

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        return super().fit(train_set, target_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)

    def to_msg_dict(self):
        return {
            "svm": SVM(
                c=self.c,
                eps=self.eps,
                shrinking=self.shrinking,
                nu=self.nu,
                **self.get_kernel_params(),
                **self.get_platt_params(),
            )
        }
