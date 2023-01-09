from dataclasses import dataclass, field
from ..pb.bastionlab_linfa_pb2 import TrainingRequest
from ..polars.remote_polars import RemoteArray
from typing import Dict, Optional, List
from ..config import CONFIG


@dataclass
class Trainer:
    identifier: Optional[str] = None

    def to_msg_dict():
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


def get_client():
    from ..config import CONFIG

    if CONFIG["linfa_client"] == None:
        raise Exception("BastionLab Linfa client is not initialized.")

    return CONFIG["linfa_client"]


@dataclass
class GaussianNb(Trainer):
    var_smoothing: float = 1e-9

    def to_msg_dict(self):
        return {
            "gaussian_nb": TrainingRequest.GaussianNb(var_smoothing=self.var_smoothing)
        }

    def fit(self, train_set: RemoteArray, target_set: RemoteArray):
        client = get_client()
        model = client._train(train_set, target_set, self)
        self.identifier = model.identifier
        return model

    def predict(self, test_set: RemoteArray):
        client = get_client()
        return client._predict(self, test_set)


@dataclass
class LinearRegression(Trainer):
    fit_intercept: bool = True

    def to_msg_dict(self):
        return {
            "linear_regression": TrainingRequest.LinearRegression(
                fit_intercept=self.fit_intercept
            )
        }

    def fit(self, train_set: RemoteArray, target_set: RemoteArray):
        client = get_client()
        model = client._train(train_set, target_set, self)
        self.identifier = model.identifier
        return model

    def predict(self, test_set: RemoteArray):
        client = get_client()
        return client._predict(self, test_set)


@dataclass
class LogisticRegression(Trainer):
    alpha: float = 1.0
    gradient_tolerance: float = 1e-4
    fit_intercept: bool = True
    max_iterations: int = 100
    initial_params: List[float] = field(default_factory=list)

    def to_msg_dict(self):
        return {
            "logistic_regression": TrainingRequest.LogisticRegression(
                alpha=self.alpha,
                gradient_tolerance=self.gradient_tolerance,
                fit_intercept=self.fit_intercept,
                max_iterations=self.max_iterations,
                initial_params=self.initial_params,
            )
        }

    def fit(self, train_set: RemoteArray, target_set: RemoteArray):
        client = get_client()
        model = client._train(train_set, target_set, self)
        self.identifier = model.identifier
        return model

    def predict(self, test_set: RemoteArray):
        client = get_client()
        return client._predict(self, test_set)


@dataclass
class ElasticNet(Trainer):
    penalty: float = 0.1
    l1_ratio: float = 0.1
    with_intercept: bool = False
    max_iterations: int = 1000
    tolerance: float = 1e-4

    def to_msg_dict(self):
        return {
            "elastic_net": TrainingRequest.ElasticNet(
                penalty=self.penalty,
                l1_ratio=self.l1_ratio,
                with_intercept=self.with_intercept,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance,
            )
        }

    def fit(self, train_set: RemoteArray, target_set: RemoteArray):
        client = get_client()
        model = client._train(train_set, target_set, self)
        self.identifier = model.identifier
        return model

    def predict(self, test_set: RemoteArray):
        client = get_client()
        return client._predict(self, test_set)


@dataclass
class DecisionTree(Trainer):
    class SplitQuality(Trainer):
        def to_msg_dict():
            pass

    class Gini(SplitQuality):
        def to_msg_dict():
            return {"gini": TrainingRequest.DecisionTree.Gini}

    class Entropy(SplitQuality):
        def to_msg_dict():
            return {"entropy": TrainingRequest.DecisionTree.Entropy}

    split_quality: Optional[SplitQuality] = Gini
    max_depth: Optional[int] = None
    min_weight_split: float = (2.0,)
    min_weight_leaf: float = (1.0,)
    min_impurity_decrease: float = 0.00001

    def to_msg_dict(self):
        return {
            "decision_tree": TrainingRequest.DecisionTree(
                max_depth=self.max_depth,
                min_weight_split=self.min_weight_split,
                min_weight_leaf=self.min_weight_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                split_quality=self.split_quality.to_msg_dict(),
            )
        }

    def fit(self, train_set: RemoteArray, target_set: RemoteArray):
        client = get_client()
        model = client._train(train_set, target_set, self)
        self.identifier = model.identifier
        return model

    def predict(self, test_set: RemoteArray):
        client = get_client()
        return client._predict(self, test_set)


@dataclass
class KMeans(Trainer):
    class InitMethod(Trainer):
        def to_msg_dict():
            pass

    class Random(InitMethod):
        def to_msg_dict():
            return {"random": TrainingRequest.KMeans.Random()}

    class KMeanPara(InitMethod):
        def to_msg_dict():
            return {"kmeans_para": TrainingRequest.KMeans.KMeansPara()}

    class KMeansPlusPlus(InitMethod):
        def to_msg_dict():
            return {"kmeans_plus_plus": TrainingRequest.KMeans.KMeansPlusPlus()}

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
            "kmeans": TrainingRequest.KMeans(
                n_runs=self.n_runs,
                n_clusters=self.n_clusters,
                tolerance=self.tolerance,
                max_n_iterations=self.max_n_iterations,
                **self.get_init_method(),
            )
        }

    def fit(self, train_set: RemoteArray, target_set: RemoteArray):
        client = get_client()
        model = client._train(train_set, target_set, self)
        self.identifier = model.identifier
        return model

    def predict(self, test_set: RemoteArray):
        client = get_client()
        return client._predict(self, test_set)
