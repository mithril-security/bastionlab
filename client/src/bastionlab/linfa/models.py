from dataclasses import dataclass, field
from ..pb.bastionlab_linfa_pb2 import (
    LinearRegression as ProtoLinReg,
    TweedieRegressor as ProtoTweedieRegressor,
    BinomialLogisticRegression as ProtoBinLogReg,
    MultinomialLogisticRegression as ProtoMultiLogReg,
    KMeans as ProtoKMeans,
    DecisionTree as ProtoDecisionTree,
    ElasticNet as ProtoElas,
    GaussianNb as ProtoGaussian,
)
from ..polars.remote_polars import RemoteArray
from typing import Dict, Optional, TYPE_CHECKING
import numpy as np
from .remote_linfa import Trainer

if TYPE_CHECKING:
    from .remote_linfa import Trainer


@dataclass
class GaussianNB(Trainer):
    """
    Gaussian Naïve Bayes algorithm implementation for BastionLab.

    Args:
        var_smoothing: float
            Portion of the largest variance of all features that is added to variances for calculation stability.
    """

    var_smoothing: float = 1e-9

    def _to_msg_dict(self):
        return {"gaussian_nb": ProtoGaussian(var_smoothing=self.var_smoothing)}

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        return super().fit(train_set, target_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)

    def predict_proba(self, _):
        raise NotImplementedError("predict_proba isn't implemented for GaussianNB")


class LinearRegression(Trainer):
    """
    Ordinary least squares Linear Regression.

    LinearRegression fits a linear model with coefficients w = (w1, …, wp) to minimize
    the residual sum of squares between the observed targets in the dataset, and the targets
    predicted by the linear approximation.

    Args:
        fit_intercept: bool
            Whether to calculate the intercept for this model.
            If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
    """

    def __init__(self):
        super().set_name(__class__.__name__)
        self.fit_intercept: bool = True

    def _to_msg_dict(self):
        return {"linear_regression": ProtoLinReg(fit_intercept=self.fit_intercept)}

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        super().fit(train_set, target_set)

    def predict(self, prediction_input: "RemoteArray"):
        return super().predict(prediction_input)


class Link(Trainer):
    def _to_msg_dict():
        raise NotImplementedError()


class Identity(Link):
    def _to_msg_dict(self):
        {"identity": ProtoTweedieRegressor.Identity()}


class Log(Link):
    def _to_msg_dict(self):
        {"log": ProtoTweedieRegressor.Log()}


class Logit(Link):
    def _to_msg_dict(self):
        {"logit": ProtoTweedieRegressor.Logit()}


@dataclass
class TweedieRegressor(Trainer):
    fit_intercept: bool = False
    alpha: float = (1.0,)
    max_iter: int = 100
    link: Optional[Link] = None
    tol: float = (1e-4,)
    power: float = 1.0

    def _to_msg_dict(self):
        link = (
            Identity() if self.power <= 0 else Log() if self.link is None else self.link
        )
        link = link._to_msg_dict()
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
    """
    Logistic Regression Classifier

    Args:
        alpha: float, default=1.0
            Inverse of regularization strength
        tol: float, default=1e-4
            Tolerance for stopping criteria.
        fit_intercept: bool, default=True
            Specifies if a constant (a.k.a. bias or intercept)
            should be added to the decision function.
        max_iter: int, default=100
            Maximum number of iterations taken for the solvers to converge.
        multi_class: {'binomial', 'multinomial'}, default="binomial"
            Selects between `multinomial` and `binomial` logistic regression.
    """

    alpha: float = 1.0
    tol: float = 1e-4
    fit_intercept: bool = True
    max_iter: int = 100
    multi_class: str = "binomial"
    initial_params: np.array = np.array(
        []
    )  # Undocumented Yet: dis-ambiguity between Linfa and sklearn

    def _to_msg_dict(self):
        algos = dict(
            binomial=dict(
                binomial_logistic_regression=ProtoBinLogReg(
                    alpha=self.alpha,
                    gradient_tolerance=self.tol,
                    fit_intercept=self.fit_intercept,
                    max_iterations=self.max_iter,
                    initial_params=self.initial_params,
                    shape=self.initial_params.shape,
                )
            ),
            multinomial=dict(
                multinomial_logistic_regression=ProtoMultiLogReg(
                    alpha=self.alpha,
                    gradient_tolerance=self.tol,
                    fit_intercept=self.fit_intercept,
                    max_iterations=self.max_iter,
                    initial_params=self.initial_params,
                    shape=self.initial_params.shape,
                )
            ),
        )
        if self.multi_class not in list(algos.keys()):
            raise TypeError(f"Unsupported Class Type: {self.multi_class}")
        return algos[self.multi_class]

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        return super().fit(train_set, target_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)

    def predict_proba(self, prediction_input: "RemoteArray"):
        return super().predict_proba(prediction_input)


@dataclass
class ElasticNet(Trainer):
    """Undocumented"""

    penalty: float = 0.1
    l1_ratio: float = 0.1
    with_intercept: bool = False
    max_iterations: int = 1000
    tolerance: float = 1e-4

    def _to_msg_dict(self):
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
class DecisionTreeClassifier(Trainer):
    """
    Decision Tree Classifier

    Args:
        criterion: {"gini", "entropy"}, default="gini"
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity
            and “entropy” for the Shannon information gain, see [Mathematical formulation](https://scikit-learn.org/stable/modules/tree.html#tree-mathematical-formulation).
        max_depth: int, default=None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_weight_split samples.
        min_weight_split: float, default=2.0
            The minimum number of samples required to split an internal node.
        min_weight_leaf: float, default=1.0
            The minimum number of samples required to be at a leaf node.
        min_impurity_decrease: float, default=1e-5
            A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    """

    criterion: Optional[str] = "gini"
    max_depth: Optional[int] = None
    min_weight_split: float = 2.0
    min_weight_leaf: float = 1.0
    min_impurity_decrease: float = 1e-5

    def _to_msg_dict(self):
        if self.criterion not in ["gini", "entropy"]:
            raise ValueError(
                "Please choose one of these Split Qualities: [gini, entropy]"
            )
        elif self.criterion == "log_loss":
            raise NotImplementedError("`log_loss` not supported by BastionLab ")

        criteria = dict(
            gini=dict(gini=ProtoDecisionTree.Gini()),
            entropy=dict(entropy=ProtoDecisionTree.Entropy()),
        )
        return {
            "decision_tree": ProtoDecisionTree(
                max_depth=self.max_depth,
                min_weight_split=self.min_weight_split,
                min_weight_leaf=self.min_weight_leaf,
                min_impurity_decrease=self.min_impurity_decrease,
                **criteria[self.criterion],
            )
        }

    def fit(self, train_set: "RemoteArray", target_set: "RemoteArray"):
        return super().fit(train_set, target_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)


@dataclass
class KMeans(Trainer):
    n_init: int = 10
    n_clusters: int = 0
    tolerance: float = 1e-4
    init: "str" = "k-means++"
    max_iter: int = 300
    random_state: int = 42

    def get_init(self) -> Dict:
        init = {
            "k-means++": dict(kmeans_plus_plus=ProtoKMeans.KMeansPlusPlus()),
            "random": dict(random=ProtoKMeans.Random()),
        }

        if self.init not in list(init.keys()):
            raise ValueError(
                f"Please provide one of these init methods: [k-means++, random]"
            )
        return init[self.init]

    def _to_msg_dict(self):
        return {
            "kmeans": ProtoKMeans(
                n_runs=self.n_init,
                n_clusters=self.n_clusters,
                tolerance=self.tolerance,
                max_n_iterations=self.max_iter,
                random_state=self.random_state,
                **self.get_init(),
            )
        }

    def fit(self, train_set: "RemoteArray"):
        return super().fit(train_set, train_set)

    def predict(self, test_set: "RemoteArray"):
        return super().predict(test_set)
