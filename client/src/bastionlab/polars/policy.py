from dataclasses import dataclass
from typing import List, Union
from serde import serde, InternalTagging


class Rule:
    pass


@dataclass
class AtLeastNOf(Rule):
    """
    specifies a collection of `Rule`s.

    Args:
        n : int
            Specifies the number of rules to combine.
        of : List[Rule]
            Collection of rules.
    """

    n: int
    of: Union["AtLeastNOf", "Aggregation", "TrueRule", "FalseRule", "UserId"]


@dataclass
@serde
class UserId(Rule):
    """
    BastionLab instruction `Rule` that attaches a user identifier to the safe zone.

    Args:
        id : str
            User Identifier.
    """

    id: str


@dataclass
@serde
class Aggregation(Rule):
    """
    Specifies a `Rule` for the number of rows an operation on a Remote DataFrame can aggregate.

    Args:
        min_agg_size : int
            The minimum allowable row aggregation size.
    """

    min_agg_size: int


@dataclass
@serde
class TrueRule(Rule):
    """
    BastionLab instruction `Rule` for boolean logic `True`.
    """


@dataclass
@serde
class FalseRule(Rule):
    """
    BastionLab instruction `Rule` for boolean logic `False`.
    """


class UnsafeAction:
    pass


@dataclass
@serde
class Log(UnsafeAction):
    """
    Instructs the BastionLab server to _log_ the operation performed on the associated
    Remote DataFrame.
    """


@dataclass
@serde
class Review(UnsafeAction):
    """
    Instructs the BastionLab server to request a review from the owner of the Remote DataFrame.
    """


@dataclass
@serde
class Reject(UnsafeAction):
    """
    Instructs the BastionLab server to reject the request.
    """


serde(AtLeastNOf)


@serde(tagging=InternalTagging("type"))
@dataclass
class Policy:
    """
    BastionLab Policy class.

    Express the allowable operations on Remote DataFrames [RDFs] (i.e., DataFrames on the BastionLab server).

    Args:
        safe_zone : Rule
            Describes what operations are considered _safe_ on the RDF.
        unsafe_handling : UnsafeAction
            Describes what should happen if a user violates the `safe_zone`. For example (logging operations)
    """

    safe_zone: Union[AtLeastNOf, Aggregation, TrueRule, FalseRule, UserId]
    unsafe_handling: Union[Reject, Review, Log]
    savable: bool


DEFAULT_POLICY = Policy(Aggregation(10), Review(), True)
"""
Default BastionLab Client Policy `Policy(Aggregation(10), Review(), True)`
"""

__all__ = [
    "Rule",
    "AtLeastNOf",
    "UserId",
    "Aggregation",
    "TrueRule",
    "FalseRule",
    "UnsafeAction",
    "Log",
    "Review",
    "Reject",
    "Policy",
    "DEFAULT_POLICY",
]
