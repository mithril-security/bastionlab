from dataclasses import dataclass
from typing import List, Any


class Rule:
    def _serialize(self) -> Any:
        raise NotImplementedError


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
    of: List[Rule]

    def _serialize(self) -> Any:
        return {"AtLeastNOf": [self.n, [x._serialize() for x in self.of]]}


@dataclass
class UserId(Rule):
    """
    BastionLab instruction `Rule` that attaches a user identifier to the safe zone.

    Args:
        id : str
            User Identifier.
    """

    id: str

    def _serialize(self) -> Any:
        return {"UserId": self.id}


@dataclass
class Aggregation(Rule):
    """
    Specifies a `Rule` for the number of rows an operation on a Remote DataFrame can aggregate.

    Args:
        min_agg_size : int
            The minimum allowable row aggregation size.
    """

    min_agg_size: int

    def _serialize(self) -> Any:
        return {"Aggregation": self.min_agg_size}


@dataclass
class TrueRule(Rule):
    """
    BastionLab instruction `Rule` for boolean logic `True`.
    """

    def _serialize(self) -> Any:
        return "True"


@dataclass
class FalseRule(Rule):
    """
    BastionLab instruction `Rule` for boolean logic `False`.
    """

    def _serialize(self) -> Any:
        return "False"


class UnsafeAction:
    def _serialize(self) -> Any:
        raise NotImplementedError


@dataclass
class Log(UnsafeAction):
    """
    Instructs the BastionLab server to _log_ the operation performed on the associated
    Remote DataFrame.
    """

    def _serialize(self) -> Any:
        return "Log"


@dataclass
class Review(UnsafeAction):
    """
    Instructs the BastionLab server to request a review from the owner of the Remote DataFrame.
    """

    def _serialize(self) -> Any:
        return "Review"


@dataclass
class Reject(UnsafeAction):
    """
    Instructs the BastionLab server to reject the request.
    """

    def _serialize(self) -> Any:
        return "Reject"


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

    safe_zone: Rule
    unsafe_handling: UnsafeAction
    savable: bool

    def _serialize(self) -> Any:
        return {
            "safe_zone": self.safe_zone._serialize(),
            "unsafe_handling": self.unsafe_handling._serialize(),
            "savable": self.savable,
        }


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
