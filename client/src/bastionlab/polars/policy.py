from dataclasses import dataclass
from typing import List


class Rule:
    def serialize(self) -> str:
        raise NotImplementedError


@dataclass
class AtLeastNof(Rule):
    """
    specifies a collection of `Rule`s.

    Args
    ----
    n : int
        Specifies the number of rules to combine.
    of : List[Rule]
        Collection of rules.
    """

    n: int
    of: List[Rule]

    def serialize(self) -> str:
        of_repr = ",".join([x.serialize() for x in self.of])
        return f'{{"AtLeastNof":[{self.n},[{of_repr}]]}}'


@dataclass
class UserId(Rule):
    """
    BastionLab instruction `Rule` that attaches a user identifier to the safe zone.

    Args
    ----
    id : str
        User Identifier.
    """

    id: str

    def serialize(self) -> str:
        return f'{{"UserId":"{self.id}"}}'


@dataclass
class Aggregation(Rule):
    """
    Specifies a `Rule` for the number of rows an operation on a Remote DataFrame can aggregate.

    Args
    ----
    min_agg_size : int
        The minimum allowable row aggregation size.
    """

    min_agg_size: int

    def serialize(self) -> str:
        return f'{{"Aggregation":{self.min_agg_size}}}'


@dataclass
class TrueRule(Rule):
    """
    BastionLab instruction `Rule` for boolean logic `True`.
    """

    def serialize(self) -> str:
        return '"True"'


@dataclass
class FalseRule(Rule):
    """
    BastionLab instruction `Rule` for boolean logic `False`.
    """

    def serialize(self) -> str:
        return '"False"'


class UnsafeAction:
    def serialize(self) -> str:
        raise NotImplementedError


@dataclass
class Log(UnsafeAction):
    """
    Instructs the BastionLab server to _log_ the operation performed on the associated
    Remote DataFrame.
    """

    def serialize(self) -> str:
        return '"Log"'


@dataclass
class Review(UnsafeAction):
    """
    Instructs the BastionLab server to request a review from the owner of the Remote DataFrame.
    """

    def serialize(self) -> str:
        return '"Review"'


@dataclass
class Reject(UnsafeAction):
    """
    Instructs the BastionLab server to reject the request.
    """

    def serialize(self) -> str:
        return '"Reject"'


@dataclass
class Policy:
    """
    BastionLab Policy class.

    Express the allowable operations on Remote DataFrames [RDFs] (i.e., DataFrames on the BastionLab server).

    Args
    ----
    safe_zone : Rule
        Describes what operations are considered _safe_ on the RDF.
    unsafe_handling : UnsafeAction
        Describes what should happen if a user violates the `safe_zone`. For example (logging operations)

    Examples
    --------
    >>> from bastionlab.polars.policy import Policy, Aggregation, Log
    >>> policy = Policy(safe_zone=Aggregation(10), unsafe_handling=Log())
    """

    safe_zone: Rule
    unsafe_handling: UnsafeAction

    def serialize(self) -> str:
        return f'{{"safe_zone":{self.safe_zone.serialize()},"unsafe_handling":{self.unsafe_handling.serialize()}}}'


DEFAULT_POLICY = Policy(Aggregation(10), Review())
"""
Default BastionLab Client Policy `Policy(Aggregation(10), Review())`
"""
