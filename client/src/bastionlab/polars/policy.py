from dataclasses import dataclass
from typing import List


class Rule:
    def serialize(self) -> str:
        raise NotImplementedError


@dataclass
class AtLeastNof(Rule):
    n: int
    of: List[Rule]

    def serialize(self) -> str:
        of_repr = ",".join([x.serialize() for x in self.of])
        return f'{{"AtLeastNof":[{self.n},[{of_repr}]]}}'


@dataclass
class UserId(Rule):
    id: str

    def serialize(self) -> str:
        return f'{{"UserId":{self.id}}}'


@dataclass
class Aggregation(Rule):
    min_agg_size: int

    def serialize(self) -> str:
        return f'{{"Aggregation":{self.min_agg_size}}}'


@dataclass
class TrueRule(Rule):
    def serialize(self) -> str:
        return '"True"'


@dataclass
class FalseRule(Rule):
    def serialize(self) -> str:
        return '"False"'


class UnsafeAction:
    def serialize(self) -> str:
        raise NotImplementedError


@dataclass
class Log(UnsafeAction):
    def serialize(self) -> str:
        return '"Log"'


@dataclass
class Review(UnsafeAction):
    def serialize(self) -> str:
        return '"Review"'


@dataclass
class Reject(UnsafeAction):
    def serialize(self) -> str:
        return '"Reject"'


@dataclass
class Policy:
    safe_zone: Rule
    unsafe_handling: UnsafeAction

    def serialize(self) -> str:
        return f'{{"safe_zone":{self.safe_zone.serialize()},"unsafe_handling":{self.unsafe_handling.serialize()}}}'


DEFAULT_POLICY = Policy(Aggregation(10), Review())
