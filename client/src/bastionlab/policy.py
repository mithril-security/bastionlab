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


@dataclass
class PolicyEntry:
    accept: Rule
    approval: Rule

    def serialize(self) -> str:
        return f'{{"accept":{self.accept.serialize()},"approval":{self.approval.serialize()}}}'


@dataclass
class Policy:
    fetch: PolicyEntry

    def serialize(self) -> str:
        return f'{{"fetch":{self.fetch.serialize()}}}'


DEFAULT_POLICY = Policy(PolicyEntry(Aggregation(10), TrueRule()))
