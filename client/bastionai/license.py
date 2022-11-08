from typing import Tuple, Optional, List, Union, TypedDict, Literal
from dataclasses import dataclass, is_dataclass
import cbor2
from .keys import PublicKey
import bastionai.pb.remote_torch_pb2 as pb  # type: ignore [import]
from enum import Enum


def cbor_default_encoder(encoder, value):
    if is_dataclass(value):
        return encoder.encode(value.__dict__)
    raise TypeError("unknown cbor obj", value)


@dataclass
class Rule:
    at_least_n_of: Optional[Tuple[int, List["Rule"]]] = None
    with_checkpoint: Optional[bytes] = None
    with_dataset: Optional[bytes] = None
    signed_with: Optional[PublicKey] = None

    def ser(self) -> pb.Rule:
        if self.at_least_n_of is not None:
            return pb.Rule(
                at_least_n_of=pb.Rule.AtLeastNOf(
                    n=self.at_least_n_of[0],
                    rules=[el.ser() for el in self.at_least_n_of[1]],
                )
            )
        if self.with_checkpoint is not None:
            return pb.Rule(with_checkpoint=self.with_checkpoint)
        if self.with_dataset is not None:
            return pb.Rule(with_dataset=self.with_dataset)
        if self.signed_with is not None:
            return pb.Rule(signed_with=self.signed_with.bytes)

    @staticmethod
    def deser(rule: pb.Rule) -> "Rule":
        if rule.HasField("at_least_n_of"):
            return Rule(
                at_least_n_of=(
                    rule.at_least_n_of.n,
                    [Rule.deser(r) for r in rule.at_least_n_of.rules],
                )
            )
        if rule.HasField("with_checkpoint"):
            return Rule(with_checkpoint=rule.with_checkpoint)
        if rule.HasField("with_dataset"):
            return Rule(with_dataset=rule.with_dataset)
        if rule.HasField("signed_with"):
            return Rule(signed_with=PublicKey.from_bytes_content(rule.signed_with))

    def __str__(self) -> str:
        b = "Rule("
        if self.at_least_n_of is not None:
            rules = map(str, self.at_least_n_of[1])
            b += f"at_least_n_of({self.at_least_n_of[0]}, [{', '.join(rules)}])"
        elif self.signed_with is not None:
            b += f"signed_with(hash={self.signed_with.hash.hex()})"
        elif self.with_checkpoint is not None:
            b += f"with_checkpoint({self.with_checkpoint.hex()})"
        elif self.with_dataset is not None:
            b += f"with_dataset({self.with_dataset.hex()})"
        b += ")"
        return b


@dataclass
class ResultStrategyKind(Enum):
    Checkpoint = 0
    Dataset = 1
    And = 2
    Or = 3
    Custom = 4


@dataclass
class ResultStrategy:
    strategy: ResultStrategyKind
    custom_license: Optional["License"] = None

    def ser(self) -> pb.ResultStrategy:
        return pb.ResultStrategy(
            strategy=self.strategy.value,
            custom_license=self.custom_license.ser() if self.custom_license else None,
        )

    @staticmethod
    def deser(rs: pb.ResultStrategy) -> "ResultStrategy":
        return ResultStrategy(
            strategy=ResultStrategyKind(rs.strategy), custom_license=None if not rs.HasField("custom_license") else License.deser(rs.custom_license)
        )

    def __str__(self) -> str:
        b = f"ResultStrategy(strategy={self.strategy}"
        if self.custom_license:
            b += f", custom_license={self.custom_license}"
        b += ")"
        return b


@dataclass
class License:
    train: Optional[Rule] = None
    test: Optional[Rule] = None
    list: Optional[Rule] = None
    fetch: Optional[Rule] = None
    delete: Optional[Rule] = None
    result_strategy: Optional[ResultStrategy] = None

    def ser(self) -> pb.License:
        return pb.License(
            train=self.train.ser(),
            test=self.test.ser(),
            list=self.list.ser(),
            fetch=self.fetch.ser(),
            delete=self.delete.ser(),
            result_strategy=self.result_strategy.ser(),
        )

    @staticmethod
    def deser(l: pb.License) -> "License":
        return License(
            train=Rule.deser(l.train),
            test=Rule.deser(l.test),
            list=Rule.deser(l.list),
            fetch=Rule.deser(l.fetch),
            delete=Rule.deser(l.delete),
            result_strategy=ResultStrategy.deser(l.result_strategy),
        )

    def __str__(self) -> str:
        b = "License { "
        b += f"train={self.train}, "
        b += f"test={self.test}, "
        b += f"list={self.list}, "
        b += f"fetch={self.fetch}, "
        b += f"delete={self.delete}, "
        b += f"result_strategy={self.result_strategy}"
        b += " }"
        return b


HashLike = Union[str, bytes]
PublicKeyLike = Union[str, PublicKey]


def translate_pubkey(s: PublicKeyLike) -> bytes:
    if isinstance(s, str):
        return PublicKey.from_pem_content(s)
    if isinstance(s, PublicKey):
        return s
    raise TypeError(f"Value of type {type(s)} is not supported as a publickey")


def translate_hash(s: HashLike) -> bytes:
    if isinstance(s, str):
        return bytes.fromhex(s)
    if isinstance(s, bytes):
        return s
    raise TypeError(f"Value of type {type(s)} is not supported as a hash")


class RuleBuilder(TypedDict):
    with_checkpoint: HashLike
    with_dataset: HashLike
    signed_with: PublicKeyLike
    either: List["RuleBuilder"]
    all: List["RuleBuilder"]


def translate_rule(rule: RuleBuilder) -> Rule:
    keys = list(
        map(lambda en: en[0], filter(lambda en: en[1] is not None, rule.items()))
    )
    if keys == ["signed_with"]:
        return Rule(signed_with=translate_pubkey(rule["signed_with"]))
    elif keys == ["with_checkpoint"]:
        return Rule(with_checkpoint=translate_hash(rule["with_checkpoint"]))
    elif keys == ["with_dataset"]:
        return Rule(with_dataset=translate_hash(rule["with_dataset"]))
    elif keys == ["either"]:
        rules = list(map(translate_rule, rule["either"]))
        return Rule(at_least_n_of=[1, rules])
    elif keys == ["all"]:
        rules = list(map(translate_rule, rule["all"]))
        return Rule(
            at_least_n_of=[
                len(rules),
            ]
        )
    else:
        raise TypeError(f"Invalid rule: {rule}")


def merge_rules(original: Optional[Rule], rule: Rule) -> Rule:
    if original is None:
        return rule

    original_is_or = (
        original.at_least_n_of is not None and original.at_least_n_of[0] == 1
    )
    rule_is_or = rule.at_least_n_of is not None and rule.at_least_n_of[0] == 1

    if original_is_or and rule_is_or:
        return Rule(
            at_least_n_of=[1, [*original.at_least_n_of[1], *rule.at_least_n_of[1]]]
        )
    if original_is_or:
        return Rule(at_least_n_of=[1, [*original.at_least_n_of[1], rule]])
    if rule_is_or:
        return Rule(at_least_n_of=[1, [original, *rule.at_least_n_of[1]]])

    return Rule(at_least_n_of=[1, [original, rule]])


class LicenseBuilder:
    __obj: License

    def build(self) -> License:
        self.validate()
        return self.__obj

    def __init__(self, obj: License = License()):
        self.__obj = obj

    def __str__(self) -> str:
        return self.__obj.__str__()

    @staticmethod
    def default_with_pubkey(key: PublicKeyLike) -> "LicenseBuilder":
        builder = LicenseBuilder()
        k = translate_pubkey(key)
        builder.__obj = License(
            train=Rule(signed_with=k),
            test=Rule(signed_with=k),
            list=Rule(signed_with=k),
            fetch=Rule(signed_with=k),
            delete=Rule(signed_with=k),
            result_strategy=ResultStrategy(strategy=ResultStrategyKind.And),
        )
        return builder

    def validate(self):
        def rec(rule: Optional[Rule], path: str):
            if not rule:
                raise TypeError(f"Error in license builder: rule {path} is missing")

            n = (
                (rule.at_least_n_of != None)
                + (rule.with_dataset != None)
                + (rule.with_checkpoint != None)
                + (rule.signed_with != None)
            )
            if n == 0:
                raise TypeError(
                    f"Error in license builder: rule {path} contains an empty rule"
                )
            elif n != 1:
                raise TypeError(
                    f"Error in license builder: rule {path} contains multiple directives"
                )

            if rule.at_least_n_of is not None:
                n = rule.at_least_n_of[0]
                subrules = rule.at_least_n_of[1]
                if n == 0:
                    raise TypeError(
                        f"Error in license builder: rule {path} is always true"
                    )
                if n > len(subrules):
                    raise TypeError(
                        f"Error in license builder: rule {path} is always false"
                    )

                for i, rule in enumerate(subrules):
                    rec(rule, f"{path}.at_least_n_of({n})[{i}]")

        rec(self.__obj.train, "train")
        rec(self.__obj.test, "test")
        rec(self.__obj.list, "list")
        rec(self.__obj.fetch, "fetch")
        rec(self.__obj.delete, "delete")
        if self.__obj.result_strategy is None:
            raise TypeError(f"Error in license builder: no result strategy")

    def trainable(
        self,
        rule: Optional[RuleBuilder] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[RuleBuilder]] = None,
        all: Optional[List[RuleBuilder]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = RuleBuilder(
                with_checkpoint=with_checkpoint,
                with_dataset=with_dataset,
                signed_with=signed_with,
                either=either,
                all=all,
            )
        self.__obj.train = merge_rules(self.__obj.train, translate_rule(rule))
        return self

    def fetchable(
        self,
        rule: Optional[RuleBuilder] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[RuleBuilder]] = None,
        all: Optional[List[RuleBuilder]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = RuleBuilder(
                with_checkpoint=with_checkpoint,
                with_dataset=with_dataset,
                signed_with=signed_with,
                either=either,
                all=all,
            )
        self.__obj.fetch = merge_rules(self.__obj.fetch, translate_rule(rule))
        return self

    def deletable(
        self,
        rule: Optional[RuleBuilder] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[RuleBuilder]] = None,
        all: Optional[List[RuleBuilder]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = RuleBuilder(
                with_checkpoint=with_checkpoint,
                with_dataset=with_dataset,
                signed_with=signed_with,
                either=either,
                all=all,
            )
        self.__obj.delete = merge_rules(self.__obj.delete, translate_rule(rule))
        return self

    def listable(
        self,
        rule: Optional[RuleBuilder] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[RuleBuilder]] = None,
        all: Optional[List[RuleBuilder]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = RuleBuilder(
                with_checkpoint=with_checkpoint,
                with_dataset=with_dataset,
                signed_with=signed_with,
                either=either,
                all=all,
            )
        self.__obj.list = merge_rules(self.__obj.list, translate_rule(rule))
        return self

    def testable(
        self,
        rule: Optional[RuleBuilder] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[RuleBuilder]] = None,
        all: Optional[List[RuleBuilder]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = RuleBuilder(
                with_checkpoint=with_checkpoint,
                with_dataset=with_dataset,
                signed_with=signed_with,
                either=either,
                all=all,
            )
        self.__obj.test = merge_rules(self.__obj.test, translate_rule(rule))
        return self

    def created_checkpoints_license(
        self,
        *,
        get_from_checkpoint: bool = False,
        get_from_dataset: bool = False,
        compute_from_checkpoint_or_dataset: bool = False,
        compute_from_checkpoint_and_dataset: bool = False,
        use_license: Optional[Union["LicenseBuilder", License]] = None,
    ):
        n = (
            get_from_checkpoint
            + get_from_dataset
            + compute_from_checkpoint_and_dataset
            + compute_from_checkpoint_or_dataset
            + (use_license is not None)
        )
        if n == 0:
            raise TypeError("You must supply a rule")
        if n != 1:
            raise TypeError("You must supply only one rule")

        if get_from_checkpoint:
            self.__obj.result_strategy = ResultStrategy(
                strategy=ResultStrategyKind.Checkpoint
            )
        if get_from_dataset:
            self.__obj.result_strategy = ResultStrategy(
                strategy=ResultStrategyKind.Dataset
            )
        if compute_from_checkpoint_or_dataset:
            self.__obj.result_strategy = ResultStrategy(strategy=ResultStrategyKind.Or)
        if compute_from_checkpoint_and_dataset:
            self.__obj.result_strategy = ResultStrategy(strategy=ResultStrategyKind.And)
        if use_license is not None:
            self.__obj.result_strategy = ResultStrategy(
                strategy=ResultStrategyKind.Custom,
                custom_license=use_license.build()
                if isinstance(use_license, LicenseBuilder)
                else use_license,
            )
        return self
