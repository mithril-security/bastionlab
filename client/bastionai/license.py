from typing import Tuple, Optional, List, Union, TypedDict, Literal
from dataclasses import dataclass, is_dataclass
import cbor2
from .keys import PublicKey


def cbor_default_encoder(encoder, value):
    if is_dataclass(value):
        return encoder.encode(value.__dict__)
    raise TypeError("unknown cbor obj", value)


@dataclass
class RuleDto:
    AtLeastNOf: Optional[Tuple[int, List["RuleDto"]]] = None
    WithCheckpoint: Optional[bytes] = None
    WithDataset: Optional[bytes] = None
    SignedWith: Optional[bytes] = None

    @property
    def __dict__(self) -> dict:
        if self.AtLeastNOf:
            return { 'AtLeastNOf': self.AtLeastNOf }
        if self.WithCheckpoint:
            return { 'WithCheckpoint': self.WithCheckpoint }
        if self.WithDataset:
            return { 'WithDataset': self.WithDataset }
        if self.SignedWith:
            return { 'SignedWith': self.SignedWith }

    def __str__(self) -> str:
        b = ""
        if self.AtLeastNOf is not None:
            rules = map(str, self.AtLeastNOf[1])
            b += f"AtLeastNOf({self.AtLeastNOf[0]}, [{', '.join(rules)}])"
        if self.SignedWith is not None:
            b += f"SignedWith({self.SignedWith.hex()})"
        if self.WithCheckpoint is not None:
            b += f"WithCheckpoint({self.WithCheckpoint.hex()})"
        if self.WithDataset is not None:
            b += f"WithDataset({self.WithDataset.hex()})"
        b += ""
        return b


@dataclass
class ResultStrategyCustom:
    Custom: "LicenseDto"


@dataclass
class LicenseDto:
    train: Optional[RuleDto] = None
    train_metric: Optional[RuleDto] = None
    test: Optional[RuleDto] = None
    test_metric: Optional[RuleDto] = None
    list: Optional[RuleDto] = None
    fetch: Optional[RuleDto] = None
    delete: Optional[RuleDto] = None
    result_strategy: Union[
        Literal[
            "Checkpoint",
            "Dataset",
            "And",
            "Or",
        ],
        ResultStrategyCustom,
    ] = "Or"

    def __str__(self) -> str:
        b = "License {\n"
        b += f"  train={self.train},\n"
        b += f"  train_metric={self.train_metric},\n"
        b += f"  test={self.test},\n"
        b += f"  test_metric={self.test_metric},\n"
        b += f"  list={self.list},\n"
        b += f"  fetch={self.fetch},\n"
        b += f"  delete={self.delete},\n"
        if isinstance(self.result_strategy, str):
            b += f"  result_strategy={self.result_strategy},\n"
        else:
            s = "\n  ".join(str(self.result_strategy.Custom).split("\n"))
            b += f"  result_strategy=Custom({s}),\n"
        b += "}"
        return b


HashLike = Union[str, bytes]
PublicKeyLike = Union[str, PublicKey]


def translate_pubkey(s: PublicKeyLike) -> bytes:
    if isinstance(s, str):
        return PublicKey.from_pem_content(s).bytes
    if isinstance(s, PublicKey):
        return s.bytes
    raise TypeError(f"Value of type {type(s)} is not supported as a publickey")

def translate_hash(s: HashLike) -> bytes:
    if isinstance(s, str):
        return bytes.fromhex(s)
    if isinstance(s, bytes):
        return s
    raise TypeError(f"Value of type {type(s)} is not supported as a hash")


class Rule(TypedDict):
    with_checkpoint: HashLike
    with_dataset: HashLike
    signed_with: PublicKeyLike
    either: List["Rule"]
    all: List["Rule"]


def translate_rule(rule: Rule) -> RuleDto:
    keys = list(
        map(lambda en: en[0], filter(lambda en: en[1] is not None, rule.items()))
    )
    if keys == ["signed_with"]:
        return RuleDto(SignedWith=translate_hash(rule["signed_with"]))
    elif keys == ["with_checkpoint"]:
        return RuleDto(WithCheckpoint=translate_hash(rule["with_checkpoint"]))
    elif keys == ["with_dataset"]:
        return RuleDto(WithDataset=translate_hash(rule["with_dataset"]))
    elif keys == ["either"]:
        rules = list(map(translate_rule, rule["either"]))
        return RuleDto(AtLeastNOf=[1, rules])
    elif keys == ["all"]:
        rules = list(map(translate_rule, rule["all"]))
        return RuleDto(
            AtLeastNOf=[
                len(rules),
            ]
        )
    else:
        raise TypeError(f"Invalid rule: {rule}")


def merge_rules(original: Optional[RuleDto], rule: RuleDto) -> RuleDto:
    if original is None:
        return rule

    original_is_or = original.AtLeastNOf is not None and original.AtLeastNOf[0] == 1
    rule_is_or = rule.AtLeastNOf is not None and rule.AtLeastNOf[0] == 1

    if original_is_or and rule_is_or:
        return RuleDto(AtLeastNOf=[1, [*original.AtLeastNOf[1], *rule.AtLeastNOf[1]]])
    if original_is_or:
        return RuleDto(AtLeastNOf=[1, [*original.AtLeastNOf[1], rule]])
    if rule_is_or:
        return RuleDto(AtLeastNOf=[1, [original, *rule.AtLeastNOf[1]]])

    return RuleDto(AtLeastNOf=[1, [original, rule]])


class LicenseBuilder:
    __obj: LicenseDto = LicenseDto()

    def __str__(self) -> str:
        return self.__obj.__str__()

    def ser(self) -> bytes:
        return cbor2.dumps(self.__obj, default=cbor_default_encoder)

    @staticmethod
    def default_with_pubkey(key: PublicKeyLike) -> "LicenseBuilder":
        builder = LicenseBuilder()
        k = translate_pubkey(key)
        builder.__obj = LicenseDto(
            train=RuleDto(SignedWith=k),
            train_metric=RuleDto(SignedWith=k),
            test=RuleDto(SignedWith=k),
            test_metric=RuleDto(SignedWith=k),
            list=RuleDto(SignedWith=k),
            fetch=RuleDto(SignedWith=k),
            delete=RuleDto(SignedWith=k),
            result_strategy="And",
        )
        return builder

    def validate(self):
        def rec(rule: Optional[RuleDto], path: str):
            if not rule:
                raise TypeError(f"Error in license builder: rule {path} is missing")

            n = (
                (rule.AtLeastNOf != None)
                + (rule.WithDataset != None)
                + (rule.WithCheckpoint != None)
                + (rule.SignedWith != None)
            )
            if n == 0:
                raise TypeError(
                    f"Error in license builder: rule {path} contains an empty rule"
                )
            elif n != 1:
                raise TypeError(
                    f"Error in license builder: rule {path} contains multiple directives"
                )

            if rule.AtLeastNOf is not None:
                n = rule.AtLeastNOf[0]
                subrules = rule.AtLeastNOf[1]
                if n == 0:
                    raise TypeError(
                        f"Error in license builder: rule {path} is always true"
                    )
                if n > len(subrules):
                    raise TypeError(
                        f"Error in license builder: rule {path} is always false"
                    )

                for i, rule in enumerate(subrules):
                    rec(subrules, f"{path}.AtLeastNOf({n})[{i}]")

        rec(self.__obj.train, "train")
        rec(self.__obj.train_metric, "train_metric")
        rec(self.__obj.test, "test")
        rec(self.__obj.test_metric, "test_metric")
        rec(self.__obj.list, "list")
        rec(self.__obj.fetch, "fetch")
        rec(self.__obj.delete, "delete")
        if self.__obj.result_strategy is None:
            raise TypeError(f"Error in license builder: no result strategy")

    def trainable(
        self,
        rule: Optional[Rule] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[Rule]] = None,
        all: Optional[List[Rule]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = Rule(
                with_checkpoint=with_checkpoint,
                with_dataset=with_dataset,
                signed_with=signed_with,
                either=either,
                all=all,
            )
        self.__obj.train = merge_rules(self.__obj.train, translate_rule(rule))
        return self

    def get_train_metrics(
        self,
        rule: Optional[Rule] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[Rule]] = None,
        all: Optional[List[Rule]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = Rule(
                with_checkpoint=with_checkpoint,
                with_dataset=with_dataset,
                signed_with=signed_with,
                either=either,
                all=all,
            )
        self.__obj.train_metric = merge_rules(
            self.__obj.train_metric, translate_rule(rule)
        )
        return self

    def fetchable(
        self,
        rule: Optional[Rule] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[Rule]] = None,
        all: Optional[List[Rule]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = Rule(
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
        rule: Optional[Rule] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[Rule]] = None,
        all: Optional[List[Rule]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = Rule(
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
        rule: Optional[Rule] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[Rule]] = None,
        all: Optional[List[Rule]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = Rule(
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
        rule: Optional[Rule] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[Rule]] = None,
        all: Optional[List[Rule]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = Rule(
                with_checkpoint=with_checkpoint,
                with_dataset=with_dataset,
                signed_with=signed_with,
                either=either,
                all=all,
            )
        self.__obj.test = merge_rules(self.__obj.test, translate_rule(rule))
        return self

    def get_test_metrics(
        self,
        rule: Optional[Rule] = None,
        *,
        with_checkpoint: Optional[HashLike] = None,
        with_dataset: Optional[HashLike] = None,
        signed_with: Optional[PublicKeyLike] = None,
        either: Optional[List[Rule]] = None,
        all: Optional[List[Rule]] = None,
    ) -> "LicenseBuilder":
        if rule is None:
            rule = Rule(
                with_checkpoint=with_checkpoint,
                with_dataset=with_dataset,
                signed_with=signed_with,
                either=either,
                all=all,
            )
        self.__obj.test_metric = merge_rules(
            self.__obj.test_metric, translate_rule(rule)
        )
        return self

    def created_checkpoints_license(
        self,
        *,
        get_from_checkpoint: bool = False,
        get_from_dataset: bool = False,
        compute_from_checkpoint_or_dataset: bool = False,
        compute_from_checkpoint_and_dataset: bool = False,
        use_license: Optional["LicenseBuilder"] = None,
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
            self.__obj.result_strategy = "Checkpoint"
        if get_from_dataset:
            self.__obj.result_strategy = "Dataset"
        if compute_from_checkpoint_or_dataset:
            self.__obj.result_strategy = "And"
        if compute_from_checkpoint_and_dataset:
            self.__obj.result_strategy = "Or"
        if use_license is not None:
            self.__obj.result_strategy = ResultStrategyCustom(Custom=use_license.__obj)
        return self
