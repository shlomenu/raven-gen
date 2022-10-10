from typing import List
from enum import Enum, auto
from dataclasses import dataclass, field
import numpy as np

from configuration import AttributeType, RuleType


class RuleType(Enum):
    CONSTANT = auto()
    PROGRESSION = auto()
    ARITHMETIC = auto()
    DISTRIBUTE_THREE = auto()


@dataclass
class Rule:
    name: RuleType
    attr: AttributeType
    params: List[int]
    component_idx = field(default=0)

    def __post_init__(self):
        if self.params is not None:
            self.value = np.random.choice(self.params)

    def increment(self):
        pass


@dataclass
class Constant(Rule):
    pass


@dataclass
class Progression(Rule):

    def __post_init__(self):
        super(Progression, self).__post_init__()
        self.previous_is_col_0 = True

    def increment(self):
        self.previous_is_col_0 = not self.previous_is_col_0


@dataclass
class Arithmetic(Rule):

    def __post_init__(self):
        super(Arithmetic, self).__post_init__()
        self.col_0_setting = None
        self.color_count = 0
        self.color_white_alarm = False

    def pop(self):
        col_0_setting = self.col_0_setting
        self.col_0_setting = None
        return col_0_setting

    def set_constraints_col_1(self, prev_comp, next_comp):
        if self.attr is AttributeType.POSITION:
            while True:
                next_comp.config.sample(next_comp.constraints)
                col_1_setting = next_comp.config.position.setting
                if self.value > 0:
                    if not (set(self.col_0_setting) >= set(col_1_setting)):
                        return
                else:
                    if not (set(self.col_0_setting) <= set(col_1_setting)):
                        return
        else:
            def attr_of(x): return getattr(x, self.attr.name.lower())
            next_constraint = attr_of(next_comp.constraints)
            prev_constraint = attr_of(prev_comp.constraints)
            if self.value > 0:
                next_constraint.max = prev_constraint.max - (
                    self.col_0_setting - prev_constraint.min)
            else:
                offset = 0 if self.attr is AttributeType.COLOR else 1
                min_prepruning = (next_constraint.min - offset) // 2
                next_constraint.min = min_prepruning
                if self.attr is AttributeType.COLOR:
                    next_constraint.max = self.col_0_setting
                else:
                    next_constraint.max = self.col_0_setting - min_prepruning - 1

    def col_2_setting(self, col_1_setting):
        col_0_setting = self.pop()
        if self.attr is AttributeType.POSITION:
            if self.value > 0:
                return list(
                    set(col_0_setting) | set(col_1_setting))
            else:
                return list(
                    set(col_0_setting) - set(col_1_setting))
        else:
            offset = 0 if self.attr is AttributeType.COLOR else 1
            if self.value > 0:
                return col_0_setting + (col_1_setting + offset)
            else:
                return col_0_setting - (col_1_setting + offset)


@dataclass
class DistributeThree(Rule):

    def __post_init__(self):
        super(DistributeThree, self).__post_init__()
        self.settings = []
        self.count = 0

    def create_settings(self, three_settings):
        self.settings.append(three_settings[[0, 1, 2]])
        if np.random.uniform() >= 0.5:
            self.settings.append(three_settings[[1, 2, 0]])
            self.settings.append(three_settings[[2, 0, 1]])
        else:
            self.settings.append(three_settings[[2, 0, 1]])
            self.settings.append(three_settings[[1, 2, 0]])

    def increment(self):
        self.count = (self.count + 1) % 6


@dataclass
class ComponentRules:
    all: List[Rule]

    def __post_init__(self):
        assert (len(self.all) == 4)
        self.number_or_position, self.type, self.size, self.color = \
            self.all
        self.secondary = [self.type, self.size, self.color]

    def __iter__(self):
        for rule in self.secondary:
            yield rule


@dataclass
class Rules:
    components_rules: List[ComponentRules]

    def __post_init__(self):
        assert (len(self.components_rules) ==
                1 or len(self.components_rules) == 2)

    @property
    def rulesets(self):
        if hasattr(self, "_rulesets"):
            return self._rulesets
        else:
            self._rulesets = [
                [[RuleType.PROGRESSION, AttributeType.NUMBER],
                 [RuleType.PROGRESSION, AttributeType.POSITION],
                 [RuleType.ARITHMETIC, AttributeType.NUMBER],
                 [RuleType.ARITHMETIC, AttributeType.POSITION],
                 [RuleType.DISTRIBUTE_THREE, AttributeType.NUMBER],
                 [RuleType.DISTRIBUTE_THREE, AttributeType.POSITION],
                 [RuleType.CONSTANT, AttributeType.CONFIGURATION]],
                [[RuleType.PROGRESSION, AttributeType.TYPE],
                 [RuleType.DISTRIBUTE_THREE, AttributeType.TYPE],
                 [RuleType.CONSTANT, AttributeType.TYPE]],
                [[RuleType.PROGRESSION, AttributeType.SIZE],
                 [RuleType.ARITHMETIC, AttributeType.SIZE],
                 [RuleType.DISTRIBUTE_THREE, AttributeType.SIZE],
                 [RuleType.CONSTANT, AttributeType.SIZE]],
                [[RuleType.PROGRESSION, AttributeType.COLOR],
                 [RuleType.ARITHMETIC, AttributeType.COLOR],
                 [RuleType.DISTRIBUTE_THREE, AttributeType.COLOR],
                 [RuleType.CONSTANT, AttributeType.COLOR]]]

    @staticmethod
    def make_rule(ruleset, c):
        name, attr = ruleset[np.random.choice(len(ruleset))]
        if name is RuleType.CONSTANT:
            return Constant(name, attr, params=None, component_idx=c)
        elif name is RuleType.PROGRESSION:
            return Progression(name, attr, params=[-2, -1, 1, 2], component_idx=c)
        elif name is RuleType.ARITHMETIC:
            return Arithmetic(name, attr, params=[1, -1], component_idx=c)
        elif name is RuleType.DISTRIBUTE_THREE:
            return DistributeThree(name, attr, params=None, component_idx=c)
        else:
            raise ValueError("unknown rule type")

    def make_random(self):
        return Rules([
            ComponentRules([
                Rules.make_rule(ruleset, c) for ruleset in self.rulesets
            ])
            for c in range(np.random.randint(1, 3))
        ])

    def __iter__(self):
        for component_rules in self.components_rules:
            yield component_rules

    def __getitem__(self, i):
        return self.components_rules[i]

    def __len__(self):
        return len(self.components_rules)
