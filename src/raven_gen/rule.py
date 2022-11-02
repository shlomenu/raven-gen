from typing import Iterable, List, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass, field
import copy

import numpy as np

from .attribute import AttributeType, COLOR_MIN, COLOR_MAX
from .component import Component, ComponentType, LayoutType


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
    value: Optional[int] = field(default=None, init=False)

    def __post_init__(self):
        if self.params is not None:
            self.value = np.random.choice(self.params)

    def increment(self):
        pass


def apply_rule(rule: Rule,
               prev_comp: Component,
               next_comp: Optional[Component] = None) -> Component:
    if next_comp is None:
        next_comp = prev_comp
    if rule.attr is AttributeType.ANGLE or \
            rule.attr is AttributeType.UNIFORMITY or \
            rule.attr not in AttributeType:
        raise ValueError("unsupported attribute")
    elif rule.name is RuleType.CONSTANT:
        next_comp = copy.deepcopy(next_comp)
        if rule.attr is AttributeType.CONFIGURATION:
            next_comp.sample(carryover=False)
        else:
            if rule.prev_is_col_0 and not prev_comp.uniformity.value:
                prev_comp.make_uniform(rule.attr)
            next_comp.set_uniform(rule.attr, prev_comp.setting_of(rule.attr))
    elif rule.name is RuleType.PROGRESSION:
        next_comp = copy.deepcopy(next_comp)
        if rule.attr is AttributeType.NUMBER:
            next_comp.config.number.setting += rule.value
            next_comp.sample(sample_position=True, carryover=False)
        elif rule.attr is AttributeType.POSITION:
            next_comp.config.position.setting = (
                next_comp.config.position.setting +
                rule.value) % next_comp.config.position.values.shape[0]
            next_comp.sample(carryover=False)
        else:
            if rule.prev_is_col_0 and not prev_comp.uniformity.value:
                prev_comp.make_uniform(rule.attr)
            next_comp.set_uniform(rule.attr,
                                  prev_comp.setting_of(rule.attr) + rule.value)
    elif rule.name is RuleType.ARITHMETIC:
        if rule.attr is AttributeType.SHAPE:
            raise ValueError("unsupported attribute")
        next_comp = copy.deepcopy(next_comp)
        if rule.attr is AttributeType.NUMBER:
            if rule.col_0_setting is None:  # second column
                rule.col_0_setting = prev_comp.config.number.setting
                rule.set_constraints_col_1(prev_comp, next_comp)
                next_comp.config.number.sample(next_comp.constraints)
            else:  # third column
                next_comp.config.number.setting = rule.col_2_setting(
                    prev_comp.config.number.setting)
            next_comp.sample(sample_position=True, carryover=False)
        elif rule.attr is AttributeType.POSITION:
            if rule.col_0_setting is None:  # second column
                rule.col_0_setting = prev_comp.config.position.setting
                rule.set_constraints_col_1(prev_comp, next_comp)
            else:  # third column
                col_2_setting = rule.col_2_setting(
                    prev_comp.config.position.setting)
                next_comp.config.number.setting = len(col_2_setting) - 1
                next_comp.config.position.setting = np.array(col_2_setting)
            next_comp.sample(carryover=False)
        elif rule.attr is AttributeType.SIZE:
            if rule.col_0_setting is None:  # second column
                rule.col_0_setting = prev_comp.setting_of(rule.attr)
                if not prev_comp.uniformity.value:
                    prev_comp.make_uniform(rule.attr)
                rule.set_constraints_col_1(prev_comp, next_comp)
                next_comp.attr(rule.attr).sample(next_comp.constraints)
                next_comp.make_uniform(rule.attr)
            else:  # third column
                next_comp.set_uniform(
                    rule.attr,
                    rule.col_2_setting(prev_comp.setting_of(rule.attr)))
        elif rule.attr is AttributeType.COLOR:
            rule.color_count += 1
            if rule.col_0_setting is None:
                prev_setting = prev_comp.setting_of(rule.attr)
                reset_previous = False
                if rule.color_count == 3 and rule.color_white_alarm and \
                    ((rule.value > 0 and prev_setting == COLOR_MAX) or
                        (rule.value < 0 and prev_setting == COLOR_MIN)):
                    prev_comp.attr(rule.attr).sample_unique(
                        prev_comp.constraints, prev_comp.history, inplace=True)
                    prev_setting = prev_comp.setting_of(rule.attr)
                    reset_previous = True
                rule.col_0_setting = prev_setting
                if reset_previous or not prev_comp.uniformity.value:
                    prev_comp.make_uniform(rule.attr)
                rule.set_constraints_col_1(prev_comp, next_comp)
                next_comp.attr(rule.attr).sample(next_comp.constraints)
                next_setting = next_comp.setting_of(rule.attr)
                if rule.color_count == 1:
                    rule.color_white_alarm = (next_setting == 0)
                elif rule.color_count == 3 and rule.color_white_alarm and \
                     next_setting == 0:
                    next_comp.attr(rule.attr).sample_unique(
                        next_comp.constraints, next_comp.history, inplace=True)
                next_comp.make_uniform(rule.attr)
            else:  # third column
                next_comp.set_uniform(
                    rule.attr,
                    rule.col_2_setting(prev_comp.setting_of(rule.attr)))
    elif rule.name is RuleType.DISTRIBUTE_THREE:
        if rule.attr is AttributeType.NUMBER:
            if rule.count == 0:  # first row
                rule.create_settings(
                    np.insert(
                        np.random.choice(
                            list(
                                range(prev_comp.constraints.number.min,
                                      prev_comp.config.number.setting)) +
                            list(
                                range(prev_comp.config.number.setting + 1,
                                      prev_comp.constraints.number.max + 1)),
                            size=2,
                            replace=False), 0,
                        prev_comp.config.number.setting))
                next_comp = copy.deepcopy(next_comp)
                next_comp.config.number.setting = rule.settings[0][1]
            else:
                row, col = divmod(rule.count, 2)
                if col == 0:
                    prev_comp.config.number.setting = rule.settings[row][0]
                    prev_comp.sample(sample_position=True)
                    next_comp = copy.deepcopy(prev_comp)
                    next_comp.config.number.setting = rule.settings[row][1]
                else:
                    next_comp = copy.deepcopy(next_comp)
                    next_comp.config.number.setting = rule.settings[row][2]
            next_comp.sample(sample_position=True)
        elif rule.attr is AttributeType.POSITION:
            if rule.count == 0:
                rule.create_settings(
                    np.array([
                        prev_comp.config.position.setting,
                        prev_comp.config.position.sample_unique(
                            prev_comp.config.number.value, prev_comp.history),
                        prev_comp.config.position.sample_unique(
                            prev_comp.config.number.value, prev_comp.history)
                    ]))
                next_comp = copy.deepcopy(next_comp)
                next_comp.config.position.setting = rule.settings[0][1]
            else:
                row, col = divmod(rule.count, 2)
                if col == 0:
                    prev_comp.config.number.setting = rule.settings[row][
                        0].shape[0] - 1
                    prev_comp.config.position.setting = rule.settings[row][0]
                    prev_comp.sample()
                    next_comp = copy.deepcopy(prev_comp)
                    next_comp.config.position.setting = rule.settings[row][1]
                else:
                    next_comp = copy.deepcopy(next_comp)
                    next_comp.config.position.setting = rule.settings[row][2]
            next_comp.sample()
        else:

            def attr_of(x):
                return getattr(x, rule.attr.name.lower())

            next_comp = copy.deepcopy(next_comp)
            if rule.count == 0:
                rule.create_settings(
                    np.random.choice(a=range(
                        attr_of(prev_comp.constraints).min,
                        attr_of(prev_comp.constraints).max + 1),
                                     size=3,
                                     replace=False))
                prev_comp.set_uniform(rule.attr, rule.settings[0][0])
                next_comp.set_uniform(rule.attr, rule.settings[0][1])
            else:
                row, col = divmod(rule.count, 2)
                if col == 0:
                    prev_comp.set_uniform(rule.attr, rule.settings[row][0])
                    next_comp.set_uniform(rule.attr, rule.settings[row][1])
                else:
                    next_comp.set_uniform(rule.attr, rule.settings[row][2])
    rule.increment()
    return next_comp


@dataclass
class Constant(Rule):

    def __post_init__(self):
        super(Constant, self).__post_init__()
        self.prev_is_col_0 = True

    def increment(self):
        self.prev_is_col_0 = not self.prev_is_col_0


@dataclass
class Progression(Rule):

    def __post_init__(self):
        super(Progression, self).__post_init__()
        self.prev_is_col_0 = True

    def increment(self):
        self.prev_is_col_0 = not self.prev_is_col_0


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

            def attr_of(x):
                return getattr(x, self.attr.name.lower())

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
                return list(set(col_0_setting) | set(col_1_setting))
            else:
                return list(set(col_0_setting) - set(col_1_setting))
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
    configuration: Rule
    shape: Rule
    size: Rule
    color: Rule
    component_type: ComponentType
    layout_type: LayoutType

    def __post_init__(self):
        self.all = (self.configuration, self.shape, self.size, self.color)
        self.secondary = self.all[1:]

    def __iter__(self):
        for rule in self.secondary:
            yield rule


@dataclass(frozen=True)
class Rules:
    components_rules: Tuple[ComponentRules]

    def __iter__(self):
        for component_rules in self.components_rules:
            yield component_rules

    def __getitem__(self, i):
        return self.components_rules[i]

    def __len__(self):
        return len(self.components_rules)

    def __str__(self):
        s = "\n"
        for c, comp_rules in enumerate(self):
            for rule in comp_rules.all:
                s += f"{rule!r}\n"
            if c == 0 and len(self) > 1:
                s += f"\n\t----------- \\\\ {self[0].component_type.name} > {self[0].layout_type.name} // --- " + \
                     f"// {self[1].component_type.name} > {self[1].layout_type.name} \\\\ -----------\n\n"
        return s


@dataclass
class Ruleset:

    position_rules: Iterable[RuleType] = None
    number_rules: Iterable[RuleType] = None
    shape_rules: Iterable[RuleType] = None
    size_rules: Iterable[RuleType] = None
    color_rules: Iterable[RuleType] = None

    def __post_init__(self):
        for attr_rules in ("position_rules", "number_rules", "shape_rules",
                           "size_rules", "color_rules"):
            rules = getattr(self, attr_rules)
            if rules is None:
                validated_rules = []
            else:
                validated_rules = tuple(rule for rule in rules
                                        if rule in RuleType)
            if len(validated_rules) > 0:
                setattr(self, attr_rules, validated_rules)
            else:
                exclusions = {RuleType.ARITHMETIC
                              } if attr_rules == "shape_rules" else set()
                setattr(self, attr_rules, tuple(set(RuleType) - exclusions))

    @staticmethod
    def rule(name, attr):
        if name is RuleType.CONSTANT:
            return Constant(name, attr, params=None)
        elif name is RuleType.PROGRESSION:
            return Progression(name, attr, params=[-2, -1, 1, 2])
        elif name is RuleType.ARITHMETIC:
            return Arithmetic(name, attr, params=[-1, 1])
        else:  # name is RuleType.DISTRIBUTE_THREE
            return DistributeThree(name, attr, params=None)

    def sample(self, component_and_layout_types):
        assert (len(component_and_layout_types) == 1
                or len(component_and_layout_types) == 2)
        components_rules = []
        for component_type, layout_type in component_and_layout_types:
            name, attr = (np.random.choice(self.position_rules),
                          AttributeType.POSITION) if np.random.choice(
                              (True, False)) else (np.random.choice(
                                  self.number_rules), AttributeType.NUMBER)
            components_rules.append(
                ComponentRules(
                    self.rule(
                        name, AttributeType.CONFIGURATION
                        if name is RuleType.CONSTANT else attr),
                    self.rule(np.random.choice(self.shape_rules),
                              AttributeType.SHAPE),
                    self.rule(np.random.choice(self.size_rules),
                              AttributeType.SIZE),
                    self.rule(np.random.choice(self.color_rules),
                              AttributeType.COLOR), component_type,
                    layout_type))
        return Rules(tuple(components_rules))
