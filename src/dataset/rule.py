"""
There is still some strangeness in this module.  Many of the rules 
seem to be implementing in their `apply` methods weird and not 
transparent interactions with the changes to constraints made by 
`constraints.rule_constraints` via `panel.prune`.  
"""

from typing import List, Optional
import copy

import numpy as np

from panel import Panel
from configuration import COLOR_MAX, COLOR_MIN, AttributeType, RuleType


def rule_type_to_factory(rule_type):
    if rule_type is RuleType.CONSTANT:
        return Constant
    elif rule_type is RuleType.PROGRESSION:
        return Progression
    elif rule_type is RuleType.ARITHMETIC:
        return Arithmetic
    elif rule_type is RuleType.DISTRIBUTE_THREE:
        return Distribute_Three
    else:
        raise ValueError("unknown rule type")


class Rule:

    def __init__(self,
                 name: RuleType,
                 attr: AttributeType,
                 params: list,
                 component_idx: int = 0):
        """
        Instantiate a rule by its name, attribute, parameter list and the 
        component it applies to. Each rule should be applied to all entities 
        in a component.
        """
        self.name = name
        self.attr = attr
        self.params = params
        self.component_idx = component_idx
        self.value = 0
        self.sample()

    def sample(self):
        if self.params is not None:
            self.value = np.random.choice(self.params)

    def apply(self, previous_panel: Panel, next_panel: Panel = None):
        """
        Apply the rule to a component in the AoT.

        It is assumed that this method will be called first by 
        the rule on NUMBER/POSITION attributes, and then by 
        rules on TYPE, SIZE, or COLOR.  In the first case, `next_panel` 
        is not provided and all other non-uniform attributes are 
        resampled to provide variation which may or may not be 
        overwritten in the application of other rules.  In the 
        latter case, `next_panel` is provided and differences
        between it and `previous_panel` are preserved.  

        This method is called six times in the lifetime of a `Rule` 
        instance.  Specializations of this class may contain state 
        that affects the result of `apply` calls across rows (pairs 
        of calls) or across columns (between any of the six calls).
        """
        pass


class ComponentRules:

    def __init__(self, component_rules: List[Rule]):
        assert (len(component_rules) == 4)
        self.number_position, self.type, self.size, self.color = \
            component_rules
        self.all = component_rules

    def __iter__(self):
        for rule in [self.type, self.size, self.color]:
            yield rule


class Rules:

    def __init__(self, rules: List[ComponentRules]):
        assert (len(rules) == 1 or len(rules) == 2)
        self.rules = rules

    def __iter__(self):
        for rule in self.rules:
            yield rule

    def __getitem__(self, i):
        return self.rules[i]

    def __len__(self):
        len(self.rules)


class Constant(Rule):
    """
    Unary operator. Nothing changes.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Constant, self).__init__(name, attr, param, component_idx)

    def apply(self, previous_panel: Panel, next_panel: Optional[Panel] = None):
        if next_panel is None:
            next_panel = previous_panel
        return copy.deepcopy(next_panel)


class Progression(Rule):
    """
    Unary operator. Attribute differences are the same across panels
    of a row.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Progression, self).__init__(name, attr, param, component_idx)
        self.previous_is_col_0 = True

    def apply(self, previous_panel: Panel, next_panel: Optional[Panel] = None):
        previous_layout = previous_panel.structure.components[
            self.component_idx].layout
        if next_panel is None:
            next_panel = previous_panel
        next_panel = copy.deepcopy(next_panel)
        next_layout = next_panel.structure.components[
            self.component_idx].layout
        if self.attr is AttributeType.NUMBER:
            next_layout.number.value_level = \
                next_layout.number.value_level + self.value()
            next_layout.position.sample(next_layout.number.value())
            bboxes = next_layout.position.value()
            del next_layout.entities[:]
            for i, bbox in enumerate(bboxes):
                entity = copy.deepcopy(previous_layout.entities[0])
                entity.name = str(i)
                entity.bbox = bbox
                if not previous_layout.uniformity.value():
                    entity.resample()
                next_layout.insert(entity)
        elif self.attr is AttributeType.POSITION:
            next_layout.position.value_level = (
                next_layout.position.value_level +
                self.value) % next_layout.position.values.shape[0]
            for i, bbox in enumerate(next_layout.position.value()):
                next_layout.entities[i].bbox = bbox
        elif self.attr is AttributeType.ANGLE or self.attr is AttributeType.UNIFORMITY:
            raise ValueError(f"unsupported attribute: {self.attr!s}")
        elif self.attr in AttributeType:
            attr_of = lambda entity: getattr(entity, self.attr.name.lower())
            old_value_level = attr_of(previous_layout.entities[0]).value_level
            if self.previous_is_col_0 and not previous_layout.uniformity.value(
            ):
                for entity in previous_layout.entities:
                    entity_attr = attr_of(entity)
                    entity_attr.value_level = old_value_level
            for entity in next_layout.entities:
                entity_attr = attr_of(entity)
                entity_attr.value_level = old_value_level + self.value
        else:
            raise ValueError(f"attribute not in AttributeType: {self.attr}")
        self.previous_is_col_0 = not self.previous_is_col_0
        return next_panel


class Arithmetic(Rule):
    """
    Performs an arithmetic calculation on value levels: 
    
        col_2 = col_0 +/- col_1.
    
    On POSITION: ( + ) is interpreted as set union, and
    ( - ) as set difference.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Arithmetic, self).__init__(name, attr, param, component_idx)
        self.memory = []
        self.color_count = 0
        self.color_white_alarm = False

    def apply(self, previous_panel: Panel, next_panel: Optional[Panel] = None):
        previous_layout = previous_panel.structure.components[
            self.component_idx].layout
        if next_panel is None:
            next_panel = previous_panel
        next_panel = copy.deepcopy(next_panel)
        next_layout = next_panel.structure.components[
            self.component_idx].layout
        if self.attr is AttributeType.NUMBER:
            if len(self.memory) == 0:  # second column
                previous_value_level = previous_layout.number.value_level
                self.memory.append(previous_value_level)
                if self.value > 0:
                    next_layout.layout_constraint.number.max = \
                        previous_layout.layout_constraint.number.min + \
                        previous_layout.layout_constraint.number.max - \
                            previous_value_level
                else:
                    num_min_level_orig = (
                        next_layout.layout_constraint.number.min - 1) // 2
                    next_layout.layout_constraint.number.min = num_min_level_orig
                    next_layout.layout_constraint.number.max = \
                        previous_value_level - num_min_level_orig - 1
                next_layout.reset_constraint(AttributeType.NUMBER)
                next_layout.number.sample()
            else:  # third column
                col_0_value_level, col_1_value_level = \
                    self.memory.pop(), previous_layout.number.value()
                if self.value > 0:
                    next_layout.number.value_level = col_0_value_level + \
                        col_1_value_level
                else:
                    next_layout.number.value_level = col_0_value_level - \
                        col_1_value_level
            next_layout.position.sample(next_layout.number.value())
            bboxes = next_layout.position.value()
            del next_layout.entities[:]
            for i, bbox in enumerate(bboxes):
                entity = copy.deepcopy(previous_layout.entities[0])
                entity.name = str(i)
                entity.bbox = bbox
                if not previous_layout.uniformity.value():
                    entity.resample()
                next_layout.insert(entity)
        elif self.attr is AttributeType.POSITION:
            if len(self.memory) == 0:  # second column
                previous_value_level = previous_layout.position.value_level
                self.memory.append(previous_value_level)
                while True:
                    next_layout.number.sample()
                    next_layout.position.sample(next_layout.number.value())
                    if self.value > 0:
                        if not (set(previous_value_level) >= set(
                                next_layout.position.value_level)):
                            break
                    else:
                        if not (set(previous_value_level) <= set(
                                next_layout.position.value_level)):
                            break
            else:  # third column
                col_0_value_level, col_1_value_level = \
                    self.memory.pop(), previous_layout.position.value_level
                if self.value > 0:
                    col_2_value_level = list(
                        set(col_0_value_level) | set(col_1_value_level))
                else:
                    col_2_value_level = list(
                        set(col_0_value_level) - set(col_1_value_level))
                next_layout.number.value_level = len(col_2_value_level) - 1
                next_layout.position.value_level = np.array(col_2_value_level)
            bboxes = next_layout.position.value()
            del next_layout.entities[:]
            for i, bbox in enumerate(bboxes):
                entity = copy.deepcopy(previous_layout.entities[0])
                entity.name = str(i)
                entity.bbox = bbox
                if not previous_layout.uniformity.value():
                    entity.resample()
                next_layout.insert(entity)
        elif self.attr is AttributeType.SIZE:
            if len(self.memory) == 0:  # second column
                previous_value_level = previous_layout.entities[
                    0].size.value_level
                self.memory.append(previous_value_level)
                if not previous_layout.uniformity.value():
                    for entity in previous_layout.entities:
                        entity.size.value_level = previous_value_level
                if self.value > 0:
                    next_layout.entity_constraint.size.max = \
                        previous_layout.entity_constraint.size.min + \
                        previous_layout.entity_constraint.size.max - \
                            previous_value_level
                else:
                    size_min_level_orig = (
                        previous_layout.entity_constraint.size.min - 1) // 2
                    next_layout.entity_constraint.size.min = size_min_level_orig
                    next_layout.entity_constraint.size.max = \
                        previous_value_level - size_min_level_orig - 1
                entity_0 = next_layout.entities[0]
                entity_0.reset_constraint(
                    AttributeType.SIZE, next_layout.entity_constraint.size.min,
                    next_layout.entity_constraint.size.max)
                entity_0.size.sample()
                for entity in next_layout.entities[1:]:
                    entity.reset_constraint(
                        AttributeType.SIZE,
                        next_layout.entity_constraint.size.min,
                        next_layout.entity_constraint.size.max)
                    entity.size.value_level = entity_0.size.value_level
            else:  # third column
                col_0_value_level, col_1_value_level = \
                    self.memory.pop(), previous_layout.entities[0].size.value_level
                if self.value > 0:
                    col_2_value_level = col_0_value_level + col_1_value_level
                else:
                    col_2_value_level = col_0_value_level - col_1_value_level
                for entity in next_layout.entities:
                    entity.size.value_level = col_2_value_level
        elif self.attr is AttributeType.COLOR:
            self.color_count += 1
            if len(self.memory) == 0:  # second column
                # Logic here: C_12 and C_22 could not be both 0, otherwise it's impossible to distinguish + and -
                # If C_12 == 0, we set an alarm
                # Under this alarm, if C_21 == MAX and ADD rule, then resample C_21 to ensure C_22 could be other than 0
                # Similarly, if C_21 == 0 and SUB rule, then resample C_21 to ensure C_22 could be other than 0
                # Finally, loop until C_22 is not 0

                # make sure of value consistency
                previous_value_level = previous_layout.entities[
                    0].color.value_level
                # the third time you apply this rule and find C_21 == MAX/0 if +/-
                reset_previous_layout = False
                if self.color_count == 3 and self.color_white_alarm and \
                   ((self.value > 0 and previous_value_level == COLOR_MAX) or \
                    (self.value < 0 and previous_value_level == COLOR_MIN)):
                    previous_value_level = previous_layout.entities[
                        0].color.sample_new()
                    reset_previous_layout = True
                self.memory.append(previous_value_level)
                if reset_previous_layout or not previous_layout.uniformity.value(
                ):
                    for entity in previous_layout.entities:
                        entity.color.value_level = previous_value_level
                if self.value > 0:
                    next_layout.entity_constraint.color.max = \
                        previous_layout.entity_constraint.color.min + \
                        previous_layout.entity_constraint.color.max - \
                            previous_value_level
                else:
                    next_layout.entity_constraint.color.min = \
                        next_layout.entity_constraint.color.min // 2
                    next_layout.entity_constraint.color.max = previous_value_level
                entity_0 = next_layout.entities[0]
                entity_0.reset_constraint(
                    AttributeType.COLOR,
                    next_layout.entity_constraint.color.min,
                    next_layout.entity_constraint.color.max)
                entity_0.color.sample()
                # the first time you apply this rule and get C_12 == 0
                #   set the alarm
                if self.color_count == 1:
                    self.color_white_alarm = (entity_0.color.value_level == 0)
                elif self.color_count == 3 and self.color_white_alarm and \
                     entity_0.color.value_level == 0:
                    entity_0.color.value_level = entity_0.color.sample_new()
                for entity in next_layout.entities[1:]:
                    entity.reset_constraint(
                        AttributeType.COLOR,
                        next_layout.entity_constraint.color.min,
                        next_layout.entity_constraint.color.max)
                    entity.color.value_level = entity_0.color.value_level
            else:  # third column
                col_0_value_level, col_1_value_level = \
                    self.memory.pop(), previous_layout.entities[0].color.value_level
                if self.value > 0:
                    col_2_value_level = col_0_value_level + col_1_value_level
                else:
                    col_2_value_level = col_0_value_level - col_1_value_level
                for entity in next_layout.entities:
                    entity.color.value_level = col_2_value_level
        else:
            raise ValueError("unsupported attriubute")
        return next_panel


class Distribute_Three(Rule):
    """
    Three column values are from a randomly-selected, fixed set.
    """

    def __init__(self, name, attr, param, component_idx):
        super(Distribute_Three, self).__init__(name, attr, param,
                                               component_idx)
        self.value_levels = []
        self.count = 0

    def apply(self, previous_panel: Panel, next_panel: Optional[Panel] = None):
        previous_layout = previous_panel.structure.components[
            self.component_idx].layout
        if next_panel is None:
            next_panel = previous_panel
        next_panel = copy.deepcopy(next_panel)
        next_layout = next_panel.structure.components[
            self.component_idx].layout
        if self.attr is AttributeType.NUMBER:
            if self.count == 0:  # first row
                all_value_levels = list(
                    range(previous_layout.layout_constraint.number.min,
                          previous_layout.layout_constraint.number.max + 1))
                all_value_levels.pop(
                    all_value_levels.index(previous_layout.number.value_level))
                three_value_levels = np.random.choice(all_value_levels,
                                                      size=2,
                                                      replace=False)
                three_value_levels = np.insert(
                    three_value_levels, 0, previous_layout.number.value_level)
                self.value_levels.append(three_value_levels[[0, 1, 2]])
                if np.random.uniform() >= 0.5:
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                else:
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                next_layout.number.value_level = self.value_levels[0][1]
            else:
                row, col = divmod(self.count, 2)
                if col == 0:
                    previous_layout.number.value_level = self.value_levels[
                        row][0]
                    previous_layout.resample()
                    next_panel = copy.deepcopy(previous_panel)
                    next_layout = next_panel.structure.components[
                        self.component_idx].layout
                    next_layout.number.value_level = self.value_levels[row][1]
                else:
                    next_layout.number.value_level = self.value_levels[row][2]
            next_layout.position.sample(next_layout.number.value())
            del next_layout.entities[:]
            for i, bbox in enumerate(next_layout.position.value()):
                entity = copy.deepcopy(previous_layout.entities[0])
                entity.name = str(i)
                entity.bbox = bbox
                if not previous_layout.uniformity.value():
                    entity.resample()
                next_layout.insert(entity)
            self.count = (self.count + 1) % 6
        elif self.attr is AttributeType.POSITION:
            if self.count == 0:
                # sample new does not change value_level
                num = previous_layout.number.value()
                pos_0 = previous_layout.position.value_level
                pos_1 = previous_layout.position.sample_new(num)
                pos_2 = previous_layout.position.sample_new(num, [pos_1])
                three_value_levels = np.array([pos_0, pos_1, pos_2])
                self.value_levels.append(three_value_levels[[0, 1, 2]])
                if np.random.uniform() >= 0.5:
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                else:
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                next_layout.position.value_level = self.value_levels[0][1]
            else:
                row, col = divmod(self.count, 2)
                if col == 0:
                    previous_layout.number.value_level = len(
                        self.value_levels[row][0]) - 1
                    previous_layout.resample()
                    previous_layout.position.value_level = self.value_levels[
                        row][0]
                    for entity, bbox in zip(previous_layout.entities,
                                            previous_layout.position.value()):
                        entity.bbox = bbox
                    next_panel = copy.deepcopy(previous_panel)
                    next_layout = next_panel.structure.components[
                        self.component_idx].layout
                    next_layout.position.value_level = self.value_levels[row][
                        1]
                else:
                    next_layout.position.value_level = self.value_levels[row][
                        2]
            for entity, bbox in zip(next_layout.entities,
                                    next_layout.position.value()):
                entity.bbox = bbox
            self.count = (self.count + 1) % 6
        elif self.attr is AttributeType.ANGLE or self.attr is AttributeType.UNIFORMITY:
            raise ValueError("unsupported attribute")
        elif self.attr in AttributeType:
            attr_of = lambda x: getattr(x, self.attr.name.lower())
            if self.count == 0:
                all_value_levels = range(
                    attr_of(previous_layout.entity_constraint).min,
                    attr_of(previous_layout.entity_constraint).max + 1)
                three_value_levels = np.random.choice(all_value_levels,
                                                      size=3,
                                                      replace=False)
                self.value_levels.append(three_value_levels[[0, 1, 2]])
                if np.random.uniform() >= 0.5:
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                else:
                    self.value_levels.append(three_value_levels[[2, 0, 1]])
                    self.value_levels.append(three_value_levels[[1, 2, 0]])
                for entity in previous_layout.entities:
                    entity_attr = attr_of(entity)
                    entity_attr.value_level = self.value_levels[0][0]
                for entity in next_layout.entities:
                    entity_attr = attr_of(entity)
                    entity_attr.value_level = self.value_levels[0][1]
            else:
                row, col = divmod(self.count, 2)
                if col == 0:
                    value_level = self.value_levels[row][0]
                    for entity in previous_layout.entities:
                        entity_attr = attr_of(entity)
                        entity_attr.value_level = value_level
                    value_level = self.value_levels[row][1]
                    for entity in next_layout.entities:
                        entity_attr = attr_of(entity)
                        entity_attr.value_level = value_level
                else:
                    value_level = self.value_levels[row][2]
                    for entity in next_layout.entities:
                        entity_attr = attr_of(entity)
                        entity_attr.value_level = value_level
            self.count = (self.count + 1) % 6
        else:
            raise ValueError("unsupported attriubute")
        return next_panel
