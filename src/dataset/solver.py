from typing import List

import numpy as np

from configuration import AttributeType, RuleType
from rule import Rules
from panel import Panel


def solve(rules: Rules, context: List[Panel], candidates):
    """
    search-based heuristic solver.
    :param rule_groups: rules that apply to each component
    :param context: list of eight context panels in row-major order
    :param candidates: a list of eight candidate answer AoTs
    :returns: index of the correct answer in the candidates
    """
    satisfied = [0] * len(candidates)
    for i, candidate in enumerate(candidates):
        # note that rule.component_idx should be the same as j
        for component_rules in rules:
            number_and_position_rule = component_rules[0]
            satisfied[i] += check_number_and_position(number_and_position_rule,
                                                      context, candidate)
            if number_and_position_rule.attr is AttributeType.NUMBER or \
                   number_and_position_rule.name is RuleType.ARITHMETIC:
                regenerate = True
            else:
                regenerate = False
            satisfied[i] += check_entity(component_rules[1], context,
                                         candidate, AttributeType.TYPE,
                                         regenerate)
            satisfied[i] += check_entity(component_rules[2], context,
                                         candidate, AttributeType.SIZE,
                                         regenerate)
            satisfied[i] += check_entity(component_rules[3], context,
                                         candidate, AttributeType.COLOR,
                                         regenerate)
    satisfied = np.array(satisfied)
    answer_set = np.where(satisfied == max(satisfied))[0]
    return np.random.choice(answer_set)


def check_number_and_position(number_and_position_rule, context, candidate):
    """
    check whether rule is satisfied for its layout attribute.
    :param number_and_position_rule: rule to check
    :param context: the 8 context panels 
    :param candidate: the proposed solution panel
    :returns: 0 if failure, 1 if success
    """
    success = 0
    c = number_and_position_rule.component_idx
    panel_2_0_layout = context[6].structure.components[c].layout
    panel_2_1_layout = context[7].structure.components[c].layout
    candidate_layout = candidate.structure.components[c].layout
    if number_and_position_rule.name is RuleType.CONSTANT:
        panel_2_0_pos = set(panel_2_0_layout.position.value_level)
        panel_2_1_pos = set(panel_2_1_layout.position.value_level)
        candidate_pos = set(candidate_layout.position.value_level)
        # note that set equal only when len(Number) equal and content equal
        if candidate_pos == panel_2_0_pos and candidate_pos == panel_2_1_pos:
            success = 1
    elif number_and_position_rule.name is RuleType.PROGRESSION:
        if number_and_position_rule.attr is AttributeType.NUMBER:
            panel_2_0_num = panel_2_0_layout.number.value_level
            panel_2_1_num = panel_2_1_layout.number.value_level
            candidate_num = candidate_layout.number.value_level
            if panel_2_1_num * 2 == panel_2_0_num + candidate_num:
                success = 1
        else:
            panel_2_0_pos = panel_2_0_layout.position.value_level
            panel_2_1_pos = panel_2_1_layout.position.value_level
            candidate_pos = candidate_layout.position.value_level
            most_num = len(candidate_layout.position.values)
            diff = number_and_position_rule.value
            if (set((panel_2_0_pos + diff) % most_num) == set(panel_2_1_pos)) and \
               (set((panel_2_1_pos + diff) % most_num) == set(candidate_pos)):
                success = 1
    elif number_and_position_rule.name is RuleType.ARITHMETIC:
        mode = number_and_position_rule.value
        if number_and_position_rule.attr is AttributeType.NUMBER:
            panel_2_0_num = panel_2_0_layout.number.value()
            panel_2_1_num = panel_2_1_layout.number.value()
            candidate_num = candidate_layout.number.value()
            if mode > 0 and (candidate_num == panel_2_0_num + panel_2_1_num):
                success = 1
            if mode < 0 and (candidate_num == panel_2_0_num - panel_2_1_num):
                success = 1
        else:
            panel_2_0_pos = set(panel_2_0_layout.position.value_level)
            panel_2_1_pos = set(panel_2_1_layout.position.value_level)
            candidate_pos = set(candidate_layout.position.value_level)
            if mode > 0 and (candidate_pos == panel_2_0_pos | panel_2_1_pos):
                success = 1
            if mode < 0 and (candidate_pos == panel_2_0_pos - panel_2_1_pos):
                success = 1
    else:
        three_values = number_and_position_rule.value_levels[2]
        if number_and_position_rule.attr is AttributeType.NUMBER:
            panel_2_0_num = panel_2_0_layout.number.value_level
            panel_2_1_num = panel_2_1_layout.number.value_level
            candidate_num = candidate_layout.number.value_level
            if panel_2_0_num == three_values[0] and \
               panel_2_1_num == three_values[1] and \
               candidate_num == three_values[2]:
                success = 1
        else:
            panel_2_0_pos = panel_2_0_layout.position.value_level
            panel_2_1_pos = panel_2_1_layout.position.value_level
            candidate_pos = candidate_layout.position.value_level
            if set(panel_2_0_pos) == set(three_values[0]) and \
               set(panel_2_1_pos) == set(three_values[1]) and \
               set(candidate_pos) == set(three_values[2]):
                success = 1
    return success


def check_consistency(candidate, attribute_name, c):
    candidate_layout = candidate.structure.components[c].layout
    entity_0 = candidate_layout.entities[0]
    entity_0_value = getattr(entity_0, attribute_name).value_level
    for entity in candidate_layout.entities[1:]:
        entity_value = getattr(entity, attribute_name).value_level
        if entity_value != entity_0_value:
            return False
    return True


def check_entity(rule, context, candidate, attribute, regenerate):
    """
    check whether rule is satisfied for specified attribute of an entity.
    :param rule:
    :param context: list of 8 context panels 
    :param candidate: the proposed solution panel
    :param attribute: attribute to which rule is applied
    :returns: 0 if failure, 1 if success
    """
    success = 0
    c = rule.component_idx
    panel_2_0_layout = context[6].structure.components[c].layout
    panel_2_1_layout = context[7].structure.components[c].layout
    candidate_layout = candidate.structure.components[c].layout
    attribute_name = attribute.name.lower()
    if rule.name is RuleType.CONSTANT:
        if candidate_layout.uniformity.value():
            if check_consistency(candidate, attribute_name, component_idx=c) and \
               getattr(candidate_layout.entities[0], attribute_name).value_level == \
               getattr(panel_2_1_layout.entities[0], attribute_name).value_level:
                success = 1
        else:
            panel_2_0_num = panel_2_0_layout.number.value_level
            panel_2_1_num = panel_2_1_layout.number.value_level
            candidate_num = candidate_layout.number.value_level
            if (panel_2_0_num != panel_2_1_num) or (
                    panel_2_1_num != candidate_num) or regenerate:
                success = 1
            else:
                nobreak = True
                for candidate_entity, panel_2_1_entity in \
                    zip(candidate_layout.entities, panel_2_1_layout.entities):
                    if getattr(candidate_entity, attribute_name).value_level != \
                       getattr(panel_2_1_entity, attribute_name).value_level:
                        nobreak = False
                        break
                if nobreak:
                    success = 1
    elif rule.name is RuleType.PROGRESSION:
        if check_consistency(candidate, attribute_name, component_idx=c):
            panel_2_0_value = getattr(panel_2_0_layout.entities[0],
                                      attribute_name).value_level
            panel_2_1_value = getattr(panel_2_1_layout.entities[0],
                                      attribute_name).value_level
            candidate_value = getattr(candidate_layout.entities[0],
                                      attribute_name).value_level
            if panel_2_1_value * 2 == panel_2_0_value + candidate_value:
                success = 1
    elif rule.name is RuleType.ARITHMETIC:
        if check_consistency(candidate, attribute_name, component_idx=c):
            panel_2_0_value = getattr(panel_2_0_layout.entities[0],
                                      attribute_name).value_level
            panel_2_1_value = getattr(panel_2_1_layout.entities[0],
                                      attribute_name).value_level
            candidate_value = getattr(candidate_layout.entities[0],
                                      attribute_name).value_level
            if rule.value > 0:
                if attribute is AttributeType.COLOR:
                    if candidate_value == panel_2_0_value + panel_2_1_value:
                        success = 1
                else:
                    if candidate_value == panel_2_0_value + panel_2_1_value + 1:
                        success = 1
            if rule.value < 0:
                if attribute is AttributeType.COLOR:
                    if candidate_value == panel_2_0_value - panel_2_1_value:
                        success = 1
                else:
                    if candidate_value == panel_2_0_value - panel_2_1_value - 1:
                        success = 1
    else:
        if check_consistency(candidate, attribute_name, component_idx=c):
            panel_2_0_value = getattr(panel_2_0_layout.entities[0],
                                      attribute_name).value_level
            panel_2_1_value = getattr(panel_2_1_layout.entities[0],
                                      attribute_name).value_level
            candidate_value = getattr(candidate_layout.entities[0],
                                      attribute_name).value_level
            three_values = rule.value_levels[2]
            if panel_2_0_value == three_values[0] and \
               panel_2_1_value == three_values[1] and \
               candidate_value == three_values[2]:
                success = 1
    return success
