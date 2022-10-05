from typing import List

import numpy as np
from scipy.special import comb

from configuration import AttributeType, RuleType
from rule import rule_type_to_factory, Rules, ComponentRules

# Rule, Attr, Param
# The design encodes rule priority order: Number/Position always comes first
# Number and Position could not both be sampled
# Progression on Number: Number on each Panel +1/2 or -1/2
# Progression on Position: Entities on each Panel roll over the layout
# Arithmetic on Number: Numeber on the third Panel = Number on first +/- Number on second (1 for + and -1 for -)
# Arithmetic on Position: 1 for SET_UNION and -1 for SET_DIFF
# Distribute_Three on Number: Three numbers through each row
# Distribute_Three on Position: Three positions (same number) through each row
# Constant on Number/Position: Nothing changes
# Progression on Type: Type progression defined as the number of edges on each entity (Triangle, Square, Pentagon, Hexagon, Circle)
# Distribute_Three on Type: Three types through each row
# Constant on Type: Nothing changes
# Progression on Size: Size on each entity +1/2 or -1/2
# Arithmetic on Size: Size on the third Panel = Size on the first +/- Size on the second (1 for + and -1 for -)
# Distribute_Three on Size: Three sizes through each row
# Constant on Size: Nothing changes
# Progression on Color: Color +1/2 or -1/2
# Arithmetic on Color: Color on the third Panel = Color on the first +/- Color on the second (1 for + and -1 for -)
# Distribute_Three on Color: Three colors through each row
# Constant on Color: Nothing changes
# Note that all rules on Type, Size and Color enforce value consistency in a panel
VALID_RULES = [[[RuleType.PROGRESSION, AttributeType.NUMBER, [-2, -1, 1, 2]],
                [RuleType.PROGRESSION, AttributeType.POSITION, [-2, -1, 1, 2]],
                [RuleType.ARITHMETIC, AttributeType.NUMBER, [1, -1]],
                [RuleType.ARITHMETIC, AttributeType.POSITION, [1, -1]],
                [RuleType.DISTRIBUTE_THREE, AttributeType.NUMBER, None],
                [RuleType.DISTRIBUTE_THREE, AttributeType.POSITION, None],
                [RuleType.CONSTANT, AttributeType.NUMBER_OR_POSITION, None]],
               [[RuleType.PROGRESSION, AttributeType.TYPE, [-2, -1, 1, 2]],
                [RuleType.DISTRIBUTE_THREE, AttributeType.TYPE, None],
                [RuleType.CONSTANT, AttributeType.TYPE, None]],
               [[RuleType.PROGRESSION, AttributeType.SIZE, [-2, -1, 1, 2]],
                [RuleType.ARITHMETIC, AttributeType.SIZE, [1, -1]],
                [RuleType.DISTRIBUTE_THREE, AttributeType.SIZE, None],
                [RuleType.CONSTANT, AttributeType.SIZE, None]],
               [[RuleType.PROGRESSION, AttributeType.COLOR, [-2, -1, 1, 2]],
                [RuleType.ARITHMETIC, AttributeType.COLOR, [1, -1]],
                [RuleType.DISTRIBUTE_THREE, AttributeType.COLOR, None],
                [RuleType.CONSTANT, AttributeType.COLOR, None]]]


def sample_rules() -> Rules:
    """
    samples the number of components and rules for each of their attributes.
    """

    rules = []
    for i in range(np.random.randint(1, 3)):
        component_rules = []
        for ruleset in VALID_RULES:
            rule_type, attribute, parameters = np.random.choice(ruleset)
            rule_factory = rule_type_to_factory(rule_type)
            component_rules.append(
                rule_factory(rule_type, attribute, parameters,
                             component_idx=i))
        rules.append(ComponentRules(component_rules))
    return Rules(rules)


# pay attention to Position Arithmetic, new entities (resample)
def sample_attr_avail(rules: Rules, panel_2_2) -> List[list]:
    """
    Sample available attributes whose values could be modified.
    :returns: list of [component_idx, attribute, available_times, constraints]

    Highly-revelant passant from original paper..

        - `Constant`: Attributes governed by this rule would not 
          change in the row.  If it is applied on `Number` or 
          `Position`, attribute values would not change across
          the three panels.  If it is applied on `Entity` level
          attributes, then we leave "as is" the attribute in each
          object across the three panels.  This design would render
          every object the same if `Uniformity` is set to `True`;
          otherwise, it will introduce noise in a problem instance.

        - `Progression`: Attribute values monotonically increase or 
          decrease in a row.  The increment or decrement could be either 
          1 or 2, resulting in 4 instances in this rule.

        - `Arithmetic`: There are 2 instantiations in this rule,
          resulting in either a rule of summation or one of subtraction.
          `Arithmetic` derives the value of the attribute in the third
          panel from the first two panels.  For `Position`, this rule is 
          implemented as set arithmetics.

        - `Distribute Three`: This rule first samples 3 values of 
          an attribute in a problem instance and permutes the values 
          in different rows.

        Among all attributes, we realize that `Number` and `Position`
        are strongly coupled, hence we do not allow non-`Constant` rules
        to co-occur on both of the attributes.  With 4 rules and 5 attributes,
        we could have had 20 rule-attribute combinations.  However, we 
        exclude `Arithmetic` on `Type`, as it is counterintuitive, resulting
        in 19 combinations in total. 

    """
    samples = []
    for i, component_rules in enumerate(rules):
        start_panel_layout = panel_2_2.structure.components[i].layout
        panel_2_2_layout = panel_2_2.structure.components[i].layout
        # Number/Position
        # If Rule on Number: Only change Number
        # If Rule on Position: Both Number and Position could be changed
        num = panel_2_2_layout.number.value()
        max_entities = start_panel_layout.position.values.shape[0]
        if component_rules.number_position.attr is AttributeType.NUMBER:
            num_times = 0
            min_level = start_panel_layout.orig_layout_constraint.number.min
            max_level = start_panel_layout.orig_layout_constraint.number.max
            for n_entities in range(min_level + 1, max_level + 2):
                if n_entities != num:
                    num_times += comb(max_entities, n_entities)
            if num_times > 0:
                samples.append(
                    [i, AttributeType.NUMBER, num_times, min_level, max_level])
        # Constant or on Position
        else:
            num_times = 0
            min_level = start_panel_layout.orig_layout_constraint.number.min
            max_level = start_panel_layout.orig_layout_constraint.number.max
            for n_entities in range(min_level + 1, max_level + 2):
                if n_entities != num:
                    num_times += comb(max_entities, n_entities)
            if num_times > 0:
                samples.append(
                    [i, AttributeType.NUMBER, num_times, min_level, max_level])
            pos_times = comb(max_entities, panel_2_2_layout.number.value()) - 1
            if pos_times > 0:
                samples.append(
                    [i, AttributeType.POSITION, pos_times, None, None])
        # Type, Size, Color
        for rule in component_rules:
            if rule.name is not RuleType.CONSTANT or \
                panel_2_2_layout.uniformity.value() or \
                component_rules.number_position.name is RuleType.CONSTANT or \
                (component_rules.number_position.attr is AttributeType.POSITION and \
                 component_rules.number_position.name is not RuleType.ARITHMETIC):
                bounds = getattr(start_panel_layout.orig_entity_constraint,
                                 rule.attr.name.lower())
                if bounds.max - bounds.min > 0:
                    samples.append([
                        i, rule.attr, bounds.max - bounds.min, min_level,
                        max_level
                    ])
    return samples


def sample_attr(attributes):
    """
    sample an attribute from a list of sampleable attributes and reduce this attributes count 
    of available samples by one; if this reduces the count to zero, remove this entry from 
    the list.
    """
    i = np.random.choice(len(attributes))
    component_idx, attr_name, available_samples, min_level, max_level = attributes[
        i]
    if available_samples - 1 == 0:
        del attributes[i]
    else:
        attributes[i][2] = available_samples - 1
    return component_idx, attr_name, min_level, max_level
