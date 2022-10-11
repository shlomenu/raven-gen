from typing import Optional
import copy

from attribute import AttributeType
from component import Component
from rule import Rules, RuleType


class Panel:

    def __init__(self,
                 component_1: Component,
                 component_2: Optional[Component] = None):

        self.component_1 = component_1
        if component_2:
            self.component_2 = component_2
            self.components = (self.component_1, self.component_2)
        else:
            self.components = (self.component_1, )


def prune(base: Panel, rules: Rules) -> Optional[Panel]:
    """
    Modify the bounds of attributes based on the rules that will be applied
    to them to ensure those rules can be properly expressed with the given
    range of values; if this cannot be done by tightening bounds, returns None.
    If rules cannot be applied, the panel is reset such that `self.constraints`
    does not differ from its state before the method was called.
    """
    pruned = copy.deepcopy(base)
    for component, component_rules in zip(pruned.components, rules):
        del component.entities[:]
        for rule in component_rules.all:
            if rule.attr in AttributeType and \
                    rule.attr is not AttributeType.ANGLE and \
                    rule.attr is not AttributeType.UNIFORMITY:
                if rule.attr is AttributeType.NUMBER or rule.attr is AttributeType.POSITION or \
                    rule.attr is AttributeType.CONFIGURATION:
                    bounds = getattr(component.constraints,
                                     AttributeType.NUMBER.name.lower())
                else:
                    bounds = getattr(component.constraints,
                                     rule.attr.name.lower())
                initial_min, initial_max = bounds.min, bounds.max
                if rule.name is RuleType.PROGRESSION:
                    if rule.attr is AttributeType.POSITION:
                        # bounds.max is setting indicating number of positions
                        # in this layout;
                        #
                        #   bounds.max >= bounds.min + 2 * abs(rule.value)
                        #
                        # ensures that entities can be shifted `rule.value`
                        # positions over before landing in a position that
                        # was occupied within the row.
                        bounds.max = bounds.max - 2 * abs(rule.value)
                    elif bounds:
                        # bounds.max >= bounds.min + 2 * rule.value
                        if rule.value > 0:
                            bounds.max = bounds.max - 2 * rule.value
                        else:
                            bounds.min = bounds.min - 2 * rule.value
                elif rule.name is RuleType.ARITHMETIC:
                    if rule.attr is AttributeType.POSITION:
                        # Forbid filling all positions so that set union
                        #  appears different from constancy; disallow the
                        #  sampling of disjoint subsets so that set difference
                        #  is visible.
                        if rule.value <= 0:
                            bounds.min = bounds.max // 2
                        bounds.max = bounds.max - 1
                    elif rule.attr is AttributeType.NUMBER or rule.attr is AttributeType.SIZE:
                        # bounds.max >= 2 * bounds.min + 1
                        if rule.value > 0:
                            bounds.max = bounds.max - bounds.min - 1
                        else:
                            bounds.min = 2 * bounds.min + 1
                    elif rule.attr is AttributeType.COLOR:
                        # bounds.max >= bounds.min + 1
                        #   && bounds.max >= 2 * bounds.min
                        if bounds.max - bounds.min < 1:
                            return None
                        elif rule.value > 0:
                            bounds.max = bounds.max - bounds.min
                        elif rule.value < 0:
                            bounds.min = 2 * bounds.min
                elif rule.name is RuleType.DISTRIBUTE_THREE:
                    if rule.attr is AttributeType.POSITION:
                        # There are n choose k sets of positions in which to
                        # place k entities within a panel containing at most
                        # n entities (n is initially given by `bounds.max + 1`).
                        # Since the lowest setting of number corresponds to one
                        # entity, k >= 1.  If we restrict n >= 3 and k < n, then
                        # n choose k >= 3, as is required to apply the rule.
                        if bounds.max + 1 < 3:
                            return None
                        else:
                            bounds.max = bounds.max - 1
                    elif bounds:
                        # require three distinct settings
                        if bounds.max - bounds.min + 1 < 3:
                            return None
                if bounds.max < bounds.min:
                    bounds.min, bounds.max = initial_min, initial_max
                    return None
            else:
                return None
    return pruned
