from typing import Optional
import copy

from .attribute import AttributeType
from .component import Component
from .rule import Rules, RuleType


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

    def __str__(self):
        s = f"{self.component_1.config}\n{self.component_1.uniformity}\n\n"
        for entity in self.component_1.entities:
            s += f"{entity!r}\n"
        if hasattr(self, "component_2"):
            s += f"\n\t----------- \\\\ {self.component_1.component_type.name} > {self.component_1.layout_type.name} // --- " + \
                 f"// {self.component_2.component_type.name} > {self.component_2.layout_type.name} \\\\ -----------\n\n"
            s += f"{self.component_2.config}\n{self.component_2.uniformity}\n"
            for entity in self.component_2.entities:
                s += f"{entity!r}\n"
        return s


def prune(base: Panel, rules: Rules) -> Optional[Panel]:
    """
    Modify the bounds of attributes based on the rules that will be applied
    to them to ensure those rules can be properly expressed with the given
    range of values; if this cannot be done by tightening bounds, returns None.
    If rules cannot be applied, the panel is reset such that `self.constraints`
    does not differ from its state before the method was called.

    Note that the bounds in base.components[i].constraints.number are the maximal
    and minimal number of objects that may be placed in this layout.  In particular,

        base.components[i].constraints.number == \
            base.components[i].config.position.value.shape[0]
     
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
                        # can increment/decrement layout slot indices twice without repetition:
                        #   bounds.max >= bounds.min + 2 * abs(rule.value)
                        bounds.max = bounds.max - 2 * abs(rule.value)
                    elif bounds:
                        # bounds.max >= bounds.min + 2 * rule.value
                        if rule.value > 0:
                            bounds.max = bounds.max - 2 * rule.value
                        else:
                            bounds.min = bounds.min - 2 * rule.value
                elif rule.name is RuleType.ARITHMETIC:
                    if rule.attr is AttributeType.POSITION:
                        # sets must be incomplete (union) and overlapping (difference)
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
                        # n choose k entity arrangements in layout with n slots;
                        # `value_of_setting(bounds.min)` >= 1 and we restrict
                        # n >= 3 and k < n so n choose k >= 3 (see Pascal's triangle).
                        n_slots = component.config.number.value_of_setting(
                            bounds.max)
                        if n_slots < 3:
                            return None
                        else:
                            bounds.max = bounds.max - 1
                    elif bounds:
                        # u - l + 1 settings in [u, l]: require >= 3
                        if bounds.max - bounds.min + 1 < 3:
                            return None
                if bounds.max < bounds.min:  # reset bounds if pruned
                    bounds.min, bounds.max = initial_min, initial_max
                    return None
            else:
                return None
    return pruned
