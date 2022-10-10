import copy
import random
from typing import List, Optional
import json

import numpy as np
from scipy.special import comb

from panel import Panel
from attribute import AttributeType, COLOR_MAX, COLOR_MIN
from rule import Rules, RuleType
from rendering import IMAGE_SIZE


# Note that all rules on Type, Size and Color enforce value consistency in a panel

class Matrix:

    def __init__(self, base):
        self.panels = []
        while True:
            self.rules = Rules.make_random()
            start = base.prune(self.rules)
            if start is not None:
                self.start = start
                break

        self.panels.extend(self.make_row())
        self.panels.extend(self.make_row(resample=True))
        self.panels.extend(self.make_row(resample=True))

        self.context = self.panels[:-1]
        self.panel_imgs = [panel.render() for panel in self.context] + \
            [np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)]
        self.answer = self.panels[-1]

        modifiable_attr = self.sample_modifiable()
        ground_truth = copy.deepcopy(self.answer)
        self.candidates = [ground_truth]
        for component in ground_truth.components:
            component.reset_history()
        for _ in range(7):
            c, attr = self.sample_attr(
                modifiable_attr)
            alternative = copy.deepcopy(ground_truth)
            alternative.components[c].sample_unique(
                attr, ground_truth.components[c])
            alternative.sample_unique(c, attr, ground_truth)
            self.candidates.append(alternative)

        random.shuffle(self.candidates)
        self.candidate_imgs = [candidate.render()
                               for candidate in self.candidates]
        self.answer_location = self.candidates.index(ground_truth)

        # imsave(generate_matrix_answer(imgs + answers), "./experiments/fuse/{}.jpg".format(k))
        # imsave(generate_matrix_answer(imgs + answers), "./experiments/{}/{}.jpg".format(key, k))

    def make_row(self, resample=False):
        col_0, row = copy.deepcopy(self.start), None
        if resample:
            for component in col_0.components:
                component.sample(sample_position=True,
                                 sample_number=True, carryover=False)
        for c, component_rules in enumerate(self.rules):
            col_1 = self.apply_rule(component_rules.number_or_position, col_0)
            col_2 = self.apply_rule(component_rules.number_or_position, col_1)
            for rule in component_rules:
                col_1 = self.apply_rule(rule, col_0, col_1)
            for rule in component_rules:
                col_2 = self.apply_rule(rule, col_1, col_2)
            if c == 0:
                row = [col_0, col_1, col_2]
            else:
                row[1].components[c] = col_1.components[c]
                row[2].components[c] = col_2.components[c]
        return row

    def sample_modifiable(self) -> List[list]:
        """
        Sample available attributes whose values could be modified.
        :returns: list of [component_idx, attribute, available_times, constraints]
        """
        samples = []
        for c, (component, component_rules) in enumerate(zip(self.answer.components, self.rules)):
            n_samples, max_entities = 0, component.config.position.values.shape[0]
            for n_entities in range(component.initial_constraints.number.min + 1,
                                    component.initial_constraints.number.max + 2):
                if n_entities != component.config.number.value:
                    n_samples += comb(max_entities, n_entities)
            if n_samples > 0:
                samples.append([c, AttributeType.NUMBER, n_samples])
            if component_rules.number_or_position.attr is not AttributeType.NUMBER:
                position_samples = comb(
                    max_entities, component.config.number.value) - 1
                if position_samples > 0:
                    samples.append(
                        [c, AttributeType.POSITION, position_samples])
            for rule in component_rules:
                if rule.name is not RuleType.CONSTANT or \
                    component.uniformity.value or \
                    component_rules.number_or_position.name is RuleType.CONSTANT or \
                    (component_rules.number_or_position.attr is AttributeType.POSITION and
                     component_rules.number_or_position.name is not RuleType.ARITHMETIC):
                    bounds = getattr(
                        component.initial_constraints, rule.attr.name.lower())
                    if bounds.max - bounds.min > 0:
                        samples.append([c, rule.attr, bounds.max - bounds.min])
        return samples

    def sample_attr(self, attributes):
        """
        Sample an attribute from a list of sampleable attributes and reduce this attributes count
        of available samples by one; if this reduces the count to zero, remove this entry from
        the list.
        """
        i = np.random.choice(len(attributes))
        component_idx, attr_name, available_samples = attributes[
            i]
        if available_samples - 1 == 0:
            del attributes[i]
        else:
            attributes[i][2] = available_samples - 1
        return component_idx, attr_name

    def apply_rule(self, rule, previous_panel: Panel, next_panel: Optional[Panel] = None) -> Panel:
        """
        Apply the rule to a component of a panel.

        It is assumed that this method will be called first by 
        the rule on NUMBER/POSITION attributes, and then by 
        rules on TYPE, SIZE, or COLOR.  In the first case, `next_panel` 
        is not provided and all other non-uniform attributes are 
        resampled to provide variation which may or may not be 
        overwritten in the application of other rules.  In the 
        latter case, `next_panel` is provided and differences
        between it and `previous_panel` are preserved.  
        """
        if rule.attr is AttributeType.ANGLE or \
                rule.attr is AttributeType.UNIFORMITY or \
                rule.attr not in AttributeType:
            raise ValueError("unsupported attribute")
        elif rule.name is RuleType.CONSTANT:
            if next_panel is None:
                next_panel = previous_panel
            return copy.deepcopy(next_panel)
        elif rule.name is RuleType.PROGRESSION:
            prev_comp = previous_panel.components[rule.component_idx]
            if next_panel is None:
                next_panel = previous_panel
            next_panel = copy.deepcopy(next_panel)
            next_comp = next_panel.components[rule.component_idx]
            if rule.attr is AttributeType.NUMBER:
                next_comp.config.number.setting += rule.value
                next_comp.sample(sample_position=True)
            elif rule.attr is AttributeType.POSITION:
                next_comp.config.position.setting = (
                    next_comp.config.position.setting +
                    rule.value) % next_comp.config.position.values.shape[0]
                next_comp.set_position()
            else:
                if rule.previous_is_col_0 and not prev_comp.uniformity.value:
                    prev_comp.make_uniform(rule.attr)
                next_comp.set_uniform(
                    rule.attr, prev_comp.setting_of(rule.attr) + rule.value)
        elif rule.name is RuleType.ARITHMETIC:
            if rule.attr is AttributeType.TYPE:
                raise ValueError("unsupported attribute")
            prev_comp = previous_panel.components[rule.component_idx]
            if next_panel is None:
                next_panel = previous_panel
            next_panel = copy.deepcopy(next_panel)
            next_comp = next_panel.components[rule.component_idx]
            if rule.attr is AttributeType.NUMBER:
                if rule.col_0_setting is None:  # second column
                    rule.col_0_setting = prev_comp.config.number.setting
                    rule.set_constraints_col_1(prev_comp, next_comp)
                    next_comp.config.number.sample(next_comp.constraints)
                else:  # third column
                    next_comp.config.number.setting = rule.col_2_setting(
                        prev_comp.config.number.value)
                next_comp.sample(sample_position=True)
            elif rule.attr is AttributeType.POSITION:
                if rule.col_0_setting is None:  # second column
                    rule.col_0_setting = prev_comp.config.position.setting
                    rule.set_constraints_col_1(prev_comp, next_comp)
                else:  # third column
                    col_2_setting = rule.col_2_setting(
                        prev_comp.config.position.setting)
                    next_comp.config.number.setting = len(
                        col_2_setting) - 1
                    next_comp.config.position.setting = np.array(
                        col_2_setting)
                next_comp.sample()
            elif rule.attr is AttributeType.SIZE:
                if rule.col_0_setting is None:  # second column
                    prev_setting = prev_comp.setting_of(rule.attr)
                    rule.col_0_setting = prev_setting
                    if not prev_comp.uniformity.value:
                        prev_comp.make_uniform(rule.attr)
                    rule.set_constraints_col_1(prev_comp, next_comp)
                    next_comp.attr(rule.attr).sample(next_comp.constraints)
                    next_comp.make_uniform(rule.attr)
                else:  # third column
                    next_comp.set_uniform(rule.attr, rule.col_2_setting(
                        prev_comp.setting_of(rule.attr)))
            elif rule.attr is AttributeType.COLOR:
                rule.color_count += 1
                if rule.col_0_setting is None:
                    # The arithmetic rule faces the challenge that the second
                    #  panel setting could be 0, in which case plus and minus
                    #  are not distinguishable.  This is handled for NUMBER and
                    #  SIZE by adding an offset of 1 to the addend/subtrahend
                    #  The original authors didn't like this solution here.
                    #  They may have wanted to ensure white was still used.
                    #  So instead the (1, 1) color is resampled if the
                    #  (1, 0) color is extreme and the same extreme
                    #  value (white) appeared in (0, 1).
                    prev_setting = prev_comp.setting_of(rule.attr)
                    reset_previous = False
                    if rule.color_count == 3 and rule.color_white_alarm and \
                        ((rule.value > 0 and prev_setting == COLOR_MAX) or
                            (rule.value < 0 and prev_setting == COLOR_MIN)):
                        prev_comp.attr(rule.attr).sample_unique(
                            prev_comp.constraints,
                            prev_comp.history,
                            record=True,
                            overwrite=True)
                        prev_setting = prev_comp.setting_of(rule.attr)
                        reset_previous = True
                    rule.col_0_setting = prev_setting
                    if reset_previous or not prev_comp.uniformity.value:
                        prev_comp.make_uniform(rule.attr)
                    rule.set_constraints_col_1(prev_comp, next_comp)
                    next_comp.attr(rule.attr).sample(next_comp.constraints)
                    if rule.color_count == 1:
                        rule.color_white_alarm = (
                            next_comp.setting_of(rule.attr) == 0)
                    elif rule.color_count == 3 and rule.color_white_alarm and \
                            next_comp.setting_of(rule.attr) == 0:
                        next_comp.attr(rule.attr).sample_unique(
                            next_comp.constraints, next_comp.history,
                            record=True, overwrite=True)
                    next_comp.make_uniform(rule.attr)
                else:  # third column
                    next_comp.set_uniform(
                        rule.attr,
                        rule.col_2_setting(prev_comp.setting_of(rule.attr)))
        elif rule.name is RuleType.DISTRIBUTE_THREE:
            prev_comp = previous_panel.components[rule.component_idx]
            if next_panel is None:
                next_panel = previous_panel
            next_panel = copy.deepcopy(next_panel)
            next_comp = next_panel.components[rule.component_idx]
            if rule.attr is AttributeType.NUMBER:
                if rule.count == 0:  # first row
                    rule.create_settings(np.insert(
                        np.random.choice(
                            list(range(prev_comp.config.number.min,
                                       prev_comp.config.number.setting)) +
                            list(range(prev_comp.config.number.setting + 1,
                                       prev_comp.config.number.max + 1)),
                            size=2,
                            replace=False),
                        0, prev_comp.config.number.setting))
                    next_comp.config.number.setting = rule.settings[0][1]
                else:
                    row, col = divmod(rule.count, 2)
                    if col == 0:
                        prev_comp.config.number.setting = rule.settings[
                            row][0]
                        prev_comp.sample(sample_position=True)
                        next_panel = copy.deepcopy(previous_panel)
                        next_comp = next_panel.components[rule.component_idx]
                        next_comp.config.number.setting = rule.settings[row][1]
                    else:
                        next_comp.config.number.setting = rule.settings[row][2]
                next_comp.sample(sample_position=True)
            elif rule.attr is AttributeType.POSITION:
                if rule.count == 0:
                    rule.create_settings(np.array([
                        prev_comp.config.position.setting,
                        prev_comp.config.position.sample_unique(
                            prev_comp.config.number.value, prev_comp.history),
                        prev_comp.config.position.sample_unique(
                            prev_comp.config.number.value, prev_comp.history)]))
                    next_comp.config.position.setting = rule.settings[0][1]
                else:
                    row, col = divmod(rule.count, 2)
                    if col == 0:
                        prev_comp.config.position.setting = rule.settings[
                            row][0]
                        prev_comp.sample()
                        next_panel = copy.deepcopy(previous_panel)
                        next_comp = next_panel.components[rule.component_idx]
                        next_comp.config.position.setting = rule.settings[row][
                            1]
                    else:
                        next_comp.config.position.setting = rule.settings[row][
                            2]
                next_comp.sample()
            else:
                def attr_of(x): return getattr(x, rule.attr.name.lower())
                if rule.count == 0:
                    rule.create_settings(np.random.choice(
                        a=range(attr_of(prev_comp.constraints).min,
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
        return next_panel

    def json(self):
        return json.dumps({
            "panels": [panel.json() for panel in self.panels],
            "candidates": [candidate.json() for candidate in self.candidates],
            "answer_index": self.answer_location
        })
