from __future__ import annotations
from typing import Optional
import copy

import numpy as np
from scipy.special import comb

from attribute import Angle, Color, Number, Position, Size, Type, Uniformity
from configuration import LevelType, NodeType, AttributeType
from constraints import rule_constraint
from rule import Rules


class Level:
    """
    Superclass for a level of organization within a `Panel`.  The following 
    interface is expected of all subclasses:

        `(_)sample`: return a new panel backed by different memory and constrained
            randomly-sampled values of the POSITION, TYPE, SIZE, COLOR, and ANGLE
            without modifying the original.

        `(_)resample`: update attributes in place with constrained randomly-sampled
            values of POSITION, TYPE, SIZE, COLOR, and ANGLE.  If `resample_number=True`
            is passed, NUMBER will be resampled as well.

        `(_)sample_new`: update attributes in place with constrained randomly-sampled
            values of the specified attribute which have not been sampled via this 
            method before.  

    This API is somewhat counter to the one found in `attribute.py`; there, `sample`
    is an in-place method and `sample_new` is not in-place.  Here, it is opposite.
    Only this API is directly relevant to understanding `main.py`.

    Some additional methods are provided by specific subclasses of Level. The above 
    API still has "entry points" at specific levels contained within a Panel, as 
    indicated by the presence of absence of a leading underscore.  This design
    may be refactored in the future to offer all relevant functionality as top-level
    methods of `Panel`, as the inherited usage of subclassing seems to create more 
    code duplication than it eliminates.
    """

    def __init__(self,
                 name: str,
                 level: LevelType,
                 node_type: NodeType,
                 is_pg=False):
        self.name = name
        self.level = level
        self.node_type = node_type
        self.children = []
        self.is_pg = is_pg

    def insert(self, node: Level):
        assert isinstance(node, Level)
        assert self.node_type is not NodeType.LEAF
        assert node.level is self.level.below()
        self.children.append(node)

    def _resample(self, resample_number: bool):
        """
        Resample the layout. If the number of entities change, resample also the 
        position distribution; otherwise only resample each attribute for each entity.
        :param resample_number: whether the number has been reset
        """
        assert self.is_pg
        if self.node_type is NodeType.AND:
            for child in self.children:
                child._resample(resample_number)
        else:
            self.children[0]._resample(resample_number)

    def __repr__(self):
        return self.level + "." + self.name

    def __str__(self):
        return self.level + "." + self.name


class Panel(Level):

    def __init__(self, name, is_pg=False):
        super(Panel, self).__init__(name,
                                    level=LevelType.ROOT,
                                    node_type=NodeType.OR,
                                    is_pg=is_pg)

    @property
    def structure(self):
        return self.children[0]

    @property.setter
    def structure(self, structure: Structure):
        assert (len(self.children) <= 1)
        del self.children[:]
        self.insert(structure)

    def sample(self) -> Level:
        if self.is_pg:
            raise ValueError("cannot sample on a PG")
        new_panel = Panel(self.name, is_pg=self.is_pg)
        new_panel.structure = self.structure._sample()
        return new_panel

    def resample(self, resample_number=False):
        self._resample(resample_number)

    def prune(self, rules: Rules) -> Optional[Level]:
        """
        Prune the AoT such that all branches satisfy the constraints; not in-place. 
        """
        new_panel = Panel(self.name)
        if len(self.structure.components) == len(rules):
            new_structure = self.structure._prune(rules)
            if new_structure is not None:
                new_panel.structure = new_structure
        if len(new_panel.children) == 0:
            new_panel = None
        return new_panel

    def prepare(self):
        """
        Retrieve data used to render constituent entities.
        :returns: the type of structure contained at this root and all 
            constituent entities of each component and its layout.
        """
        assert self.is_pg
        assert self.level is LevelType.SCENE
        return (self.structure.name, [
            entity for component in self.structure.components
            for entity in component.layout.entities
        ])

    def sample_new(self, component_idx: int, attribute_name: AttributeType,
                   min_level: int, max_level: int, panel: Level):
        assert self.is_pg
        self.structure._sample_new(component_idx, attribute_name, min_level,
                                   max_level, panel.structure)


class Structure(Level):

    def __init__(self, name, is_pg=False):
        super(Structure, self).__init__(name,
                                        level=LevelType.STRUCTURE,
                                        node_type=NodeType.AND,
                                        is_pg=is_pg)

    @property
    def components(self):
        return self.children

    def insert(self, component: Component):
        super(Structure, self).insert(component)
        assert (len(self.components) <= 2)

    def _sample(self):
        if self.is_pg:
            raise ValueError("Could not sample on a PG")
        new_structure = Structure(self.name, is_pg=self.is_pg)
        for component in self.components:
            new_structure.insert(component._sample())
        return new_structure

    def _prune(self, rules):
        new_structure = Structure(self.name)
        # all components must satisfy their corresponding rules
        for component_rules, component in zip(rules, self.components):
            new_component = component._prune(component_rules)
            if new_component is None:
                return None
            new_structure.insert(new_component)
        return new_structure

    def _sample_new(self, c: int, attribute_name: AttributeType,
                    min_level: int, max_level: int, structure: Structure):
        self.components[c]._sample_new(attribute_name, min_level, max_level,
                                       structure.components[c])


class Component(Level):

    def __init__(self, name, is_pg=False):
        super(Component, self).__init__(name,
                                        level=LevelType.COMPONENT,
                                        node_type=NodeType.OR,
                                        is_pg=is_pg)

    @property
    def layout(self):
        return self.children[0]

    @property.setter
    def layout(self, layout: Layout):
        assert (len(self.children) <= 1)
        del self.children[:]
        self.insert(layout)

    def _sample(self):
        if self.is_pg:
            raise ValueError("Could not sample on a PG")
        new_component = Component(self.name, is_pg=self.is_pg)
        new_component.insert(self.layout._sample())
        return new_component

    def _prune(self, rules: Rules):
        new_component = Component(self.name)
        new_layout = self.layout._update_constraint(rules)
        if new_layout is not None:
            new_component.layout = new_layout
        if len(new_component.children) == 0:
            new_component = None
        return new_component

    def _sample_new(self, attr_name, min_level, max_level, component):
        self.layout._sample_new(attr_name, min_level, max_level,
                                component.layout)


class Layout(Level):
    """
    Layout is the highest level of the hierarchy with attributes; it 
    has the attributes Number, Position, and Uniformity. Layouts should
    be deep-copied to preserve the state of their attributes. 
    """

    def __init__(self,
                 name,
                 layout_constraints,
                 entity_constraints,
                 orig_layout_constraints=None,
                 orig_entity_constraints=None,
                 sample_new_num_count=None,
                 is_pg=False):
        super(Layout, self).__init__(name,
                                     level=LevelType.LAYOUT,
                                     node_type=NodeType.AND,
                                     is_pg=is_pg)
        self.layout_constraints = layout_constraints
        self.entity_constraints = entity_constraints
        self.number = Number(min_level=layout_constraints.number.min,
                             max_level=layout_constraints.number.max)
        self.position = Position(
            pos_type=layout_constraints.position.position_type,
            pos_list=layout_constraints.position.positions)
        self.uniformity = Uniformity(
            min_level=layout_constraints.uniformity.min,
            max_level=layout_constraints.uniformity.max)
        self.number.sample()
        self.position.sample(self.number.value())
        self.uniformity.sample()
        # store initial layout_constraints and entity_constraints for answer generation
        if orig_layout_constraints is None:
            orig_layout_constraints = copy.deepcopy(self.layout_constraints)
        self.orig_layout_constraints = orig_layout_constraints
        if orig_entity_constraints is None:
            orig_entity_constraints = copy.deepcopy(self.entity_constraints)
        self.orig_entity_constraints = orig_entity_constraints
        if sample_new_num_count is None:
            self.sample_new_num_count = dict()
            max_entities = self.position.values.shape[0]
            for n_entities in range(layout_constraints.number.min + 1,
                                    layout_constraints.number.max + 2):
                self.sample_new_num_count[n_entities] = [
                    comb(max_entities, n_entities), []
                ]
        else:
            self.sample_new_num_count = sample_new_num_count

    @property
    def entities(self):
        return self.children

    def add_new(self, *bboxes):
        """
        Add new entities into this level.
        :param bboxes: tuple of new entities
        """
        name = self.number.value()
        for i, bbox in enumerate(bboxes):
            name += i
            new_entity = copy.deepcopy(self.entities[0])
            new_entity.name = str(name)
            new_entity.bbox = bbox
            if not self.uniformity.value():
                new_entity.resample()
            self.insert(new_entity)

    def resample(self, resample_number=False):
        self._resample(resample_number)

    def __populate(self, receiver, bboxes):
        if self.uniformity.value():
            new_entity = Entity(name=str(0),
                                bbox=bboxes[0],
                                entity_constraints=self.entity_constraints)
            receiver.insert(new_entity)
            for i, bbox in enumerate(bboxes[1:], start=1):
                new_entity = copy.deepcopy(new_entity)
                new_entity.name = str(i)
                new_entity.bbox = bbox
                receiver.insert(new_entity)
        else:
            for i, bbox in enumerate(bboxes):
                receiver.insert(
                    Entity(name=str(i),
                           bbox=bbox,
                           entity_constraints=self.entity_constraints))
        return receiver

    def _sample(self):
        """
        Though Layout is an "and" node, we do not enumerate all possible configurations, but rather
        we treat it as a sampling process such that different configurations are sampled.  After 
        the sampling, the lower level entities are instantiated.
        :returns: a separate sampled layout
        """
        new_layout = copy.deepcopy(self)
        new_layout.is_pg = True
        return self.__populate(new_layout, self.position.value())

    def _resample(self, resample_number: bool):
        """
        Resample each attribute for every entity. This function is called across rows.
        :param resample_number: whether to resample the Number attribute
        """
        if resample_number:
            self.number.sample()
        del self.entities[:]
        self.position.sample(self.number.value())
        self.__populate(self, self.position.value())

    def _update_constraint(self, rules: Rules) -> Optional[Level]:
        """
        Update the constraints of the layout. If one constraint is not satisfied, 
        return None so that this structure is disgarded.
        """
        new_layout_constraints, new_entity_constraints = rule_constraint(
            rules, self.layout_constraints, self.entity_constraints)
        for bounds in [
                new_layout_constraints.number,
                new_layout_constraints.uniformity, new_entity_constraints.type,
                new_entity_constraints.size, new_entity_constraints.color
        ]:
            if bounds.min > bounds.max:
                return None

        return Layout(self.name, new_layout_constraints,
                      new_entity_constraints, self.orig_layout_constraints,
                      self.orig_entity_constraints, self.sample_new_num_count)

    def reset_constraint(self, attribute):
        """
        Propagates changes to the layout constraints to their respective attributes.
        """
        attribute_name = attribute.name.lower()
        attribute = getattr(self, attribute_name)
        constraint = getattr(self.layout_constraints, attribute_name)
        attribute.min_level, attribute.max_level = constraint.min, constraint.max

    def _sample_new(self, attribute_name, min_level, max_level, layout):
        if attribute_name is AttributeType.NUMBER:
            while True:
                value_level = self.number.sample_new(min_level, max_level)
                if layout.sample_new_num_count[value_level][0] == 0:
                    continue
                new_number_value_level = self.number.value(value_level)
                new_value_level = self.position.sample_new(
                    new_number_value_level)
                new_position_set = set(new_value_level)
                if new_position_set not in layout.sample_new_num_count[
                        value_level][1]:
                    layout.sample_new_num_count[value_level][0] -= 1
                    layout.sample_new_num_count[value_level][1].append(
                        new_position_set)
                    break
            self.number.value_level = value_level
            self.position.value_level = new_value_level
            del self.entities[:]
            for i, bbox in enumerate(self.position.value()):
                self.insert(
                    Entity(name=str(i),
                           bbox=bbox,
                           entity_constraints=self.entity_constraints))
        elif attribute_name is AttributeType.POSITION:
            new_value_level = self.position.sample_new(self.number.value())
            layout.position.previous_values.append(new_value_level)
            self.position.value_level = new_value_level
            for bbox, entity in zip(self.position.value(), self.entities):
                entity.bbox = bbox
        elif attribute_name is AttributeType.ANGLE or \
                attribute_name is AttributeType.UNIFORMITY:
            raise ValueError(
                f"unsupported operation on attribute of type: {attribute_name!s}"
            )
        elif attribute_name in AttributeType:
            for entity, orig_entity in zip(self.entities, layout.entities):
                attr_of = lambda e: getattr(e, attribute_name.name.lower())
                attribute, orig_attribute = attr_of(entity), attr_of(
                    orig_entity)
                attribute.value_level = attribute.sample_new(
                    min_level, max_level)
                orig_attribute.previous_values.append(attribute.value_level)
        else:
            raise ValueError("unsupported operation")


class Entity(Level):

    def __init__(self, name, bbox, entity_constraints):
        super(Entity, self).__init__(name,
                                     level=LevelType.ENTITY,
                                     node_type=NodeType.LEAF,
                                     is_pg=True)
        # Attributes
        # Sample each attribute such that the value lies in the admissible range
        # Otherwise, random sample
        self.entity_constraints = entity_constraints
        self.bbox = bbox
        self.type = Type(min_level=entity_constraints.type.min,
                         max_level=entity_constraints.type.max)
        self.size = Size(min_level=entity_constraints.size.min,
                         max_level=entity_constraints.size.max)
        self.color = Color(min_level=entity_constraints.color.min,
                           max_level=entity_constraints.color.max)
        self.angle = Angle(min_level=entity_constraints.angle.min,
                           max_level=entity_constraints.angle.max)
        self._sample()

    def _sample(self):
        self.type.sample()
        self.size.sample()
        self.color.sample()
        self.angle.sample()

    def reset_constraint(self, attribute, min_level, max_level):
        attribute_name = attribute.name.lower()
        constraint = getattr(self.entity_constraints, attribute_name)
        constraint.min, constraint.max = min_level, max_level
        attribute = getattr(self, attribute_name)
        attribute.min_level, attribute.max_level = min_level, max_level

    def resample(self):
        self._sample()