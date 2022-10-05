# -*- coding: utf-8 -*-

import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from configuration import (META_STRUCTURE_FORMAT, DEFAULT_WIDTH, IMAGE_SIZE,
                           Shape, Point, LevelType, RuleType, AttributeType)
from rendering import render_entity


def serialize_level(level):
    assert level.is_pg
    n_tree = ""
    if level.level is LevelType.LAYOUT:
        n_tree += level.name + "./"
    else:
        n_tree += level.name + "."
        for child in level.children:
            n_tree += serialize_level(child)
            n_tree += "."
        n_tree += "/"
    return n_tree


def serialize_panel(panel):
    """
    META_STRUCTURE_FORMAT is provided by configuration.py
    """
    n_tree = serialize_level(panel)
    meta_structure = np.zeros(len(META_STRUCTURE_FORMAT), np.uint8)
    split = n_tree.split(".")
    for node in split:
        try:
            node_index = META_STRUCTURE_FORMAT.index(node)
            meta_structure[node_index] = 1
        except ValueError:
            continue
    return split, meta_structure


def serialize_rules(rule_groups):
    """
    Meta matrix format:

        ["Constant", "Progression", "Arithmetic", "Distribute_Three", 
         "Number", "Position", "Type", "Size", "Color"]
    
    """
    meta_matrix = np.zeros((8, 9), np.uint8)
    counter = 0
    for rule_group in rule_groups:
        for rule in rule_group:
            if rule.name is RuleType.CONSTANT:
                meta_matrix[counter, 0] = 1
            elif rule.name is RuleType.PROGRESSION:
                meta_matrix[counter, 1] = 1
            elif rule.name is RuleType.ARITHMETIC:
                meta_matrix[counter, 2] = 1
            else:
                meta_matrix[counter, 3] = 1
            if rule.attr is AttributeType.NUMBER_OR_POSITION:
                meta_matrix[counter, 4] = 1
                meta_matrix[counter, 5] = 1
            elif rule.attr is AttributeType.NUMBER:
                meta_matrix[counter, 4] = 1
            elif rule.attr is AttributeType.POSITION:
                meta_matrix[counter, 5] = 1
            elif rule.attr is AttributeType.TYPE:
                meta_matrix[counter, 6] = 1
            elif rule.attr is AttributeType.SIZE:
                meta_matrix[counter, 7] = 1
            else:
                meta_matrix[counter, 8] = 1
            counter += 1
    return meta_matrix, np.bitwise_or.reduce(meta_matrix)


class Bunch:

    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def get_real_bbox(entity):
    entity_type = entity.type.value()
    assert entity_type is not Shape.NONE
    center = Point(x=int(entity.bbox.x_c * IMAGE_SIZE),
                   y=int(entity.bbox.y_c * IMAGE_SIZE))
    M = cv2.getRotationMatrix2D(center, entity.angle.value(), 1)
    unit = min(entity.bbox.max_w, entity.bbox.max_h) * IMAGE_SIZE / 2
    delta = DEFAULT_WIDTH * 1.5 // IMAGE_SIZE
    if entity_type is Shape.CIRCLE:
        radius = unit * entity.size.value()
        real_bbox = [
            center.x * 1.0 // IMAGE_SIZE, center.y * 1.0 // IMAGE_SIZE,
            2 * radius // IMAGE_SIZE + delta, 2 * radius // IMAGE_SIZE + delta
        ]
    else:
        if entity_type is Shape.SQUARE:
            dl = int(unit / 2 * np.sqrt(2) * entity.size.value())
        else:
            dl = int(unit * entity.size.value())
        if entity_type is Shape.TRIANGLE:
            homo_pts = np.array([[center.y, center.x - dl, 1],
                                 [
                                     center.y + int(dl / 2.0 * np.sqrt(3)),
                                     center.x + int(dl / 2.0), 1
                                 ],
                                 [
                                     center.y - int(dl / 2.0 * np.sqrt(3)),
                                     center.x + int(dl / 2.0), 1
                                 ]], np.int32)
        if entity_type is Shape.SQUARE:
            homo_pts = np.array([[center.y - dl, center.x - dl, 1],
                                 [center.y - dl, center.x + dl, 1],
                                 [center.y + dl, center.x + dl, 1],
                                 [center.y + dl, center.x - dl, 1]], np.int32)
        if entity_type is Shape.PENTAGON:
            homo_pts = np.array([[center.y, center.x - dl, 1],
                                 [
                                     center.y - int(dl * np.cos(np.pi / 10)),
                                     center.x - int(dl * np.sin(np.pi / 10)), 1
                                 ],
                                 [
                                     center.y - int(dl * np.sin(np.pi / 5)),
                                     center.x + int(dl * np.cos(np.pi / 5)), 1
                                 ],
                                 [
                                     center.y + int(dl * np.sin(np.pi / 5)),
                                     center.x + int(dl * np.cos(np.pi / 5)), 1
                                 ],
                                 [
                                     center.y + int(dl * np.cos(np.pi / 10)),
                                     center.x - int(dl * np.sin(np.pi / 10)), 1
                                 ]], np.int32)
        if entity_type is Shape.HEXAGON:
            homo_pts = np.array([[center.y, center.x - dl, 1],
                                 [
                                     center.y - int(dl / 2.0 * np.sqrt(3)),
                                     center.x - int(dl / 2.0), 1
                                 ],
                                 [
                                     center.y - int(dl / 2.0 * np.sqrt(3)),
                                     center.x + int(dl / 2.0), 1
                                 ], [center.y, center.x + dl, 1],
                                 [
                                     center.y + int(dl / 2.0 * np.sqrt(3)),
                                     center.x + int(dl / 2.0), 1
                                 ],
                                 [
                                     center.y + int(dl / 2.0 * np.sqrt(3)),
                                     center.x - int(dl / 2.0), 1
                                 ]], np.int32)
        rotated = np.dot(M, homo_pts.T)
        min_x = min(rotated[1, :]) / IMAGE_SIZE
        max_x = max(rotated[1, :]) / IMAGE_SIZE
        min_y = min(rotated[0, :]) / IMAGE_SIZE
        max_y = max(rotated[0, :]) / IMAGE_SIZE
        real_bbox = [(min_x + max_x) // 2, (min_y + max_y) // 2,
                     max_x - min_x + delta, max_y - min_y + delta]
    return list(np.round(real_bbox, 4))


def get_mask(entity):
    dummy_entity = Bunch()
    dummy_entity.bbox = entity.bbox
    dummy_entity.type = Bunch(value=entity.type.value())
    dummy_entity.size = Bunch(value=entity.size.value())
    dummy_entity.color = Bunch(value=0)
    dummy_entity.angle = Bunch(value=entity.angle.value())
    return render_entity(dummy_entity) // 255


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    :param img: ndarray containing 1s in masked regions and 0s in background regions
    :returns: string formatted list of run lengths in alternating (start, length) arrangement
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return str(runs.tolist()).replace(" ", "")


def rle_decode(mask_rle, shape):
    '''
    :param mask_rle: string formatted list of alternating starts and run lengths (from rle_encode)
    :param shape: (height, width) of array to return 
    :returns: ndarray containing 1s in maked regions and 0 in background regions
    '''
    s = mask_rle[1:-1].split(",")
    starts = np.asarray(s[::2], dtype=int) - 1
    lengths = np.asarray(s[1::2], dtype=int)
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


def dom_problem(instances, rule_groups):
    data = ET.Element("Data")
    panels = ET.SubElement(data, "Panels")
    for i, panel in enumerate(instances):
        panel_i = ET.SubElement(panels, "Panel")
        struct = panel.structure
        struct_i = ET.SubElement(panel_i, "Struct")
        struct_i.set("name", struct.name)
        for j, component in enumerate(struct.components):
            component_j = ET.SubElement(struct_i, "Component")
            component_j.set("id", str(j))
            component_j.set("name", component.name)
            layout = component.layout
            layout_k = ET.SubElement(component_j, "Layout")
            layout_k.set("name", layout.name)
            layout_k.set("Number", str(layout.number.value_level))
            layout_k.set("Position", json.dumps(layout.position.values))
            layout_k.set("Uniformity", str(layout.uniformity.value_level))
            for l, entity in enumerate(layout.entities):
                entity_l = ET.SubElement(layout_k, "Entity")
                entity_l.set("bbox", json.dumps(entity.bbox))
                entity_l.set(
                    "real_bbox",
                    json.dumps(
                        get_real_bbox(entity.bbox, entity.type.value(),
                                      entity.size.value(),
                                      entity.angle.value())))
                entity_l.set("mask", rle_encode(get_mask(entity.bbox)))
                entity_l.set("Type", str(entity.type.value_level))
                entity_l.set("Size", str(entity.size.value_level))
                entity_l.set("Color", str(entity.color.value_level))
                entity_l.set("Angle", str(entity.angle.value_level))
    rules = ET.SubElement(data, "Rules")
    for i, rule_group in enumerate(rule_groups):
        rule_group_i = ET.SubElement(rules, "Rule_Group")
        rule_group_i.set("id", str(i))
        for rule in rule_group:
            rule_j = ET.SubElement(rule_group_i, "Rule")
            rule_j.set("name", rule.name)
            rule_j.set("attr", rule.attr)
    return ET.tostring(data)
