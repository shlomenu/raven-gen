# -*- coding: utf-8 -*-

import argparse
import copy
import os
import random

import numpy as np
from tqdm import trange

from build_panel import (center_single, distribute_four, distribute_nine,
                         in_center_single_out_center_single,
                         in_distribute_four_out_center_single,
                         left_center_single_right_center_single,
                         up_center_single_down_center_single)
from configuration import IMAGE_SIZE
from rendering import render_panel
from sampling import sample_attr, sample_attr_avail, sample_rules
from serialization import dom_problem, serialize_panel, serialize_rules
from solver import solve


def make_row(start, rules, resample=False):
    col_0, row = copy.deepcopy(start), None
    if resample:
        col_0.resample(True)
    for c, component_rules in enumerate(rules):
        col_1 = component_rules.number_position.apply(col_0)
        col_2 = component_rules.number_position.apply(col_1)
        for rule in component_rules:
            col_1 = rule.apply(col_0, col_1)
        for rule in component_rules:
            col_2 = rule.apply(col_1, col_2)
        if c == 0:
            row = [col_0, col_1, col_2]
        else:
            row[1].structure.components[c] = col_1.structure.components[c]
            row[2].structure.components[c] = col_2.structure.components[c]
    return row


def make_problem(panel):
    while True:
        rules = sample_rules()
        new_panel = panel.prune(rules)
        if new_panel is not None:
            break

    start = new_panel.sample()

    panel_0_0, panel_0_1, panel_0_2 = make_row(start, rules)
    panel_1_0, panel_1_1, panel_1_2 = make_row(start, rules, resample=True)
    panel_2_0, panel_2_1, panel_2_2 = make_row(start, rules, resample=True)

    imgs = [
        render_panel(panel_0_0),
        render_panel(panel_0_1),
        render_panel(panel_0_2),
        render_panel(panel_1_0),
        render_panel(panel_1_1),
        render_panel(panel_1_2),
        render_panel(panel_2_0),
        render_panel(panel_2_1),
        np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    ]
    context = [
        panel_0_0, panel_0_1, panel_0_2, panel_1_0, panel_1_1, panel_1_2,
        panel_2_0, panel_2_1
    ]

    modifiable_attr = sample_attr_avail(rules, panel_2_2)
    answer_aot = copy.deepcopy(panel_2_2)
    candidates = [answer_aot]
    for _ in range(7):
        component_idx, attr_name, min_level, max_level = sample_attr(
            modifiable_attr)
        answer_j = copy.deepcopy(answer_aot)
        answer_j.sample_new(component_idx, attr_name, min_level, max_level,
                            answer_aot)
        candidates.append(answer_j)

    random.shuffle(candidates)
    answers = []
    for candidate in candidates:
        answers.append(render_panel(candidate))
    # imsave(generate_matrix_answer(imgs + answers), "./experiments/fuse/{}.jpg".format(k))
    # imsave(generate_matrix_answer(imgs + answers), "./experiments/{}/{}.jpg".format(key, k))

    meta_matrix, meta_target = serialize_rules(rules)
    structure, meta_structure = serialize_panel(start)
    return (imgs[0:8] + answers, candidates.index(answer_aot),
            solve(rules, context, candidates), meta_matrix, meta_target,
            structure, meta_structure, dom_problem(context + candidates,
                                                   rules))


def main():
    parser = argparse.ArgumentParser(description="parser for RAVEN")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20000,
        help="number of samples for each component configuration")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="~/Datasets/",
        help="path to folder where the generated dataset will be saved.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed for dataset generation")
    parser.add_argument("--fuse",
                        type=int,
                        default=0,
                        help="whether to fuse different configurations")
    parser.add_argument("--val",
                        type=float,
                        default=2,
                        help="the proportion of the size of validation set")
    parser.add_argument("--test",
                        type=float,
                        default=2,
                        help="the proportion of the size of test set")
    args = parser.parse_args()

    configs = {
        "center_single":
        center_single(),
        "distribute_four":
        distribute_four(),
        "distribute_nine":
        distribute_nine(),
        "left_center_single_right_center_single":
        left_center_single_right_center_single(),
        "up_center_single_down_center_single":
        up_center_single_down_center_single(),
        "in_center_single_out_center_single":
        in_center_single_out_center_single(),
        "in_distribute_four_out_center_single":
        in_distribute_four_out_center_single()
    }

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.fuse:
        if not os.path.exists(os.path.join(args.save_dir, "fuse")):
            os.mkdir(os.path.join(args.save_dir, "fuse"))
        acc = 0
        for k in trange(args.num_samples * len(configs)):
            if k < args.num_samples * (1 - args.val - args.test):
                set_name = "train"
            elif k < args.num_samples * (1 - args.test):
                set_name = "val"
            else:
                set_name = "test"

            tree_name = random.choice(configs.keys())
            root = configs[tree_name]
            image, target, predicted, meta_matrix, meta_target, structure, meta_structure, dom = make_problem(
                root)
            np.savez("{}/RAVEN_{}_{}.npz".format(args.save_dir, k, set_name),
                     image=image,
                     target=target,
                     predict=predicted,
                     meta_matrix=meta_matrix,
                     meta_target=meta_target,
                     structure=structure,
                     meta_structure=meta_structure)
            with open("{}/RAVEN_{}_{}.xml".format(args.save_dir, k, set_name),
                      "w") as f:
                f.write(dom)
            if target == predicted:
                acc += 1
        print("Accuracy: {}".format(
            float(acc) / (args.num_samples * len(configs))))
    else:
        for key in configs.keys():
            if not os.path.exists(os.path.join(args.save_dir, key)):
                os.mkdir(os.path.join(args.save_dir, key))
        for key in configs.keys():
            acc = 0
            for k in trange(args.num_samples):
                count_num = k % 10
                if count_num < (10 - args.val - args.test):
                    set_name = "train"
                elif count_num < (10 - args.test):
                    set_name = "val"
                else:
                    set_name = "test"

                root = configs[key]
                image, target, predicted, meta_matrix, meta_target, structure, meta_structure, dom = make_problem(
                    root)
                np.savez("{}/{}/RAVEN_{}_{}.npz".format(
                    args.save_dir, key, k, set_name),
                         image=image,
                         target=target,
                         predict=predicted,
                         meta_matrix=meta_matrix,
                         meta_target=meta_target,
                         structure=structure,
                         meta_structure=meta_structure)
                with open(
                        "{}/{}/RAVEN_{}_{}.xml".format(args.save_dir, key, k,
                                                       set_name), "w") as f:
                    f.write(dom)
                if target == predicted:
                    acc += 1
            print("Accuracy of {}: {}".format(key,
                                              float(acc) / args.num_samples))


if __name__ == "__main__":
    main()
