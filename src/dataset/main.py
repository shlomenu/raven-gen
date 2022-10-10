# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np
from tqdm import trange

from panel import Panel
from matrix import Matrix

if __name__ == "__main__":
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

    base_panels = Panel.make_all()

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.fuse:
        if not os.path.exists(os.path.join(args.save_dir, "fuse")):
            os.mkdir(os.path.join(args.save_dir, "fuse"))
        acc = 0
        for k in trange(args.num_samples * len(base_panels)):
            if k < args.num_samples * (1 - args.val - args.test):
                set_name = "train"
            elif k < args.num_samples * (1 - args.test):
                set_name = "val"
            else:
                set_name = "test"

            rpm = Matrix(base_panels[random.choice(list(base_panels.keys()))])
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
            float(acc) / (args.num_samples * len(base_panels))))
    else:
        for key in base_panels.keys():
            if not os.path.exists(os.path.join(args.save_dir, key)):
                os.mkdir(os.path.join(args.save_dir, key))
        for key in base_panels.keys():
            acc = 0
            for k in trange(args.num_samples):
                count_num = k % 10
                if count_num < (10 - args.val - args.test):
                    set_name = "train"
                elif count_num < (10 - args.test):
                    set_name = "val"
                else:
                    set_name = "test"

                rpm = Matrix(base_panels[key])
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
