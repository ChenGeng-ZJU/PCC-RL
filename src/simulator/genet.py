import argparse
import csv
import os.path as osp
import os
import time
from typing import Callable, Dict, List, Set, Union
from pathlib import Path

import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events

from common.utils import (read_json_file, set_seed, write_json_file)
from plot_scripts.plot_packet_log import PacketLog
from simulator.aurora import Aurora
# from simulator.cubic import Cubic
from simulator.network_simulator.cubic import Cubic
from simulator.network_simulator.bbr import BBR
from simulator.trace import generate_trace
from simulator.network_simulator.bbr import BBR
from simulator.compare_syn_real_traces import compare

MODEL_PATH = ""


def parse_args():
    """Parse arguments from the command line."""
    parser = argparse.ArgumentParser("BO training in simulator.")
    parser.add_argument('--save-dir', type=str, required=True,
                        help="directory to save testing and intermediate results.")
    parser.add_argument('--model-path', type=str, default=None,
                        help="path to Aurora model to start from.")
    parser.add_argument("--config-file", type=str,
                        help="Path to configuration file.")
    parser.add_argument("--bo-rounds", type=int, default=30,
                        help="Rounds of BO.")
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--bbr', action="store_true")

    return parser.parse_args()


# TODO: add more new configuraiton adding policies
class RandomizationRanges:
    """Manage randomization ranges used in GENET training."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        if filename and os.path.exists(filename):
            self.rand_ranges = read_json_file(filename)
            assert isinstance(self.rand_ranges, List) and len(
                self.rand_ranges) >= 1, "rand_ranges object should be a list with length at least 1."
            weight_sum = 0
            for rand_range in self.rand_ranges:
                weight_sum += rand_range['weight']
            assert weight_sum == 1.0, "Weight sum should be 1."
            self.parameters = set(self.rand_ranges[0].keys())
            self.parameters.remove('weight')
        else:
            self.rand_ranges = []
            self.parameters = set()

    def add_ranges(self, range_maps: List[Dict[str, Union[List[float], float]]],
                   prob: float = 0.3) -> None:
        """Add a list of ranges into the randomization ranges.

        The sum of weights of newlly added ranges is prob.
        """
        for rand_range in self.rand_ranges:
            rand_range['weight'] *= (1 - prob)
        if self.rand_ranges:
            weight = prob / len(range_maps)
        else:
            weight = 1 / len(range_maps)
        for range_map in range_maps:
            range_map_to_add = dict()
            for param in self.parameters:
                if param == 'duration':
                    range_map_to_add[param] = [30, 30]
                    continue

                assert param in range_map, "range_map does not contain '{}'".format(
                    param)
                range_map_to_add[param] = [range_map[param], range_map[param]]
            range_map_to_add['weight'] = weight
            self.rand_ranges.append(range_map_to_add)

    def get_original_range(self) -> Dict[str, List[float]]:
        start_range = dict()
        for param_name in self.parameters:
            start_range[param_name] = self.rand_ranges[0][param_name]
        return start_range

    def get_ranges(self) -> List[Dict[str, List[float]]]:
        return self.rand_ranges

    def get_parameter_names(self) -> Set[str]:
        return self.parameters

    def dump(self, filename: str) -> None:
        write_json_file(filename, self.rand_ranges)


# class BasicObserver:
#     def update(self, event, instance):
#         """Does whatever you want with the event and `BayesianOptimization` instance."""
#         print("Event `{}` was observed".format(event))

def find_best_model(model_path, boit=0, omit_start=False):
    best_step = -1
    best_reward = -1
    flag = False
    with open(osp.join(model_path, "validation_log_{}.csv".format(boit)), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            # breakpoint()
            step, _, reward, _, _, _, _, _ = row[0].split('\t')
            if step=="n_calls":
                continue
            if omit_start:
                if not flag:
                    flag = True
                    continue
            reward = float(reward)
            step = int(float(step))
            # print(step)
            if best_step == -1:
                best_step, best_reward = step, reward
            elif reward > best_reward:
                best_step, best_reward = step, reward
    return osp.join(model_path, "bo_{}_model_step_{}.ckpt".format(boit, best_step)), best_reward


class Genet:
    """Genet implementation with Bayesian Optimization.

    Args
        config_file: configuration file which contains the large ranges of all parameters.
        black_box_function: a function to maximize reward_diff = heuristic_reward - rl_reward.
        heuristic: an object which is the abstraction of a rule-based method.
        rl_method: an object which is the abstraction of a RL-based method.
        seed: random seed.
    """

    def __init__(self, config_file: str, save_dir: str,
                 black_box_function: Callable, heuristic, rl_method,
                 seed: int = 42):
        self.black_box_function = black_box_function
        self.seed = seed
        self.config_file = config_file
        self.cur_config_file = config_file
        self.rand_ranges = RandomizationRanges(config_file)
        self.param_names = self.rand_ranges.get_parameter_names()
        self.pbounds = self.rand_ranges.get_original_range()
        self.seed = seed

        self.save_dir = save_dir
        self.heuristic = heuristic
        self.rl_method = rl_method
        # my_observer = BasicObserver()
        # self.optimizer.subscribe(
        #     event=Events.OPTIMIZATION_STEP,
        #     subscriber=my_observer,
        #     callback=None)
    
    def update_rl_model(self, boit):
        print("updating best model...")
        best_model, best_reward = find_best_model(self.save_dir, boit)
        print("using model: {}, which has reward {}".format(best_model, best_reward))
        self.rl_method = Aurora(seed=self.seed, log_dir=self.save_dir,
                                pretrained_model_path=best_model,
                                timesteps_per_actorbatch=7200, delta_scale=1)

    def train(self, heu = 'cubic'):
        """Genet trains rl_method."""
        from time import time
        for i in range(120):
            # self.seed += 1
            print("start finding param")
            t1 = time()
            best_param = self.find_best_param()
            t2 = time()
            print("end finding parameter, time elpased: {}".format(t2-t1))
            print(best_param)
            self.rand_ranges.add_ranges([best_param['params']])
            self.cur_config_file = os.path.join(
                self.save_dir, "bo_"+str(i) + ".json")
            self.rand_ranges.dump(self.cur_config_file)
            self.rl_method.train(i, self.cur_config_file, 3.6e4, 500)
            # self.rl_method.train(i, self.cur_config_file, 3.6e5, 500)
            # self.rl_method.train(i, self.cur_config_file, 10000, 500)
            # self.rl_method.train(self.cur_config_file, 800, 500)
            t3 = time()
            # self.update_rl_model(i)
            print("finish a training, time elapsed = {}".format(t3 - t2))
            print("Start Ploting...")
            # best_model, best_reward = find_best_model(self.save_dir, i)
            name = self.save_dir.split('/')[-1] + "_bo_{}".format(i+1)
            model_path = osp.join(self.save_dir, "bo_{}_model_step_{}.ckpt".format(i, 36000))
            # model_path = best_model
            compare(model_path, name)
            print("End Ploting...")
            

    def find_best_param(self):
        optimizer = BayesianOptimization(
            f=lambda bandwidth, delay, queue, loss, T_s: black_box_function(
                bandwidth, delay, queue, loss, T_s,
                heuristic=self.heuristic, rl_method=self.rl_method),
            pbounds=self.pbounds, random_state=self.seed)
        optimizer.maximize(init_points=13, n_iter=2, kappa=20, xi=0.1)
        best_param = optimizer.max
        return best_param

SAVEDIR=""

def black_box_function(bandwidth: float, delay: float, queue: Union[int, float],
                       loss: float, T_s: float,
                       heuristic, rl_method) -> float:
    queue = int(queue)
    t_start = time.time()
    delay_noise = 0
    trace = generate_trace(duration_range=(30, 30),
                           bandwidth_range=(1, bandwidth),
                           delay_range=(delay, delay),
                           loss_rate_range=(loss, loss),
                           queue_size_range=(queue, queue),
                           T_s_range=(T_s, T_s),
                           delay_noise_range=(delay_noise, delay_noise),
                           constant_bw=False)
    trace.dump(osp.join(SAVEDIR, "trace.json"))
    heuristic_mi_level_reward, heuristic_pkt_level_reward = heuristic.test(
        trace, rl_method.log_dir)
    print("trace generation used {}s".format(time.time() - t_start))
    t_start = time.time()
    _, reward_list, _, _, _, _, _, _, _, rl_pkt_log = rl_method.test(
        trace, rl_method.log_dir)
    # print("rl_method used {}s".format(time.time() - t_start))
    rl_mi_level_reward = np.mean(reward_list)

    rl_pkt_level_reward = PacketLog.from_log(rl_pkt_log).get_reward("", trace)
    return heuristic_pkt_level_reward - rl_pkt_level_reward
    # return heuristic_mi_level_reward - rl_mi_level_reward


def main():
    # print(find_best_model('/data/gengchen/PCC-RL/data/udr-large-genet-081700', 0))
    args = parse_args()
    set_seed(args.seed)
    Path(args.save_dir).mkdir(exist_ok=True, parents=True)
    SAVEDIR = args.save_dir

    cubic = Cubic(args.seed)
    bbr = BBR(args.save_dir)
    # pre_model, _ = find_best_model(args.model_path)
    pre_model = args.model_path
    print(pre_model)
    aurora = Aurora(seed=args.seed, log_dir=args.save_dir,
                    pretrained_model_path=pre_model,
                    timesteps_per_actorbatch=7200, delta_scale=1)
    name = args.save_dir.split('/')[-1] + "_BeforeBO"
    if not args.bbr:
        compare(pre_model, name)
        genet = Genet(args.config_file, args.save_dir, black_box_function, cubic, aurora)
        genet.train()
    else:
        compare(pre_model, name)
        print("using bbr")
        genet = Genet(args.config_file, args.save_dir, black_box_function, bbr, aurora)
        genet.train("bbr")


if __name__ == "__main__":
    main()
