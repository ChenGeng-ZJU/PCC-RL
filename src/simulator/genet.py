import argparse
import os
import time
from typing import Callable, Dict, List, Set, Union

import numpy as np
from bayes_opt import BayesianOptimization
# from bayes_opt.event import DEFAULT_EVENTS, Events
from common.utils import (read_json_file, set_seed, write_json_file)
# from plot_scripts.plot_packet_log import PacketLog

from simulator.aurora import Aurora
from simulator.cubic import Cubic
from simulator.trace import generate_trace

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
    parser.add_argument('--seed', type=int, default=42, help='seed')

    return parser.parse_args()


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


class Genet:
    """Genet implementation with Bayesian Optimization.

    Args
        config_file: configuration file which contains the large ranges of all parameters.
        black_box_function: a function to maximize reward_diff = heuristic_reward - rl_reward.
        heuristic: an object which is the abstraction of a rule-based method.
        rl_method: an object which is the abstraction of a RL-based method.
        seed: random seed.
    """

    def __init__(self, config_file: str, save_dir: str, black_box_function: Callable,
                 heuristic, rl_method, seed: int = 42):
        self.config_file = config_file
        self.cur_config_file = config_file
        self.rand_ranges = RandomizationRanges(config_file)
        self.param_names = self.rand_ranges.get_parameter_names()
        pbounds = self.rand_ranges.get_original_range()

        self.save_dir = save_dir
        self.heuristic = heuristic
        self.rl_method = rl_method
        self.optimizer = BayesianOptimization(
            f=lambda bandwidth, delay, queue, loss, T_s, d_bw, d_delay: black_box_function(
                bandwidth, delay, queue, loss, T_s, d_bw, d_delay,
                heuristic=self.heuristic, rl_method=self.rl_method),
            pbounds=pbounds, random_state=seed)
        # my_observer = BasicObserver()
        # self.optimizer.subscribe(
        #     event=Events.OPTIMIZATION_STEP,
        #     subscriber=my_observer,
        #     callback=None)

    def train(self):
        """Genet trains rl_method."""
        for i in range(12):
            best_param = self.find_best_param()
            print(best_param)
            self.rand_ranges.add_ranges([best_param['params']])
            self.cur_config_file = os.path.join(
                self.save_dir, "bo_"+str(i) + ".json")
            self.rand_ranges.dump(self.cur_config_file)
            self.rl_method.train(self.cur_config_file, 3.6e4, 500)

    def find_best_param(self):
        self.optimizer.maximize(init_points=13, n_iter=2, kappa=20, xi=0.1)
        best_param = self.optimizer.max
        return best_param


def black_box_function(bandwidth, delay, queue, loss, T_s, heuristic, rl_method):
    t_start = time.time()
    trace = generate_trace(duration_range=(30, 30),
                           bandwidth_range=(1, bandwidth),
                           delay_range=(delay, delay),
                           loss_rate_range=(loss, loss),
                           queue_size_range=(queue, queue),
                           T_s_range=(T_s, T_s),
                           delay_noise_range=(0, 0),
                           constant_bw=False)
    print("trace generation used {}s".format(time.time() - t_start))
    t_start = time.time()
    heuristic_reward, _ = heuristic.test(trace)
    print("heuristic used {}s".format(time.time() - t_start))
    t_start = time.time()
    _, reward_list, _, _, _, _, \
        _, _, _, _ = rl_method.test(trace, 'test')
    print("rl_method used {}s".format(time.time() - t_start))
    rl_method_reward = np.mean(reward_list)
    return heuristic_reward - rl_method_reward
    # reward_diffs = []
    # cubic_rewards, cubic_pkt_log = test_cubic_on_trace(trace, "tmp", 20)
    #
    # aurora = Aurora(seed=20, log_dir="tmp", timesteps_per_actorbatch=10,
    #                 pretrained_model_path=MODEL_PATH, delta_scale=1)
    # ts_list, reward_list, loss_list, tput_list, delay_list, send_rate_list, \
    #     action_list, obs_list, mi_list, pkt_log = aurora.test(trace, 'tmp')
    # reward_diffs.append(PacketLog.from_log(cubic_pkt_log).get_reward(None, trace=trace)
    #                     - PacketLog.from_log(pkt_log).get_reward(None, trace=trace))
    # reward_diffs.append(np.mean(np.array(cubic_rewards)
    #                             ) - np.mean(np.array(reward_list)))

    # return np.mean(np.array(reward_diffs))


def main():
    args = parse_args()
    set_seed(args.seed)

    cubic = Cubic(args.save_dir, args.seed)
    aurora = Aurora(seed=args.seed, log_dir=args.save_dir,
                    pretrained_model_path=args.model_path,
                    timesteps_per_actorbatch=10, delta_scale=1)
    genet = Genet(args.config_file, args.save_dir, black_box_function, cubic,
                  aurora)
    genet.train()


if __name__ == "__main__":
    main()