import csv
import os
import pandas as pd
import subprocess
import glob
from pathlib import Path
import os.path as osp

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from common.utils import set_seed, read_json_file, compute_std_of_mean
from simulator.aurora import Aurora
from simulator.evaluate_cubic import test_on_traces as test_cubic_on_traces
from simulator.evaluate_bbr import test_on_traces as test_bbr_on_traces
from simulator.trace import generate_trace, Trace
from tqdm import tqdm
from pathlib import Path

def compare_metric(model_path, name, save_dir, vals2test, key):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    f = open(os.path.join(save_dir, 'syn_vs_real_traces_{}.csv'.format(name)), 'w', 1)
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['syn_reward', 'syn_reward_err', 'cubic_syn_reward', 'cubic_syn_reward_err', 'bbr_syn_reward', 'bbr_syn_reward_err'])
    aurora_rewards = []
    aurora_errors = []
    cubic1_rewards = []
    cubic1_errors = []
    bbr1_rewards = []
    bbr1_errors = []
    duration_range=(10, 10)
    bw_range=(1, 1)
    delay_range=(100, 100)
    lr_range=(0, 0)
    queue_size_range=(10, 10)
    T_s_range=(0, 0)
    delay_noise_range=(0, 0)
    for bwi, bw in enumerate(tqdm(vals2test[key], desc=key)):
        val = bw
        if key == 'bandwidth':
            bw_range = (val, val)
        elif key == 'delay':
            delay_range = (val, val)
        elif key == 'loss':
            lr_range = (val, val)
        elif key == "queue":
            queue_size_range = (val, val)
        elif key == "T_s":
            T_s_range = (val, val)
        elif key == "delay_noise":
            delay_noise_range = (val, val)
        else:
            raise NotImplementedError
        syn_traces = [generate_trace(duration_range=duration_range,
                                bandwidth_range=bw_range,
                                delay_range=delay_range,
                                loss_rate_range=lr_range,
                                queue_size_range=queue_size_range,
                                T_s_range=T_s_range,
                                delay_noise_range=delay_noise_range,
                                constant_bw=False) for _ in range(1)]
        tmpsvp = osp.join('tmp', '{}_{}_{}'.format(name, key, bwi))
        Path(tmpsvp).mkdir(exist_ok=True, parents=True)
        syn_traces[-1].dump(osp.join(tmpsvp, "trace.json"))

        aurora_udr_big = Aurora(seed=20, log_dir=tmpsvp, timesteps_per_actorbatch=10,
                                pretrained_model_path=model_path, delta_scale=1)

        cubic_rewards, _ = test_cubic_on_traces(syn_traces, [tmpsvp]*len(syn_traces), seed=20)
        bbr_rewards, _ = test_bbr_on_traces(syn_traces, [tmpsvp]*len(syn_traces), seed=20)

        results, _ = aurora_udr_big.test_on_traces(
                syn_traces, [tmpsvp]*len(syn_traces))
        # print(np.mean(np.array(cubic_rewards), axis=0))
        avg_cubic_rewards = np.mean([np.mean(r) for r in cubic_rewards])
        avg_cubic_rewards_errs = compute_std_of_mean([np.mean(r) for r in cubic_rewards])

        avg_bbr_rewards = np.mean([np.mean(r) for r in bbr_rewards])
        avg_bbr_rewards_errs = compute_std_of_mean([np.mean(r) for r in bbr_rewards])

        udr_big_rewards = np.array([np.mean([row[1] for row in result]) for result in results])
        avg_udr_big_rewards = np.mean(udr_big_rewards)
        avg_udr_big_rewards_errs = compute_std_of_mean([np.mean(r) for r in udr_big_rewards])

        writer.writerow([avg_udr_big_rewards, avg_udr_big_rewards_errs,
                        avg_cubic_rewards, avg_cubic_rewards_errs,
                        avg_bbr_rewards, avg_bbr_rewards_errs])
        aurora_rewards.append(avg_udr_big_rewards)
        aurora_errors.append(avg_udr_big_rewards_errs)
        cubic1_rewards.append(avg_cubic_rewards)
        cubic1_errors.append(avg_cubic_rewards_errs)
        bbr1_rewards.append(avg_bbr_rewards)
        bbr1_errors.append(avg_bbr_rewards_errs)

    width = 0.7
    fig, ax = plt.subplots()
    ax.bar([i*4 for i in range(len(aurora_rewards))], aurora_rewards, width, yerr=aurora_errors, color="C4", alpha = 1, label="DRL-based Policy")
    ax.bar([i*4 + 1 for i in range(len(cubic1_rewards))], cubic1_rewards, width, yerr=cubic1_errors, color="C0", alpha = 1, label="Rule-based Policy({})".format('cubic'))
    ax.bar([i*4 + 2 for i in range(len(bbr1_rewards))], bbr1_rewards, width, yerr=bbr1_errors, color="C1", alpha = 1, label="Rule-based Policy({})".format('bbr'))
    ax.set_xticks([i*4+1 for i in range(len(aurora_rewards))])
    ax.set_xticklabels(vals2test[key])
    ax.set_ylim(-500, 800)
    plt.xlabel(key)
    plt.ylabel("Rewards")
    plt.title(name)
    ax.legend()
    plt.savefig(os.path.join(save_dir, 'syn_vs_real_traces_{}.jpg'.format(name)), bbox_inches='tight')
    plt.close()
    # plt.show()
    f.close()
    return np.array(aurora_rewards).mean()

def compare(model_path, name):
    plt.style.use('seaborn-deep')
    # plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 28
    plt.rcParams['axes.labelsize'] = 38
    plt.rcParams['legend.fontsize'] = 24
    plt.rcParams["figure.figsize"] = (11,6)

    rpath = "/data/gengchen/PCC-RL"
    MODEL_PATH = model_path
    SAVE_DIR = osp.join(rpath, 'figs')
    print(MODEL_PATH)

    set_seed(28)

    vals2test = {
        "bandwidth": [1, 2, 3, 4, 5, 6],
        "delay": [5, 50, 100, 150, 200],
        "loss": [0, 0.01, 0.02, 0.03, 0.04, 0.05],
        "queue": [2, 10, 50, 100, 150, 200],
        "T_s": [0, 1, 2, 3, 4, 5, 6],
        "delay_noise": [0, 20, 40, 60, 80, 100],
    }

    reward = []
    basename = '_'.join(name.split('_')[:-1])
    for key in vals2test:
        reward.append(compare_metric(MODEL_PATH, name + "_" + key, osp.join(SAVE_DIR, basename, key), vals2test, key))
    return np.array(reward).mean()


if __name__ == "__main__":
    rpath = "/data/gengchen/PCC-RL"
    name = "udr-large-genet-082010"
    # compare(osp.join(rpath, "data", "udr-large-genet-082001", "bo_10_model_step_72000.ckpt"), "test-082001-10-72000")
    # reward = []
    # for i in range(7200, 302400, 7200):
    #     model = osp.join(rpath, "data", "udr-large-081815", "bo_0_model_step_{}.ckpt".format(i))
    #     reward.append(compare(model, "udr-large-081815-step-{}".format(i)))
    # print(reward)
    # reward = np.array(reward)
    # np.save(osp.join(rpath, "figs", "result1.npy"), reward)

    # for i in range(1, 20):
    #     # model, reward = ge.find_best_model(osp.join(rpath, 'data/udr-large-genet-081801'), i)
    #     # print(model)
    #     name = 'udr-large-genet-test-bo{}'.format(i) 
    #     # compare(model, name)
    for i in range(80):
        compare(
            osp.join(rpath, "data", name, "bo_{}_model_step_36000.ckpt".format(i)),
            name+"_fix_bo_{}".format(i)
        )