import csv
from logging import error
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
from plot_scripts.plot_time_series import plot_time

def test_with_param(rang, name, model_path, trace_cnt = 50, plot_graph = False):
    # trace_cnt = 50
    bw = rang['bandwidth_upper_bound'][0]
    delay = rang['delay'][0]
    loss = rang['loss'][0]
    queue = rang['queue'][0]
    ts = rang['T_s'][0]
    syn_traces = [generate_trace(duration_range=(30, 30),
                            bandwidth_lower_bound_range=(0.1, 0.1),
                            bandwidth_upper_bound_range=(bw, bw),
                            delay_range=(delay, delay),
                            loss_rate_range=(loss, loss),
                            queue_size_range=(queue, queue),
                            T_s_range=(ts, ts),
                            delay_noise_range=(0., 0.),
                            constant_bw=False,
                            seed=x*123+5) for x in range(trace_cnt)]
    # tmpsvp = osp.join('tmp', '{}_{}_{}'.format(name, key, bwi))
    tmpsvps = [osp.join('log', name, str(tracei % 3)) for tracei in range(trace_cnt)]
    for i in range(trace_cnt):
        Path(tmpsvps[i]).mkdir(exist_ok=True, parents=True)
        syn_traces[i].dump(osp.join(tmpsvps[i], "trace.json"))
    # Path(tmpsvp).mkdir(exist_ok=True, parents=True)

    aurora_udr_big = Aurora(seed=20, log_dir=tmpsvps[0], timesteps_per_actorbatch=10, pretrained_model_path=model_path, delta_scale=1)

    # cubic_rewards, _ = test_cubic_on_traces(syn_traces, tmpsvps, seed=20)
    bbr_rewards, _ = test_bbr_on_traces(syn_traces, tmpsvps, seed=20)
    results, _ = aurora_udr_big.test_on_traces(syn_traces, tmpsvps)

    # plot scripts
    if plot_graph:
        for i in range(0, min(3, trace_cnt)):
            di = tmpsvps[i]
            # __import__("ipdb").set_trace()
            for method in ['aurora', 'cubic', 'bbr']:
                plot_time([osp.join(di, '{}_simulation_log.csv'.format(method))], 
                            osp.join(di, "trace.json"),
                            di)
    
    avg_bbr_rewards = np.mean([np.mean(r) for r in bbr_rewards])
    avg_bbr_rewards_errs = compute_std_of_mean([np.mean(r) for r in bbr_rewards])

    udr_big_rewards = np.array([np.mean([row[1] for row in result]) for result in results])
    avg_udr_big_rewards = np.mean(udr_big_rewards)
    avg_udr_big_rewards_errs = compute_std_of_mean([np.mean(r) for r in udr_big_rewards])
    
    return avg_udr_big_rewards - avg_bbr_rewards, avg_udr_big_rewards, avg_bbr_rewards, avg_udr_big_rewards_errs, avg_bbr_rewards_errs, udr_big_rewards, bbr_rewards


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
    duration_range=(30, 30)
    bw_range=(0.1, 100)
    # TODO
    delay_range=(100, 100)
    lr_range=(0, 0)
    queue_size_range=(1.6, 1.6)
    T_s_range=(3, 3)
    delay_noise_range=(0, 0)
    for bwi, bw in enumerate(tqdm(vals2test[key], desc=key)):
        # __import__("ipdb").set_trace()
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
        trace_cnt = 10
        syn_traces = [generate_trace(duration_range=duration_range,
                                bandwidth_lower_bound_range=(0.1, 0.1),
                                bandwidth_upper_bound_range=bw_range,
                                delay_range=delay_range,
                                loss_rate_range=lr_range,
                                queue_size_range=queue_size_range,
                                T_s_range=T_s_range,
                                delay_noise_range=delay_noise_range,
                                constant_bw=False,
                                seed=x*123) for x in range(trace_cnt)]
        # tmpsvp = osp.join('tmp', '{}_{}_{}'.format(name, key, bwi))
        tmpsvps = [osp.join('log', name, key, str(bw), str(tracei)) for tracei in range(trace_cnt)]
        for i in range(trace_cnt):
            Path(tmpsvps[i]).mkdir(exist_ok=True, parents=True)
            syn_traces[i].dump(osp.join(tmpsvps[i], "trace.json"))
        # Path(tmpsvp).mkdir(exist_ok=True, parents=True)

        aurora_udr_big = Aurora(seed=20, log_dir=tmpsvps[0], timesteps_per_actorbatch=10,
                                pretrained_model_path=model_path, delta_scale=1)

        cubic_rewards, _ = test_cubic_on_traces(syn_traces, tmpsvps, seed=20)
        bbr_rewards, _ = test_bbr_on_traces(syn_traces, tmpsvps, seed=20)
        results, _ = aurora_udr_big.test_on_traces(syn_traces, tmpsvps)

        # plot scripts
        for i in range(0, trace_cnt, 5):
            di = tmpsvps[i]
            # __import__("ipdb").set_trace()
            for method in ['aurora', 'cubic', 'bbr']:
                plot_time([osp.join(di, '{}_simulation_log.csv'.format(method))], 
                           osp.join(di, "trace.json"),
                           di)

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
    # plt.style.use('seaborn-deep')
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 28
    # plt.rcParams['axes.labelsize'] = 38
    # plt.rcParams['legend.fontsize'] = 24
    # plt.rcParams["figure.figsize"] = (11,6)

    rpath = "/data/gengchen/PCC-RL"
    MODEL_PATH = model_path
    SAVE_DIR = osp.join(rpath, 'figs')
    print(MODEL_PATH)

    set_seed(42)

    vals2test = {
        "bandwidth": [10, 20, 30, 40, 50, 60, 70, 90, 100],
        "delay": [5, 50, 100, 150, 200],
        "loss": [0, 0.01, 0.02, 0.03, 0.04, 0.05],
        "queue": [0.2, 0.5, 1.2, 1.8, 2.3, 3],
        "T_s": [0, 0.1, 0.2, 0.4, 0.5, 0.8, 1, 3, 6, 9, 15],
        # "delay_noise": [0, 20, 40, 60, 80, 100],
    }

    reward = []
    basename = '_'.join(name.split('_')[:-1])
    for key in vals2test:
        reward.append(compare_metric(MODEL_PATH, name + "_" + key, osp.join(SAVE_DIR, basename, key), vals2test, key))
    return np.array(reward).mean()

import json

def calculate_std_of_err(arr, brr):
    temp = np.array(arr) - np.array(brr)
    return np.std(temp) / temp.shape[0] 

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--intv", type=int, default=1)
    parser.add_argument("--bo", type=int, default=0)
    args = parser.parse_args()
    name = "test2d_0913_with_bo{}".format(args.bo)
    rpath = "/data/gengchen/PCC-RL/data/bbr-0829"
    model_path = os.path.join(rpath, 'bo_{}_model_step_{}.ckpt'.format(args.bo, 36000))
    f = open('{}.csv'.format(name), 'a', 1)
    import pandas as pd
    try:
        df = pd.read_csv("{}.csv".format(name))
    except:
        df = None
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(['bw', 'ts', 'deltar', 'estd', 'aurora_reward', 'bbr_reward', 'aurora_error', 'bbr_error', 'trace_cnt'])
    rang = {
        "bandwidth_lower_bound": [
            0.1,
            0.1
        ],
        "bandwidth_upper_bound": [
            100,
            100
        ],
        "delay": [
            100,
            100
        ],
        "loss": [
            0,
            0
        ],
        "queue": [
            1.6,
            1.6
        ],
        "T_s": [
            3,
            3
        ],
        "duration": [
            30,
            30
        ],
        "delay_noise": [
            0,
            0
        ],
        "weight": 1
    }
    # bws = [0.1, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # bws = [0.1, 1, 2, 4, 7, 9, 10, 12, 14, 18, 20, 22, 25, 28]
    # bws = [0.2]
    # bws = [ 30,  40,  50,  60,  70,  80,  90, 100]
    # Tss = [0.1, 0.2, 0.4, 0.8, 1, 1.5, 2, 4, 8, 10, 12, 15, 17, 20, 22, 24, 30]
    # bws = [1.5]
    # bws = [2.5]
    # bws = [3]
    # bws = [120, 140, 160, 180, 200, 250]
    # Tss = [ 0.1,  0.2,  0.4,  0.8,  1. ,  1.5,  2. ,  4. ,  8. , 10. , 12. , 15. , 17. , 20. , 22. , 24. , 30. ]
    Tss = [ 0.1,  0.2,  0.4,  0.8,  1. ,  1.5,  2. ,  4. ,  8. , 10. , 12. , 15. , 17. , 20. , 22.]
    bws = [6.0e-01, 8.0e-01, 1.0e+00, 1.5e+00, 2.0e+00, 2.5e+00, 3.0e+00, 4.0e+00, 5.0e+00, 7.0e+00, 1.0e+01, 1.5e+01, 2.0e+01, 2.5e+01, 4.0e+01, 6.0e+01, 8.0e+01, 1.0e+02]
    f = open("log_{}".format(args.bo), 'a')
    from tqdm import tqdm
    for bw in tqdm(bws, total = len(bws)):
        for Tsi in tqdm(range(args.start, len(Tss), args.intv), total = len(Tss) / args.intv):
            Ts = Tss[Tsi]
            # breakpoint()
            if df is not None and ((df['bw'] == str(bw)) & (df['ts'] == str(Ts))).sum() > 0:
                continue
        # for Ts in tqdm(Tss, total = len(Tss)):
            rang["bandwidth_upper_bound"] = [bw, bw]
            rang["T_s"] = [Ts, Ts]
            arr = []
            brr = []
            deltaR, ar, br, ae, be, ars, brs = test_with_param(rang, "t0_{}_bw_{}_T_{}".format(name, bw, Ts), model_path, trace_cnt = 3)
            arr.extend(ars)
            brr.extend(brs)
            errorstd = calculate_std_of_err(arr, brr)
            deltaR = np.mean(np.array(arr) - np.array(brr))
            tcnt = 3
            # breakpoint()
            while (( errorstd > 5 ) and tcnt < 18 ) and bw < 50:
                print(tcnt)
                if deltaR - errorstd > 10:
                    break
                deltaR, ar, br, ae, be, ars, brs = test_with_param(rang, "t0_{}_bw_{}_T_{}".format(name, bw, Ts), model_path, trace_cnt = 5)
                tcnt += 5
                arr.extend(ars)
                brr.extend(brs)
                temp = np.array(arr) - np.array(brr)
                errorstd = calculate_std_of_err(arr, brr)
                deltaR = np.mean(np.array(arr) - np.array(brr))
            f.writelines(
                [','.join(map(str, arr)), ','.join(map(str, brr))]
            )
            arr = np.array(arr)
            brr = np.array(brr)
            deltaR = np.mean(arr - brr)
            ar = np.mean(arr)
            br = np.mean(brr)
            ae = np.std(arr) / arr.shape[0]
            be = np.std(brr) / brr.shape[0]
            writer.writerow([
                bw, Ts, deltaR, errorstd, ar, br, ae, be, tcnt
            ])
    exit(0)
    # for i in tqdm(range(1, 25)):
    #     prev_model = osp.join(rpath, 'bo_{}_model_step_36000.ckpt'.format(i-1))
    #     cur_model = osp.join(rpath, 'bo_{}_model_step_36000.ckpt'.format(i))
    #     with open(osp.join(rpath, 'bo_{}.json'.format(i))) as f:
    #         rang = json.load(f)[-1]
    #     test_with_param(rang, 'bo_{}_prev_model_bbr_0829'.format(i), prev_model) 
    #     test_with_param(rang, 'bo_{}_after_model_bbr_0829'.format(i), cur_model) 
    # exit(0)
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--intv", type=int, required=True)
    args = parser.parse_args()
    # name = "udr-large-genet-082010"
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
    # for i in range(80):
    #     compare(
    #         osp.join(rpath, "data", name, "bo_{}_model_step_36000.ckpt".format(i)),
    #         name+"_fix_bo_{}".format(i)
    #     )
    for i in range(args.start, 120, args.intv):
        compare(
            osp.join(rpath, "data", "udr-large-genet-082010", "bo_{}_model_step_36000.ckpt".format(i)),
            "new_cubic_genet_bo_{}".format(i)
        )
