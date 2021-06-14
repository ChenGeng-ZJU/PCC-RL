# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import csv
import heapq
import os
import random
import sys
import time
import math
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import gym
from gym import spaces
from gym.envs.registration import register
from gym.utils import seeding
import numpy as np

from common import sender_obs
from common.utils import pcc_aurora_reward, read_json_file
from simulator.trace import Trace

import pandas as pd

MAX_CWND = 5000
MIN_CWND = 4

MAX_RATE = 20000
MIN_RATE = 5

REWARD_SCALE = 0.001

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'

BYTES_PER_PACKET = 1500

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = True
MAX_LATENCY_NOISE = 1.1

# DEBUG = True
DEBUG = False

MI_RTT_PROPORTION = 1.0

# PACKET_LOG_FLAG = False
PACKET_LOG_FLAG = True


def debug_print(msg):
    if DEBUG:
        print(msg, file=sys.stderr, flush=True)


class EmuReplay:
    def __init__(self, ):
        df = pd.read_csv('aurora_emulation_log.csv')
        self.ts = df['timestamp'].tolist()
        self.send_rate = df['send_rate'].tolist()
        self.idx = 0

    def get_ts(self):
        if self.idx > len(self.ts):
            self.idx = len(self.ts) -1
        ts = self.ts[self.idx]
        self.idx += 1
        return ts

    def get_rate(self):
        return self.send_rate[self.idx] / 8 / BYTES_PER_PACKET

    def reset(self):
        self.idx = 0


class Link():

    def __init__(self, trace: Trace):
        self.trace = trace
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

        self.queue_size = self.trace.get_queue_size()
        self.pkt_in_queue = 0

    def get_cur_queue_delay(self, event_time):
        self.pkt_in_queue = max(0, self.pkt_in_queue -
                                (event_time - self.queue_delay_update_time) *
                                self.get_bandwidth(event_time))
        self.queue_delay_update_time = event_time
        cur_queue_delay = math.ceil(
            self.pkt_in_queue) / self.get_bandwidth(event_time)
        return cur_queue_delay

    def get_cur_latency(self, event_time):
        q_delay = self.get_cur_queue_delay(event_time)
        # print('queue delay: ', q_delay)
        return self.trace.get_delay(event_time) / 1000.0 + q_delay

    def packet_enters_link(self, event_time):
        if (random.random() < self.trace.get_loss_rate()):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        extra_delay = 1.0 / self.get_bandwidth(event_time)
        if 1 + math.ceil(self.pkt_in_queue) > self.queue_size:
            # print("{}\tDrop!".format(event_time))
            return False
        self.queue_delay += extra_delay
        self.pkt_in_queue += 1
        return True

    def print_debug(self):
        print("Link:")
        # TODO: Do not use timestamp 0.
        print("Bandwidth: %.3fMbps" % (self.trace.get_bandwidth(0)))
        # TODO: Do not use timestamp 0.
        print("Delay: %.3fms" % (self.trace.get_delay(0)))
        print("Queue Delay: %.3fms" % (self.queue_delay * 1000))
        print("One Packet Queue Delay: %.3fms" % (
            1000.0 * 1 / (self.trace.get_bandwidth(0) * 1e6 / 8 / BYTES_PER_PACKET)))
        print("Queue size: %dpackets" % self.queue_size)
        print("Loss: %.4f" % self.trace.get_loss_rate())

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.pkt_in_queue = 0

    def get_bandwidth(self, ts):
        return self.trace.get_bandwidth(ts) * 1e6 / 8 / BYTES_PER_PACKET


class Network():

    def __init__(self, senders, links, env):
        self.event_count = 0
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.queue_initial_packets()
        self.env = env

        self.pkt_log = []

    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.q, (0, sender, EVENT_TYPE_SEND,
                                    0, 0.0, False, self.event_count, sender.rto, 0))
            self.event_count += 1

    def reset(self):
        self.pkt_log = []
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()

    def get_cur_time(self):
        return self.cur_time

    def run_for_dur(self, dur, action=None):
        # if self.cur_time > 1.75:
        #     pass
        # else:
        #     self.senders[0].rate = self.env.replay.get_rate()
        #     dur = self.env.replay.get_ts() - self.cur_time
        end_time = min(self.cur_time + dur, self.env.current_trace.timestamps[-1])
        debug_print('MI from {} to {}, dur {}'.format(
            self.cur_time, end_time, dur))
        for sender in self.senders:
            sender.reset_obs()
        while True:
            event_time, sender, event_type, next_hop, cur_latency, dropped, \
                event_id, rto, event_queue_delay = self.q[0]
            if event_time >= end_time:
                self.cur_time = end_time
                break
            event_time, sender, event_type, next_hop, cur_latency, dropped, \
                event_id, rto, event_queue_delay = heapq.heappop(self.q)
            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            new_event_queue_delay = event_queue_delay
            push_new_event = False
            debug_print("Got %d event %s, to link %d, latency %f at time %f, "
                        "next_hop %d, dropped %s, event_q length %f, "
                        "sender rate %f, duration: %f, queue_size: %f, "
                        "rto: %f, cwnd: %f, ssthresh: %f, sender rto %f, "
                        "pkt in flight %d, wait time %d" % (
                            event_id, event_type, next_hop, cur_latency,
                            event_time, next_hop, dropped, len(self.q),
                            sender.rate, dur, self.links[0].queue_size,
                            rto, sender.cwnd, sender.ssthresh, sender.rto,
                            int(sender.bytes_in_flight/BYTES_PER_PACKET),
                            sender.pkt_loss_wait_time))
            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    # if cur_latency > 1.0:
                    #     sender.timeout(cur_latency)
                    # sender.on_packet_lost(cur_latency)
                    if rto >= 0 and cur_latency > rto and sender.pkt_loss_wait_time <= 0:
                        sender.timeout()
                        dropped = True
                        new_dropped = True
                    elif dropped:
                        sender.on_packet_lost(cur_latency)
                        if PACKET_LOG_FLAG:
                            self.pkt_log.append([self.cur_time, event_id, 'lost',
                                                 BYTES_PER_PACKET])
                    else:
                        sender.on_packet_acked(cur_latency)
                        debug_print('Ack packet at {}'.format(self.cur_time))
                        # log packet acked
                        if PACKET_LOG_FLAG:
                            self.pkt_log.append([self.cur_time, event_id, 'acked',
                                                 BYTES_PER_PACKET, cur_latency,
                                                 event_queue_delay])
                else:
                    new_next_hop = next_hop + 1
                    new_event_queue_delay += sender.path[next_hop].get_cur_queue_delay(
                        self.cur_time)
                    link_latency = sender.path[next_hop].get_cur_latency(
                        self.cur_time)
                    # link_latency *= self.env.current_trace.get_delay_noise_replay(self.cur_time)
                    # if USE_LATENCY_NOISE:
                    #     link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            elif event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    if sender.can_send_packet():
                        sender.on_packet_sent()
                        # print('Send packet at {}'.format(self.cur_time))
                        if PACKET_LOG_FLAG:
                            self.pkt_log.append([self.cur_time, event_id, 'sent',
                                                 BYTES_PER_PACKET])
                        push_new_event = True
                    heapq.heappush(self.q, (self.cur_time + (1.0 / sender.rate),
                                            sender, EVENT_TYPE_SEND, 0, 0.0,
                                            False, self.event_count, sender.rto,
                                            0))
                    self.event_count += 1

                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1

                new_event_queue_delay += sender.path[next_hop].get_cur_queue_delay(
                    self.cur_time)
                link_latency = sender.path[next_hop].get_cur_latency(
                    self.cur_time)
                # if USE_LATENCY_NOISE:
                #     link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)

                # link_latency += self.env.current_trace.get_delay_noise(self.cur_time) / 1000
                # link_latency *= self.env.current_trace.get_delay_noise_replay(self.cur_time)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(
                    self.cur_time)
                if not new_dropped:
                    sender.queue_delay_samples.append(new_event_queue_delay)

            if push_new_event:
                heapq.heappush(self.q, (new_event_time, sender, new_event_type,
                                        new_next_hop, new_latency, new_dropped,
                                        event_id, rto, new_event_queue_delay))
        for sender in self.senders:
            sender.record_run()

        sender_mi = self.senders[0].get_run_data()
        throughput = sender_mi.get("recv rate")  # bits/sec
        latency = sender_mi.get("avg latency") # second
        loss = sender_mi.get("loss ratio")
        debug_print("thpt %f, delay %f, loss %f, bytes sent %f, bytes acked %f" % (
            throughput/1e6, latency, loss, sender_mi.bytes_sent, sender_mi.bytes_acked))
        reward = pcc_aurora_reward(
            throughput / 8 / BYTES_PER_PACKET, latency, loss,
            np.mean(self.env.current_trace.bandwidths) * 1e6 / 8 / BYTES_PER_PACKET)

        if latency > 0.0:
            self.env.run_dur = MI_RTT_PROPORTION * sender_mi.get("avg latency") + (1 / self.links[0].get_bandwidth(self.cur_time))
            # self.env.run_dur = max(MI_RTT_PROPORTION * sender_mi.get("avg latency"), 5 * (1 / self.senders[0].rate))
        # print(self.env.run_dur)
        return reward * REWARD_SCALE


class Sender():

    def __init__(self, rate, path, dest, features, cwnd=25, history_len=10,
                 delta_scale=1):
        self.id = Sender._get_next_id()
        self.delta_scale = delta_scale
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.queue_delay_samples = []
        self.prev_rtt_samples = self.rtt_samples
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd
        self.use_cwnd = False
        self.rto = -1
        self.ssthresh = 0
        self.pkt_loss_wait_time = -1
        self.estRTT = 1000000 / 1e6  # SynInterval in emulation
        self.RTTVar = self.estRTT / 2  # RTT variance
        # self.got_data = False

    _next_id = 1

    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta):
        # if self.got_data:
        delta *= self.delta_scale
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def apply_cwnd_delta(self, delta):
        delta *= self.delta_scale
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if self.use_cwnd:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        self.estRTT = (7.0 * self.estRTT + rtt) / 8.0  # RTT of emulation way
        self.RTTVar = (self.RTTVar * 7.0 + abs(rtt - self.estRTT) * 1.0) / 8.0

        self.acked += 1
        self.rtt_samples.append(rtt)
        # self.rtt_samples.append(self.estRTT)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self, rtt):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate
        # print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
        if self.rate < MIN_RATE:
            self.rate = MIN_RATE

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        # if self.cwnd > MAX_CWND:
        #     self.cwnd = MAX_CWND
        # if self.cwnd < MIN_CWND:
        #     self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        # if not self.got_data and smi.rtt_samples:
        #     self.got_data = True
        #     self.history.step(smi)
        # else:
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()

        #obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)
        # print(self.acked, self.sent)
        rtt_samples = self.rtt_samples if self.rtt_samples else self.prev_rtt_samples
        # if not self.rtt_samples:
        #     print(self.obs_start_time, obs_end_time, self.rate)
        # rtt_samples is empty when there is no packet acked in MI
        # Solution: inherit from previous rtt_samples.

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            queue_delay_samples=self.queue_delay_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        if self.rtt_samples:
            self.prev_rtt_samples = self.rtt_samples
        self.rtt_samples = []
        self.queue_delay_samples = []
        self.obs_start_time = self.net.get_cur_time()

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.estRTT = 1000000 / 1e6  # SynInterval in emulation
        self.RTTVar = self.estRTT / 2  # RTT variance

        # self.got_data = False

    def timeout(self):
        # placeholder
        pass


class SimulatedNetworkEnv(gym.Env):

    def __init__(self, traces, history_len=10,
                 features="sent latency inflation,latency ratio,send ratio",
                 congestion_control_type="aurora", train_flag=False,
                 delta_scale=1.0):
        """Network environment used in simulation.
        congestion_control_type: aurora is pcc-rl. cubic is TCPCubic.
        """
        assert congestion_control_type in {"aurora", "cubic"}, \
            "Unrecognized congestion_control_type {}.".format(
                congestion_control_type)
        # self.replay = EmuReplay()
        self.delta_scale = delta_scale
        self.traces = traces
        self.current_trace = np.random.choice(self.traces)
        self.train_flag = train_flag
        self.congestion_control_type = congestion_control_type
        if self.congestion_control_type == 'aurora':
            self.use_cwnd = False
        elif self.congestion_control_type == 'cubic':
            self.use_cwnd = True

        self.history_len = history_len
        # print("History length: %d" % history_len)
        self.features = features.split(",")
        # print("Features: %s" % str(self.features))

        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links, self)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        if self.use_cwnd:
            self.action_space = spaces.Box(
                np.array([-1e12, -1e12]), np.array([1e12, 1e12]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                np.array([-1e12]), np.array([1e12]), dtype=np.float32)

        self.observation_space = None
        # use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec,
                                                    self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.episodes_run = -1

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs).reshape(-1,)
        return sender_obs

    def step(self, actions):
        #print("Actions: %s" % str(actions))
        # print(actions)
        for i in range(0, 1):  # len(actions)):
            #print("Updating rate for sender %d" % i)
            action = actions
            self.senders[i].apply_rate_delta(action[0])
            if self.use_cwnd:
                self.senders[i].apply_cwnd_delta(action[1])
        # print("Running for %fs" % self.run_dur)
        reward = self.net.run_for_dur(self.run_dur, action=actions[0])
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()

        should_stop = self.current_trace.is_finished(self.net.get_cur_time())

        self.reward_sum += reward
        # print('env step: {}s'.format(time.time() - t_start))
        return sender_obs, reward, should_stop, {}

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def create_new_links_and_senders(self):
        # self.replay.reset()
        self.links = [Link(self.current_trace), Link(self.current_trace)]
        if self.congestion_control_type == "aurora":
            if not self.train_flag:

                self.senders = [Sender( #self.replay.get_rate(),
                    # 2500000 / 8 /BYTES_PER_PACKET / 0.048,
                    # 12000000 / 8 /BYTES_PER_PACKET / 0.048,
                    # 10 / (self.current_trace.get_delay(0) *2/1000),
                    100,
                    [self.links[0], self.links[1]], 0,
                                       self.features,
                                       history_len=self.history_len,
                                       delta_scale=self.delta_scale)]
            else:
                # self.senders = [Sender(random.uniform(0.3, 1.5) * bw,
                #                        [self.links[0], self.links[1]], 0,
                #                        self.features,
                #                        history_len=self.history_len)]
                # self.senders = [Sender(random.uniform(10/bw, 1.5) * bw,
                #                        [self.links[0], self.links[1]], 0,
                #                        self.features,
                #                        history_len=self.history_len,
                #                        delta_scale=self.delta_scale)]
                self.senders = [Sender(100,
                                       [self.links[0], self.links[1]], 0,
                                       self.features,
                                       history_len=self.history_len,
                                       delta_scale=self.delta_scale)]
        elif self.congestion_control_type == "cubic":
            raise NotImplementedError
        else:
            raise RuntimeError("Unrecognized congestion_control_type {}".format(
                self.congestion_control_type))
        # self.run_dur = 3 * lat
        # self.run_dur = 1 * lat
        if not self.senders[0].rtt_samples:
            # self.run_dur = 0.473
            # self.run_dur = 5 / self.senders[0].rate
            self.run_dur = 0.01
            # self.run_dur = self.current_trace.get_delay(0) * 2 / 1000
            # self.run_dur = self.replay.get_ts() -  0

    def reset(self):
        self.steps_taken = 0
        self.net.reset()
        self.current_trace = np.random.choice(self.traces)
        self.current_trace.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links, self)
        self.episodes_run += 1
        # self.replay.reset()
        self.net.run_for_dur(self.run_dur)
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        # print("Reward: %0.2f, Ewma Reward: %0.2f" % (self.reward_sum, self.reward_ewma))
        self.reward_sum = 0.0
        return self._get_all_sender_obs()


register(id='PccNs-v0', entry_point='simulator.network:SimulatedNetworkEnv')