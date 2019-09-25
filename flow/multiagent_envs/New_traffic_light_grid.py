"""Multi-agent environments for scenarios with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m grid.
"""

import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from flow.core import rewards
from flow.envs.my_green_wave_env import NewPO_TrafficLightGridEnv
from flow.multiagent_envs.multiagent_env import MultiEnv


class New_MultiTrafficLightGridPOEnv(NewPO_TrafficLightGridEnv, MultiEnv):

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):
        super().__init__(env_params, sim_params, scenario, simulator)

    @property
    def observation_space(self):
        """State space that is partially observed.

        Velocities, distance to intersections, edge number (for nearby
        vehicles) from each direction, local edge information, and traffic
        light state.
        """
        tl_box = Box(
            low=0.,
            high=1,
            shape=(2 * self.num_observed + 2 * 2 + 3 * self.num_traffic_lights,),
            dtype=np.float32)
        return tl_box

    @property
    def action_space(self):
        """See class definition."""
        if self.discrete:
            return Discrete(2)
        else:
            return Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32)

    def get_state(self):
        max_speed = max(
            self.k.scenario.speed_limit(edge)
            for edge in self.k.scenario.get_edge_list())

        speeds = []
        dist_to_intersec = []
        edge_number = []

        all_observed_ids = []
        veh_num = 0
        distance = [[],[],[],[],[],[]]
        speed = [[],[],[],[],[],[]]

        # todo : code optimize

        long_length10 = self.k.scenario.edge_length("1_0")
        long_length11 = self.k.scenario.edge_length("1_0") + self.k.scenario.edge_length("1_1")
        long_length20 = self.k.scenario.edge_length("2_0")
        long_length21 = self.k.scenario.edge_length("2_0") + self.k.scenario.edge_length("2_1")
        long_length30 = self.k.scenario.edge_length("3_0")
        long_length31 = self.k.scenario.edge_length("3_0") + self.k.scenario.edge_length("3_1")
        long_length32 = self.k.scenario.edge_length("3_0") + self.k.scenario.edge_length("3_1") + self.k.scenario.edge_length("3_2")
        long_length40 = self.k.scenario.edge_length("4_0")
        long_length41 = self.k.scenario.edge_length("4_0") + self.k.scenario.edge_length("4_1")
        long_length42 = self.k.scenario.edge_length("4_0") + self.k.scenario.edge_length("4_1") + self.k.scenario.edge_length("4_2")
        long_length50 = self.k.scenario.edge_length("5_0")
        long_length51 = self.k.scenario.edge_length("5_0") + self.k.scenario.edge_length("5_1")
        long_length52 = self.k.scenario.edge_length("5_0") + self.k.scenario.edge_length("5_1") + self.k.scenario.edge_length("5_2")
        long_length60 = self.k.scenario.edge_length("6_0")
        long_length61 = self.k.scenario.edge_length("6_0") + self.k.scenario.edge_length("6_1")
        long_length62 = self.k.scenario.edge_length("6_0") + self.k.scenario.edge_length("6_1") + self.k.scenario.edge_length("6_2")

        to_normal = max([long_length11, long_length21, long_length32, long_length42, long_length52, long_length62])

        def my_normalize(a):
            return a/to_normal

        for veh_id in self.k.vehicle.get_ids():
            veh_num = veh_num + 1
            veh_pos = self.k.vehicle.get_position(veh_id)
            veh_speed = self.k.vehicle.get_speed(veh_id) / max_speed
            if self.k.vehicle.get_edge(veh_id) == "1_0":
                distance[0].append(my_normalize(long_length10 - veh_pos))
                speed[0].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "1_1":
                distance[0].append(my_normalize(long_length11 - veh_pos))
                speed[0].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":2_0":
                distance[0].append(my_normalize(long_length10))
                speed[0].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "2_0":
                distance[1].append(my_normalize(long_length20 - veh_pos))
                speed[1].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "2_1":
                distance[1].append(my_normalize(long_length21 - veh_pos))
                speed[1].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":3_0":
                distance[1].append(my_normalize(long_length20))
                speed[1].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "3_0":
                distance[2].append(my_normalize(long_length30 - veh_pos))
                speed[2].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "3_1":
                distance[2].append(my_normalize(long_length31 - veh_pos))
                speed[2].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "3_2":
                distance[2].append(my_normalize(long_length32 - veh_pos))
                speed[2].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":6_0":
                distance[2].append(my_normalize(long_length30))
                speed[2].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":10_0":
                distance[2].append(my_normalize(long_length31))
                speed[2].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "4_0":
                distance[3].append(my_normalize(long_length40 - veh_pos))
                speed[3].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "4_1":
                distance[3].append(my_normalize(long_length41 - veh_pos))
                speed[3].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "4_2":
                distance[3].append(my_normalize(long_length42 - veh_pos))
                speed[3].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":7_0":
                distance[3].append(my_normalize(long_length40))
                speed[3].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":12_0":
                distance[3].append(my_normalize(long_length41))
                speed[3].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "5_0":
                distance[4].append(my_normalize(long_length50 - veh_pos))
                speed[4].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "5_1":
                distance[4].append(my_normalize(long_length51 - veh_pos))
                speed[4].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "5_2":
                distance[4].append(my_normalize(long_length52 - veh_pos))
                speed[4].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":8_0":
                distance[4].append(my_normalize(long_length50))
                speed[4].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":13_0":
                distance[4].append(my_normalize(long_length51))
                speed[4].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "6_0":
                distance[5].append(my_normalize(long_length60 - veh_pos))
                speed[5].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "6_1":
                distance[5].append(my_normalize(long_length61 - veh_pos))
                speed[5].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == "6_2":
                distance[5].append(my_normalize(long_length62 - veh_pos))
                speed[5].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":9_0":
                distance[5].append(my_normalize(long_length60))
                speed[5].append(veh_speed)
            elif self.k.vehicle.get_edge(veh_id) == ":15_0":
                distance[5].append(my_normalize(long_length61))
                speed[5].append(veh_speed)

        real_dist = [[1] * self.num_observed, [1] * self.num_observed, [1] * self.num_observed, [1] * self.num_observed, [1] * self.num_observed, [1] * self.num_observed]
        real_speed = [[1] * self.num_observed, [1] * self.num_observed, [1] * self.num_observed, [1] * self.num_observed, [1] * self.num_observed, [1] * self.num_observed]

        real_density = []
        real_velocity_avg = []

        for i in range(len(distance)):
            if len(real_speed) > 0 :
                real_velocity_avg.append(sum(real_speed[i])/float(len(real_speed)))
                real_density.append(len(real_speed[i])/float(12))
            else :
                real_velocity_avg.append(1)
                real_density.append(0)

        for i in range(len(distance)):
            if len(distance[i]) < self.num_observed :
                diff = self.num_observed - len(distance[i])
                distance[i] += [1] * diff
                speed[i] += [1] * diff
            else :
                distance[i] = distance[i][:self.num_observed]
                speed[i] = speed[i][:self.num_observed]

            order = np.argsort(np.array(distance[i]))

            for j in range(self.num_observed) :
                real_dist[i][order[j]] = distance[i][j]
                real_speed[i][order[j]] = speed[i][j]

        obs = {"1": np.array(np.concatenate([real_speed[0], real_speed[1], real_density[0:2], real_velocity_avg[0:2],
        self.last_change.flatten().tolist(),self.direction.flatten().tolist(),self.currently_yellow.flatten().tolist()])),
        "4":np.array(np.concatenate([real_speed[2], real_speed[3], real_density[2:4], real_velocity_avg[2:4],
        self.last_change.flatten().tolist(),self.direction.flatten().tolist(),self.currently_yellow.flatten().tolist()])),
        "5":np.array(np.concatenate([real_speed[4], real_speed[5], real_density[4:6], real_velocity_avg[4:6],
        self.last_change.flatten().tolist(),self.direction.flatten().tolist(),self.currently_yellow.flatten().tolist()]))}

        return obs

    def _apply_rl_actions(self, rl_actions):

        i = 0
        for rl_id, rl_action in rl_actions.items():
            #node = ["1", "4", "5"]
            action = rl_action > 0.0
            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id=rl_id,
                            state="Gr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id=rl_id,
                            state='rG')
                    self.currently_yellow[i] = 0
            else:
                if action:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id=rl_id,
                            state='yr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id=rl_id,
                            state='ry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1
            i = i + 1

    def compute_reward(self, rl_actions, **kwargs):

        if kwargs['fail'] == False :
            rew = rewards.average_velocity(self) / 3
        else :
            rew = (rewards.average_velocity(self) - 5000) / 3

        rews = {"1": rew, "4": rew, "5": rew}

        return rews


    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        for veh_ids in self.observed_ids:
            for veh_id in veh_ids:
                self.k.vehicle.set_observed(veh_id)
