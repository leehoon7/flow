"""Environments for scenarios with traffic lights.

These environments are used to train traffic lights to regulate traffic flow
through an n x m grid.
"""

import numpy as np
import re

from math import exp

from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple

from flow.core import rewards
from flow.envs.base_env import Env

ADDITIONAL_ENV_PARAMS = {
    "switch_time": 2.0,
    "tl_type": "controlled",
    "discrete": False,
}

ADDITIONAL_PO_ENV_PARAMS = {
    "num_observed": 5,
    "target_velocity": 10,
}



class NewTrafficLightGridEnv(Env):
    """Environment used to train traffic lights.

    Required from env_params:

    * switch_time: minimum time a light must be constant before
      it switches (in seconds).
      Earlier RL commands are ignored.
    * tl_type: whether the traffic lights should be actuated by sumo or RL,
      options are respectively "actuated" and "controlled"
    * discrete: determines whether the action space is meant to be discrete or
      continuous

    States
        An observation is the distance of each vehicle to its intersection, a
        number uniquely identifying which edge the vehicle is on, and the speed
        of the vehicle.

    Actions
        The action space consist of a list of float variables ranging from 0-1
        specifying whether a traffic light is supposed to switch or not. The
        actions are sent to the traffic light in the grid from left to right
        and then top to bottom.

    Rewards
        The reward is the negative per vehicle delay minus a penalty for
        switching traffic lights

    Termination
        A rollout is terminated once the time horizon is reached.

    Additional
        Vehicles are rerouted to the start of their original routes once they
        reach the end of the network in order to ensure a constant number of
        vehicles.
    """

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):

        self.num_traffic_lights = 3
        self.tl_type = env_params.additional_params.get('tl_type')

        super().__init__(env_params, sim_params, scenario, simulator)

        # Saving env variables for plotting
        self.steps = env_params.horizon
        self.obs_var_labels = {
            'edges': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'velocities': np.zeros((self.steps, self.k.vehicle.num_vehicles)),
            'positions': np.zeros((self.steps, self.k.vehicle.num_vehicles))
        }

        self.last_change = np.zeros((self.num_traffic_lights, 1))
        self.direction = np.zeros((self.num_traffic_lights, 1))
        self.currently_yellow = np.zeros((self.num_traffic_lights, 1))
        self.min_switch_time = env_params.additional_params["switch_time"]

        if self.tl_type != "actuated":
            self.k.traffic_light.set_state(node_id='1', state="Gr")
            self.k.traffic_light.set_state(node_id='4', state="Gr")
            self.k.traffic_light.set_state(node_id='5', state="Gr")
            self.currently_yellow[0] = 0
            self.currently_yellow[1] = 0
            self.currently_yellow[2] = 0

        self.discrete = env_params.additional_params.get("discrete", False)

        self.acc_penalty = np.zeros((self.k.vehicle.num_vehicles, 1))

    @property
    def action_space(self):
        return Box(
            low=-1,
            high=1,
            shape=(self.num_traffic_lights,),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        speed = Box(
            low=0,
            high=1,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        dist_to_intersec = Box(
            low=0.,
            high=np.inf,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        edge_num = Box(
            low=0.,
            high=1,
            shape=(self.initial_vehicles.num_vehicles,),
            dtype=np.float32)
        traffic_lights = Box(
            low=0.,
            high=1,
            shape=(3 * self.rows * self.cols,),
            dtype=np.float32)
        return Tuple((speed, dist_to_intersec, edge_num, traffic_lights))

    def get_state(self):
        """See class definition."""
        # compute the normalizers
        grid_array = self.net_params.additional_params["grid_array"]
        max_dist = max(grid_array["short_length"],
                       grid_array["long_length"],
                       grid_array["inner_length"])

        # get the state arrays
        speeds = [
            self.k.vehicle.get_speed(veh_id) / self.k.scenario.max_speed()
            for veh_id in self.k.vehicle.get_ids()
        ]
        dist_to_intersec = [
            self.get_distance_to_intersection(veh_id) / max_dist
            for veh_id in self.k.vehicle.get_ids()
        ]
        edges = [
            self._convert_edge(self.k.vehicle.get_edge(veh_id)) /
            (self.k.scenario.network.num_edges - 1)
            for veh_id in self.k.vehicle.get_ids()
        ]

        state = [
            speeds, dist_to_intersec, edges,
            self.last_change.flatten().tolist(),
            self.direction.flatten().tolist(),
            self.currently_yellow.flatten().tolist()
        ]
        return np.array(state)

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        # check if the action space is discrete
        if self.discrete:
            # convert single value to list of 0's and 1's
            rl_mask = [int(x) for x in list('{0:0b}'.format(rl_actions))]
            rl_mask = [0] * (self.num_traffic_lights - len(rl_mask)) + rl_mask
        else:
            # convert values less than 0.5 to zero and above to 1. 0's indicate
            # that should not switch the direction
            rl_mask = rl_actions > 0.0
        for i, action in enumerate(rl_mask):
            node = ["1", "4", "5"]
            if self.currently_yellow[i] == 1:  # currently yellow
                self.last_change[i] += self.sim_step
                # Check if our timer has exceeded the yellow phase, meaning it
                # should switch to red
                if self.last_change[i] >= self.min_switch_time:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id=node[i],
                            state="Gr")
                    else:
                        self.k.traffic_light.set_state(
                            node_id=node[i],
                            state='rG')
                    self.currently_yellow[i] = 0
            else:
                if action:
                    if self.direction[i] == 0:
                        self.k.traffic_light.set_state(
                            node_id=node[i],
                            state='yr')
                    else:
                        self.k.traffic_light.set_state(
                            node_id=node[i],
                            state='ry')
                    self.last_change[i] = 0.0
                    self.direction[i] = not self.direction[i]
                    self.currently_yellow[i] = 1

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""
        return - rewards.min_delay_unscaled(self) \
            - rewards.boolean_action_penalty(rl_actions >= 0.5, gain=1.0)

    # ===============================
    # ============ UTILS ============
    # ===============================

    def get_distance_to_intersection(self, veh_ids):
        """Determine the distance from a vehicle to its next intersection.

        Parameters
        ----------
        veh_ids : str or str list
            vehicle(s) identifier(s)

        Returns
        -------
        float (or float list)
            distance to closest intersection
        """
        if isinstance(veh_ids, list):
            return [self.get_distance_to_intersection(veh_id)
                    for veh_id in veh_ids]
        return self.find_intersection_dist(veh_ids)

    def find_intersection_dist(self, veh_id):
        """Return distance from intersection.

        Return the distance from the vehicle's current position to the position
        of the node it is heading toward.
        """
        edge_id = self.k.vehicle.get_edge(veh_id)
        # FIXME this might not be the best way of handling this
        if edge_id == "":
            return -10
        if 'center' in edge_id:
            return 0
        edge_len = self.k.scenario.edge_length(edge_id)
        relative_pos = self.k.vehicle.get_position(veh_id)
        dist = edge_len - relative_pos
        return dist

    def _convert_edge(self, edges):
        """Convert the string edge to a number.

        Start at the bottom left vertical edge and going right and then up, so
        the bottom left vertical edge is zero, the right edge beside it  is 1.

        The numbers are assigned along the lowest column, then the lowest row,
        then the second lowest column, etc. Left goes before right, top goes
        before bot.

        The values are are zero indexed.

        Parameters
        ----------
        edges : list of str or str
            name of the edge(s)

        Returns
        -------
        list of int or int
            a number uniquely identifying each edge
        """
        if isinstance(edges, list):
            return [self._split_edge(edge) for edge in edges]
        else:
            return self._split_edge(edges)

    def _split_edge(self, edge):
        """Act as utility function for convert_edge."""
        if edge:
            if edge[0] == ":":  # center
                center_index = int(edge.split("center")[1][0])
                base = ((self.cols + 1) * self.rows * 2) \
                    + ((self.rows + 1) * self.cols * 2)
                return base + center_index + 1
            else:
                pattern = re.compile(r"[a-zA-Z]+")
                edge_type = pattern.match(edge).group()
                edge = edge.split(edge_type)[1].split('_')
                row_index, col_index = [int(x) for x in edge]
                if edge_type in ['bot', 'top']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * (row_index + 1))
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'bot' else edge_num + 1
                if edge_type in ['left', 'right']:
                    rows_below = 2 * (self.cols + 1) * row_index
                    cols_below = 2 * (self.cols * row_index)
                    edge_num = rows_below + cols_below + 2 * col_index + 1
                    return edge_num if edge_type == 'left' else edge_num + 1
        else:
            return 0

    def additional_command(self):
        """See parent class.

        Used to insert vehicles that are on the exit edge and place them
        back on their entrance edge.
        """
        for veh_id in self.k.vehicle.get_ids():
            self._reroute_if_final_edge(veh_id)

    def _reroute_if_final_edge(self, veh_id):
        """Reroute vehicle associated with veh_id.

        Checks if an edge is the final edge. If it is return the route it
        should start off at.
        """
        edge = self.k.vehicle.get_edge(veh_id)
        if edge == "":
            return
        if edge[0] == ":":  # center edge
            return
        pattern = re.compile(r"[a-zA-Z]+")
        edge_type = pattern.match(edge).group()
        edge = edge.split(edge_type)[1].split('_')
        row_index, col_index = [int(x) for x in edge]

        # find the route that we're going to place the vehicle on if we are
        # going to remove it
        route_id = None
        if edge_type == 'bot' and col_index == self.cols:
            route_id = "bot{}_0".format(row_index)
        elif edge_type == 'top' and col_index == 0:
            route_id = "top{}_{}".format(row_index, self.cols)
        elif edge_type == 'left' and row_index == 0:
            route_id = "left{}_{}".format(self.rows, col_index)
        elif edge_type == 'right' and row_index == self.rows:
            route_id = "right0_{}".format(col_index)

        if route_id is not None:
            type_id = self.k.vehicle.get_type(veh_id)
            lane_index = self.k.vehicle.get_lane(veh_id)
            # remove the vehicle
            self.k.vehicle.remove(veh_id)
            # reintroduce it at the start of the network
            self.k.vehicle.add(
                veh_id=veh_id,
                edge=route_id,
                type_id=str(type_id),
                lane=str(lane_index),
                pos="0",
                speed="max")

    # FIXME it doesn't make sense to pass a list of edges since the function
    # returns a flattened list with no padding, so we would lose information
    def k_closest_to_intersection(self, edges, k):
        """Return the vehicle IDs of k closest vehicles to an intersection.

        For each edge in edges, return the ids (veh_id) of the k vehicles
        in edge that are closest to an intersection (the intersection they
        are heading towards).

        - Performs no check on whether or not edge is going towards an
          intersection or not.
        - Does no padding if there are less than k vehicles on an edge.
        """
        if k < 0:
            raise ValueError("Function k_closest_to_intersection called with"
                             "parameter k={}, but k should be non-negative"
                             .format(k))

        if isinstance(edges, list):
            ids = [self.k_closest_to_intersection(edge, k) for edge in edges]
            # flatten the list before returning it
            return [veh_id for sublist in ids for veh_id in sublist]

        # get the ids of all the vehicles on the edge 'edges' ordered by
        # increasing distance to intersection
        veh_ids_ordered = sorted(
            self.k.vehicle.get_ids_by_edge(edges),
            key=self.get_distance_to_intersection)

        # return the ids of the k vehicles closest to the intersection
        return veh_ids_ordered[:k]


class NewPO_TrafficLightGridEnv(NewTrafficLightGridEnv):

    def __init__(self, env_params, sim_params, scenario, simulator='traci'):

        super().__init__(env_params, sim_params, scenario, simulator)
        self.num_observed = env_params.additional_params.get("num_observed", 2)
        self.observed_ids = []

    @property
    def observation_space(self):

        tl_box = Box(
            low=0.,
            high=1,
            shape=(6 * self.num_observed + 2 * 6 +
                   3 * self.num_traffic_lights,),
            dtype=np.float32)
        return tl_box

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


        return np.array(
            np.concatenate([
                real_speed[0],real_speed[1],real_speed[2],real_speed[3],real_speed[4],real_speed[5],
                real_density, real_velocity_avg,
                self.last_change.flatten().tolist(),
                self.direction.flatten().tolist(),
                self.currently_yellow.flatten().tolist()
            ]))

    def compute_reward(self, rl_actions, **kwargs):
        """See class definition."""

        if kwargs['fail'] == False :
            return rewards.average_velocity(self)
        else :
            return rewards.average_velocity(self) - 600

        '''
        if kwargs['fail'] == False :
            return  -rewards.min_delay_unscaled(self) - 10 * rewards.boolean_action_penalty(rl_actions >= 0.5, gain=1.0)
        else :
            return  -rewards.min_delay_unscaled(self) - 10 * rewards.boolean_action_penalty(rl_actions >= 0.5, gain=1.0) - 600
        '''


    def additional_command(self):
        """See class definition."""
        # specify observed vehicles
        [self.k.vehicle.set_observed(veh_id) for veh_id in self.observed_ids]


class GreenWaveTestEnv(NewTrafficLightGridEnv):
    """
    Class for use in testing.

    This class overrides RL methods of green wave so we can test
    construction without needing to specify RL methods
    """

    def _apply_rl_actions(self, rl_actions):
        """See class definition."""
        pass

    def compute_reward(self, rl_actions, **kwargs):
        """No return, for testing purposes."""
        return 0
