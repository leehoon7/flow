"""Contains the grid scenario class."""

import numpy as np
from numpy import pi, sin, cos, linspace

from flow.scenarios.base_scenario import Scenario
from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from collections import defaultdict

ADDITIONAL_NET_PARAMS = {
    "speed_limit": 10,
}

length = {
    "length_a" : 25,
    "length_b" : 10,
    "length_c" : 5,
    "length_d" : 40,
    "length_e" : 20,
}


class NewSimpleGridScenario(Scenario):

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):

        self.speed_limit = net_params.additional_params["speed_limit"]
        self.use_traffic_lights = False
        self.len = [
            length["length_a"],
            length["length_b"],
            length["length_c"],
            length["length_d"],
            length["length_e"],
        ]
        self.num_edges = 25

        # name of the scenario (DO NOT CHANGE)
        self.name = "BobLoblawsLawBlog"

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        node_type = "traffic_light" if self.use_traffic_lights else "priority"
        a, b, c, d, e = self.len[0], self.len[1], self.len[2], self.len[3], self.len[4]
        idxy = [
            ["0", 0, 0],
            ["1", a, 0],
            ["2", a, b+c],
            ["3", a, -b-c],
            ["4", a+d, b+c],
            ["5", a+d, -b-c],
            ["6", a+d, 2*b+c],
            ["7", a+d, c],
            ["8", a+d, -c],
            ["9", a+d, -2*b-c],
            ["10", a+2*d, 2*b+c],
            ["11", a+2*d, b+c],
            ["12", a+2*d, c],
            ["13", a+2*d, -c],
            ["14", a+2*d, -b-c],
            ["15", a+2*d, -2*b-c],
            ["16", a+b+2*d, b+c],
            ["17", a+b+2*d, -b-c],
            ["18", a+b+2*d, b+c+e],
            ["19", a+b+2*d, -b-c-e],
            ["20", 0, b+c+e],
            ["21", 0, -b-c-e],
        ]
        nodes = []
        for i in range(len(idxy)):
            if i == 1 or i == 4 or i == 5 :
                node_type = "traffic_light"
            else :
                node_type = "priority"
            nodes.append({
                "id": idxy[i][0],
                "x": idxy[i][1],
                "y": idxy[i][2],
                "type": node_type,
            })
        return nodes

    def specify_edges(self, net_params):

        a, b, c, d, e = self.len[0], self.len[1], self.len[2], self.len[3], self.len[4]
        edges = []
        #edges = ["0","1_0","1_1","2_0","2_1","3_0","3_1","3_2","4_0","4_1","4_2","5_0","5_1","5_2","6_0","6_1","6_2","7_0","7_1","7_2","7_3","8_0","8_1","8_2","8_3"]
        #id, from, to, length
        idftl = [
            ["0", 1, 0, a],
            ["1_0", 2, 1, b+c],
            ["1_1", 4, 2, d],
            ["2_0", 3, 1, b+c],
            ["2_1", 5, 3, d],
            ["3_0", 6, 4, b],
            ["3_1", 10, 6, d],
            ["3_2", 11, 10, b],
            ["4_0", 7, 4, b],
            ["4_1", 12, 7, d],
            ["4_2", 11, 12, b],
            ["5_0", 8, 5, b],
            ["5_1", 13, 8, d],
            ["5_2", 14, 13, b],
            ["6_0", 9, 5, b],
            ["6_1", 15, 9, d],
            ["6_2", 14, 15, b],
            ["7_0", 16, 11, b],
            ["7_1", 18, 16, e],
            ["7_2", 20, 18, a+b+2*d],
            ["7_3", 0, 20, b+c+e],
            ["8_0", 17, 14, b],
            ["8_1", 19, 17, e],
            ["8_2", 21, 19, a+b+2*d],
            ["8_3", 0, 21, b+c+e],
            #["9", 20, 21, 2*(b+c+e)],
        ]
        for i in range(len(idftl)):
            edges.append({
                "id": idftl[i][0],
                "type": "lane",
                "priority": 78,
                "from": str(idftl[i][1]),
                "to": str(idftl[i][2]),
                "length": idftl[i][3],
            })

        return edges

    def specify_routes(self, net_params):

        routes = defaultdict(list)

        routes['7_3'] = ['7_3', '7_2', '7_1', '7_0', '3_2', '3_1', '3_0', '1_1', '1_0', '0', '7_3']
        routes['7_1'] = ['7_1', '7_0', '3_2', '3_1', '3_0', '1_1', '1_0', '0', '7_3', '7_2', '7_1']
        routes['7_2'] = ['7_2', '7_1', '7_0', '4_2', '4_1', '4_0', '1_1', '1_0', '0', '7_3', '7_2']
        routes['8_3'] = ['8_3', '8_2', '8_1', '8_0', '5_2', '5_1', '5_0', '2_1', '2_0', '0', '8_3']
        routes['8_1'] = ['8_1', '8_0', '5_2', '5_1', '5_0', '2_1', '2_0', '0', '8_3', '8_2', '8_1']
        routes['8_2'] = ['8_2', '8_1', '8_0', '6_2', '6_1', '6_0', '2_1', '2_0', '0', '8_3', '8_2']

        routes['3_1'] = ['3_1', '3_0', '1_1', '1_0', '0', '7_3', '7_2', '7_1', '7_0', '3_2', '3_1']
        routes['4_1'] = ['4_1', '4_0', '1_1', '1_0', '0', '7_3', '7_2', '7_1', '7_0', '4_2', '4_1']
        routes['5_1'] = ['5_1', '5_0', '2_1', '2_0', '0', '8_3', '8_2', '8_1', '8_0', '5_2', '5_1']
        routes['6_1'] = ['6_1', '6_0', '2_1', '2_0', '0', '8_3', '8_2', '8_1', '8_0', '6_2', '6_1']

        return routes

    def specify_types(self, net_params):
        types = [{
            "id": "lane",
            "numLanes": 1,
            "speed": self.speed_limit,
        }]

        return types

    def specify_connections(self, net_params):
        return None

    def specify_edge_starts(self):
        return None

    @staticmethod
    def gen_custom_start_pos(cls, net_params, initial_config, num_vehicles):

        start_pos = []

        start_pos += [('3_1', 4),('3_1', 11),('3_1', 18),('3_1', 25),('3_1', 32),('3_1', 39)]
        start_pos += [('4_1', 4),('4_1', 11),('4_1', 18),('4_1', 25),('4_1', 32),('4_1', 39)]
        start_pos += [('5_1', 4),('5_1', 11),('5_1', 18),('5_1', 25),('5_1', 32),('5_1', 39)]
        start_pos += [('6_1', 4),('6_1', 11),('6_1', 18),('6_1', 25),('6_1', 32),('6_1', 39)]
        start_pos += [('7_3', 6),('7_3', 16),('7_3', 26)]
        start_pos += [('8_3', 6),('8_3', 16),('8_3', 26)]
        start_pos += [('7_2', 6),('7_2', 16),('7_2', 26),('7_2', 36)]
        start_pos += [('8_2', 6),('8_2', 16),('8_2', 26),('8_2', 36)]
        start_pos += [('7_1', 6)]
        start_pos += [('8_1', 6)]

        start_lanes = [0] * len(start_pos)

        return start_pos, start_lanes

    @property
    def node_mapping(self):
        mapping = {}
        mapping["center1"] = ["edge2", "edge1"]
        return None
