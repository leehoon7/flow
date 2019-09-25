"""Grid/green wave example."""

import json

import ray
try:
    from ray.rllib.agents.agent import get_agent_class
except ImportError:
    from ray.rllib.agents.registry import get_agent_class
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    InFlows, SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter, ContinuousRouter, IDMController

# time horizon of a single rollout
HORIZON = 350
# number of rollouts per training iteration
N_ROLLOUTS = 20
# number of parallel workers
N_CPUS = 3

def get_non_flow_params(add_net_params):
    initial = InitialConfig(spacing='custom')
    net = NetParams(no_internal_links=False,additional_params=add_net_params)
    return initial, net


tot_cars = 40

additional_env_params = {
        'target_velocity': 10,
        'switch_time': 1.0,
        'num_observed': 5,
        'discrete': False,
        'tl_type': 'controlled'
    }

additional_net_params = {
    'speed_limit': 10,
}

vehicles = VehicleParams()
vehicles.add(
    veh_id='idm',
    acceleration_controller=(IDMController, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=0,
        decel=7.5,  # avoid collisions at emergency stops
        speed_mode="all_checks",
    ),
    routing_controller=(ContinuousRouter, {}),
    num_vehicles=tot_cars)

flow_params = dict(
    exp_tag='my_green_wave',
    env_name='NewPO_TrafficLightGridEnv',
    scenario='NewSimpleGridScenario',
    simulator='traci',
    sim=SumoParams(
        sim_step=1,
        render=False,
    ),
    env=EnvParams(
        horizon=HORIZON,
        additional_params=additional_env_params,
    ),
    net=None,
    veh=vehicles,
    initial=None,
)


def setup_exps():

    initial_config, net_params = get_non_flow_params(add_net_params=additional_net_params)

    flow_params['initial'] = initial_config
    flow_params['net'] = net_params

    alg_run = 'PPO'

    agent_cls = get_agent_class(alg_run)
    config = agent_cls._default_config.copy()
    config['num_workers'] = N_CPUS
    config['train_batch_size'] = HORIZON * N_ROLLOUTS
    config['gamma'] = 0.999  # discount rate
    config['model'].update({'fcnet_hiddens': [32, 32]})
    config['use_gae'] = True
    config['lambda'] = 0.97
    config['kl_target'] = 0.02
    config['num_sgd_iter'] = 10
    config['clip_actions'] = False  # FIXME(ev) temporary ray bug
    config['horizon'] = HORIZON

    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    create_env, gym_name = make_create_env(params=flow_params, version=0)

    register_env(gym_name, create_env)
    return alg_run, gym_name, config


if __name__ == '__main__':
    alg_run, gym_name, config = setup_exps()
    ray.init(num_cpus=N_CPUS+1, num_gpus=0, redirect_output=False)
    trials = run_experiments({
        flow_params['exp_tag']: {
            'run': alg_run,
            'env': gym_name,
            'config': {
                **config
            },
            'checkpoint_freq': 10,
            'max_failures': 999,
            'stop': {
                'training_iteration': 10000,
            },
        }
    })
