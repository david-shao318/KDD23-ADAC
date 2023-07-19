import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from ADAC_traffic_master.ADAC.dac_mdp import dac_builder
from ADAC_traffic_master.ADAC.utils import ReplayBuffer
from resco_benchmark.config.agent_config import agent_configs
from resco_benchmark.config.map_config import map_configs
from resco_benchmark.config.mdp_config import mdp_configs
from resco_benchmark.config.signal_config import signal_configs
from resco_benchmark.multi_signal import MultiSignal

random_seeds = [i for i in range(3000, 4000, 5)]

for settings in [
    # ['STOCHASTIC', 'NotADAC', 'Nil'],
    # ['MAXPRESSURE', 'NotADAC', 'Nil'],
    # ['MAXWAVE', 'NotADAC', 'Nil'],
    # ['IDQN', 'NotADAC', 'Nil'],
    ['STOCHASTICWAVE', 'ADAC', 'Average_Cat'],
    # ['STOCHASTICWAVE', 'ADAC', 'Nil'],
]:

    s1, s2, s3 = settings

    time_per_setting_S = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", type=str, default=s1,
                    choices=['STOCHASTIC', 'MAXWAVE', 'MAXPRESSURE', 'IDQN', 'IPPO', 'MPLight', 'MA2C', 'FMA2C',
                             'MPLightFULL', 'FMA2CFull', 'FMA2CVAL', 'STOCHASTICWAVE', 'CYCLIC'])
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--eps", type=int, default=7)       # number of episodes (different random seed each time)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--map", type=str, default='corniche',
                    choices=['grid4x4', 'arterial4x4', 'ingolstadt1', 'ingolstadt7', 'ingolstadt21',
                             'cologne1', 'cologne3', 'cologne8', 'corniche'
                             ])
    ap.add_argument("--pwd", type=str, default='./resco_benchmark')
    ap.add_argument("--log_dir", type=str, default=os.path.join('./resco_benchmark', 'results' + os.sep))

    # SET TO TRUE FOR GUI
    ap.add_argument("--gui", type=bool, default=False)

    ap.add_argument("--libsumo", type=bool, default=False)
    ap.add_argument("--tr", type=int, default=0)  # can't multi-thread with libsumo, provide a trial number

    # configurations to enable ADAC and method of considering for neighboring intersections
    ap.add_argument("--which", type=str, default=s2, choices=['ADAC', 'NotADAC'])
    ap.add_argument("--how", type=str, default=s3, choices=['Nil', 'Average', 'Average_Cat'])

    # scaling factor for traffic in SUMO
    # (note that in the corniche environment, the traffic trip files only include taxi data)
    ap.add_argument("--traffic", type=int, default=10)

    # set > 0 to enforce a maximum wait time for all vehicles at an intersection (ADAC ONLY)
    # for example, if max_wait == 60, all vehicles will wait <= 60 seconds before its light turns green
    ap.add_argument("--max_wait", type=int, default=0)

    # set < 0 for ADAC, otherwise set to custom cost value
    ap.add_argument("--cost", type=float, default=-0.5)

    # set True to generate emissions file (use resco_benchmark/results/emissions_stats.ipynb to parse:
    # requires BeautifulSoup4)
    ap.add_argument("--emissions", type=bool, default=False)

    args = ap.parse_args()

    if args.libsumo and 'LIBSUMO_AS_TRACI' not in os.environ:
        raise EnvironmentError("Set LIBSUMO_AS_TRACI to nonempty value to enable libsumo")

    print('<<' * 10, f"{args.agent}, {args.eps} Episodes, {args.map}, {args.which}, {args.how}, Traffic {args.traffic}",
          '>>' * 10)


    ### BEGIN ADAC SPECIFIC

    ADAC_specific_S = time.time()
    if args.which == 'ADAC':

        # given an observation tensor, include neighboring states if --how is configured to Average or Average_Cat
        def process_obs(obs):
            if args.how != 'Nil' and args.which == 'ADAC':
                what_type = 'list'
                processed = []

                if type(obs) is dict:
                    what_type = 'dict'
                    processed = {}
                    obs = np.array([obs[jn_name] for jn_name in jn_names])

                for i in range(len(jn_names)):
                    jn_name = jn_names[i]
                    if args.how == 'Average':
                        neighbours = [i] + [n - 1 for n in neighbour_list[i]]
                        averaged = np.mean(obs[neighbours], 0)
                    elif args.how == 'Average_Cat':
                        neighbours = [n - 1 for n in neighbour_list[i]]
                        averaged = np.mean(obs[neighbours], 0)
                        averaged = np.hstack((obs[i], averaged))
                    else:
                        raise ValueError(f'Invalid argument {args.how} for --how')

                    if what_type == 'dict':
                        processed[jn_name] = averaged
                    else:
                        processed.append(averaged)

                if what_type == 'list':
                    processed = np.array(processed)

                return processed

            else:
                return obs


        # LOAD BUFFER FILES FOR ADAC

        jn_names = [intersection for intersection in signal_configs[args.map] if
                    intersection not in ['phase_pairs', 'valid_acts']]
        neighbour_list = []
        for jn_name in jn_names:
            neighbour_list.append(
                [jn_names.index(neighbour) for neighbour in
                 signal_configs[args.map][jn_name]['downstream'].values()
                 if neighbour is not None]
            )

        all_states = []
        all_next_states = []
        all_rewards = []
        all_actions = []
        all_dones = []

        buffer_path = './resco_benchmark/Buffer/'

        for jn_name in jn_names:
            print('Loading', jn_name, 'files.....')

            fname = buffer_path + jn_name + '_state.npy'
            c = np.load(fname)
            all_states.append(c)

            fname = buffer_path + jn_name + '_next_state.npy'
            c = np.load(fname)
            all_next_states.append(c)

            fname = buffer_path + jn_name + '_reward.npy'
            c = np.load(fname)
            c = [i[0] for i in c]  # + np.random.randn()
            all_rewards.append(c)

            fname = buffer_path + jn_name + '_action.npy'
            c = np.load(fname)
            c = [i[0] for i in c]
            all_actions.append(c)

            fname = buffer_path + jn_name + '_not_done.npy'
            c = np.load(fname)
            c = [i[0] for i in c]
            all_dones.append(c)

        all_states = np.array(all_states)
        all_next_states = np.array(all_next_states)
        all_rewards = np.array(all_rewards)
        all_actions = np.array(all_actions)
        all_dones = np.array(all_dones)

        all_states = process_obs(all_states)
        all_next_states = process_obs(all_next_states)

        # construct MDPs for ADAC
        all_policies = {}

        ADAC_build_MDP_S = time.time()

        for states, actions, next_states, rewards, dones, jn_name in zip(all_states, all_actions, all_next_states,
                                                                         all_rewards, all_dones, jn_names):
            print('Building', jn_name, 'MDPs.....')

            num_actions = len(np.unique(actions))
            state_dim = states.shape[-1]
            buffer = ReplayBuffer(state_dim=state_dim,
                                  is_atari=False,
                                  atari_preprocessing=None,
                                  batch_size=128,
                                  buffer_size=len(states),
                                  device='cpu')
            for (s, a, n, r, d) in zip(states, actions, next_states, rewards, dones):
                buffer.add(s, a, n, r, d, 0, 0)

            # set to 1 for ADAC (determine cost based on max reward)
            # set to 0 for manually configured DAC (manually specified cost)
            specific_config = 1 if args.cost < 0 else 0

            dac = dac_builder(num_actions,
                              state_dim,
                              buffer,
                              None,
                              'cpu',
                              nn_mode='number',
                              gamma=0.99,
                              num_configs=specific_config,
                              cost=args.cost,
                              save_to_file=f'./resco_benchmark/pickled_ADAC/{args.map}_{args.agent}_traffic{str(args.traffic)}_cost{"ADAC" if args.cost < 0 else args.cost}_{args.how}_{jn_name}',
                              build_from_file=f'./resco_benchmark/pickled_ADAC/{args.map}_{args.agent}_traffic{str(args.traffic)}_cost{"ADAC" if args.cost < 0 else args.cost}_{args.how}_{jn_name}')
            policies = dac.get_policies()[0]
            all_policies[jn_name] = policies

            # sample action, given observation dict
            def sample_action(obs):
                obs = process_obs(obs)
                action = {}
                for jn_name in jn_names:
                    action[jn_name] = all_policies[jn_name].select_action(obs[jn_name])
                return action

        ADAC_build_MDP_E = time.time()

    ADAC_specific_E = time.time()

    ### END ADAC SPECIFIC



    ### RUN RESCO TRIAL

    def run_trial(args, trial):
        mdp_config = mdp_configs.get(args.agent)
        if mdp_config is not None:
            mdp_map_config = mdp_config.get(args.map)
            if mdp_map_config is not None:
                mdp_config = mdp_map_config
            mdp_configs[args.agent] = mdp_config

        agt_config = agent_configs[args.agent]
        agt_map_config = agt_config.get(args.map)
        if agt_map_config is not None:
            agt_config = agt_map_config
        alg = agt_config['agent']

        if mdp_config is not None:
            agt_config['mdp'] = mdp_config
            management = agt_config['mdp'].get('management')
            if management is not None:  # Save some time and precompute the reverse mapping
                supervisors = dict()
                for manager in management:
                    workers = management[manager]
                    for worker in workers:
                        supervisors[worker] = manager
                mdp_config['supervisors'] = supervisors

        map_config = map_configs[args.map]
        num_steps_eps = int((map_config['end_time'] - map_config['start_time']) / map_config['step_length'])
        route = map_config['route']
        if route is not None: route = os.path.join(args.pwd, route)
        if args.map == 'grid4x4' or args.map == 'arterial4x4':
            if not os.path.exists(route): raise EnvironmentError(
                "You must decompress environment files defining traffic flow")

        if args.which == 'ADAC':
            path = args.how + '_' + args.which + '_traffic' + str(args.traffic) + '-tr' + str(trial)
        else:
            path = alg.__name__ + '_traffic' + str(args.traffic) + '-tr' + str(trial)

        env = MultiSignal(path,
                          args.map,
                          os.path.join(args.pwd, map_config['net']),
                          agt_config['state'],
                          agt_config['reward'],
                          route=route, step_length=map_config['step_length'], yellow_length=map_config['yellow_length'],
                          step_ratio=map_config['step_ratio'], end_time=map_config['end_time'],
                          max_distance=agt_config['max_distance'], lights=map_config['lights'], gui=args.gui,
                          log_dir=args.log_dir, libsumo=args.libsumo, warmup=map_config['warmup'],
                          emissions=args.emissions)

        agt_config['episodes'] = int(args.eps * 0.8)  # schedulers decay over 80% of steps
        if agt_config['episodes'] <= 0 and args.agent in ['IDQN', 'IPPO', 'MPLight', 'MA2C', 'FMA2C', 'MPLightFULL',
                                                          'FMA2CFull', 'FMA2CVAL']:
            raise ValueError("agt_config['episodes'] should be greater than 0. Set args.eps >= 2.")
        agt_config['steps'] = agt_config['episodes'] * num_steps_eps
        agt_config['log_dir'] = os.path.join(args.log_dir, env.connection_name)
        agt_config['num_lights'] = len(env.all_ts_ids)

        # Get agent id's, observation shapes, and action sizes from env
        obs_act = dict()
        for key in env.obs_shape:
            obs_act[key] = [env.obs_shape[key], len(env.phases[key]) if key in env.phases else None]
        agent = alg(agt_config, obs_act, args.map, trial)

        movements = ['S-W', 'S-S', 'S-E', 'W-N', 'W-W', 'W-S', 'N-E', 'N-N', 'N-W', 'E-S', 'E-E', 'E-N']

        for e in range(args.eps):
            obs = env.reset(seed=random_seeds[e], traffic_scale=args.traffic)
            done = False
            printed = False
            while not done:
                if args.which == 'ADAC':
                    act = sample_action(obs)

                    # include w parameter for MAX WAIT TIME for any vehicle
                    if args.max_wait != 0:
                        longest_wait_lanes = env.lane_with_max_wait(args.max_wait)
                        for ts in longest_wait_lanes:
                            if longest_wait_lanes[ts] is not None:
                                direction = None
                                for k, v in signal_configs[args.map][ts]['lane_sets'].items():
                                    if longest_wait_lanes[ts] in v:
                                        direction = k
                                assert direction != ''
                                movement_index = movements.index(direction)
                                phases_with_movement = [i for i in range(len(signal_configs[args.map]['phase_pairs']))
                                                        if movement_index in signal_configs[args.map]['phase_pairs'][i]]
                                for p in reversed(phases_with_movement):
                                    if p in signal_configs[args.map]['valid_acts'][ts]:
                                        p_id = signal_configs[args.map]['valid_acts'][ts][p]
                                        # print(f'MAX WAIT TIME OVERRIDE: {act[ts]} -> {p_id}')
                                        act[ts] = p_id

                    # logging
                    if e == 0 and printed is False:
                        print()
                        print(args.which, args.how)
                        print()
                        printed = True
                else:
                    act = agent.act(obs)
                obs, rew, done, info = env.step(act)
                agent.observe(obs, rew, done, info)
        env.close()

        return path


    # RUN TRIALS
    if args.procs == 1 or args.libsumo:
        path = run_trial(args, args.tr)
    else:
        pool = mp.Pool(processes=args.procs)
        for trial in range(1, args.trials + 1):
            pool.apply_async(run_trial, args=(args, trial))
        pool.close()
        pool.join()

    time_per_setting_E = time.time()

    # STATS
    if args.procs == 1 or args.libsumo:
        with open(args.log_dir + 'stats/' + path + 'stats_rebuttal.txt', 'w') as file:
            file.write('Total test run hours:' + str(args.eps) + '\n')
            file.write('Total run time for this setting:' + str(time_per_setting_E - time_per_setting_S) + '\n')
            file.write('ADAC specific time (loading + building MDPs):' + str(ADAC_specific_E - ADAC_specific_S) + '\n')
            if args.which == 'ADAC':
                file.write('Time taken to build ADAC MDPs (only building MDPs):' + str(
                    ADAC_build_MDP_E - ADAC_build_MDP_S) + '\n')
