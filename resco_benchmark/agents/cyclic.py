from resco_benchmark.agents.agent import IndependentAgent, Agent


class CYCLIC(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            act_space = obs_act[key][1]
            self.agents[key] = CYCLICAgent(act_space)


class CYCLICAgent(Agent):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        self.counter = 0

    def act(self, observation):
        self.counter += 1
        return (self.counter // 2) % self.num_actions

    def observe(self, observation, reward, done, info):
        pass
