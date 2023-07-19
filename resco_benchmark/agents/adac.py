from resco_benchmark.agents.agent import IndependentAgent, Agent
import numpy as np

# THIS AGENT IS CURRENTLY NOT IN USE
# Not fully implemented

# RUN ADAC USING --which=ADAC ARGUMENT

class ADAC(IndependentAgent):
    def __init__(self, config, obs_act, map_name, thread_number):
        super().__init__(config, obs_act, map_name, thread_number)
        for key in obs_act:
            act_space = obs_act[key][1]
            self.agents[key] = ADACAgent(act_space)


class ADACAgent(Agent):
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions

    def act(self, observation):
        return  # TODO: move sample_action from resco_adac_4.0.py to here for closer RESCO integration

    def observe(self, observation, reward, done, info):
        pass
