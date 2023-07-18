# Adaptive Deep Averagers with Costs (A-DAC)

Note that the scope of this documentation is limited to what is relevant to traffic
signal control in the paper. Refer to the paper as necessary.

This codebase makes use of code from two other sources:
1. [Benchmarking Batch Deep Reinforcement Learning Algorithms (for BCQ)](https://github.com/sfujim/BCQ)
2. [Reinforcement Learning Benchmarks for Traffic Signal Control (RESCO)](https://github.com/Pi-Star-Lab/RESCO)

Please refer to the documentation provided for troubleshooting the relevant pieces of code above.

Please refer to [SUMO Documentation](https://sumo.dlr.de/docs/Installing/index.html) for issues
with SUMO (Simulation of Urban Mobility).

## Troubleshooting SUMO

Please refer to [SUMO Documentation](https://sumo.dlr.de/docs/Installing/index.html) for installation
steps, including specific instructions for your OS.

Specifically, **macOS** users may run into issues with XQuartz, the application used for the GUI
of SUMO.
1. Add `export SUMO_HOME=/your/path/to/sumo` to .bashrc or .zshrc.
2. XQuartz requires permission to run in the background to automatically open when the GUI
for SUMO is triggered. (Requires logout/login.) You may also _manually open_ XQuartz.
3. A common display error may be fixed by resetting `export DISPLAY=:0.0` prior to opening XQuartz and
any SUMO command.

## A-DAC

The core ADAC algorithm is found in [ADAC_traffic_master/ADAC/dac_mdp.py](./ADAC_traffic_master/ADAC/dac_mdp.py),
which defines several important classes: `dac_builder`, `dac_policy`, and the `ANN` wrapper
for `AnnoyIndex` objects from the Spotify Annoy Library.

Running any ADAC agent requires several steps.

1. Construction of the buffer (an object of type `StandardBuffer` in all traffic control environments).
2. Filling the buffer with data. Each piece of data is in the form of a 5-tuple: (state, action, next_state, reward, done).
3. With the buffer and other configuration parameters, a `dac_builder` object is constructed, which processes the buffer,
creates a `dac_policy` object, builds the MDP stored by `dac_policy`, and solves it.
4. At this stage, we are done training ADAC. Calling `select_action` with a state tensor on the `dac_policy` will return
the action selected by ADAC.

### Testing ADAC with Manually Inputted Data

We can easily test the behavior of ADAC with a manually-defined dataset.
Simply run [ADAC_traffic_master/ADAC/dac_mdp.py](./ADAC_traffic_master/ADAC/dac_mdp.py),
specifying what data to put into the buffer in `state`, `action`, `next_state`, `reward`, and
`done`. Each of these should be a list of length n, where n is the number of data points. Each
`state` and `next_state` value should be an m-length list. Each value in `action` should be
an integer from 0 to k-1, where k is the number of possible actions. Each value in `reward`
should be a float denoting reward from a given data point. Each value in `done` is 0 or 1
to specify the end of a chain of actions.

The states given in `next_state` form the set of core states that we have experience data
from. Any other states that we wish to evaluate are non-core states, and may be specified in
`eval_state` in the same way `state` and `next_state` are defined.

The `cost` variable sets the penalty coefficient of the value calculation.
Setting to any value where `0 > cost >= -1`, such as -1, results in ADAC.

The variable `epsilon` specifies the desired convergence of the value iteration process.

#### Instructions

1. Set the buffer: `state`, `action`, `next_state`, `reward`, and `done`.
2. Set non-core states to evaluate: `eval_states`.
3. Set parameters.
4. Running [ADAC_traffic_master/ADAC/dac_mdp.py](./ADAC_traffic_master/ADAC/dac_mdp.py)
gives the values and actions on all core and specified non-core states, along with the calculated
rewards at each core state.

### Parameters of `dac_builder`

The `dac_builder` constructor takes several parameters:
1. `num_actions`: number of possible actions that the agent can take
2. `state_dim`: dimension of the state vector
3. `buffer`: a `StandardBuffer` object with the data buffer used to train ADAC
4. `q_model`: not in use
5. `device`: device to use for PyTorch (`'cpu'` or `'cuda'` if available; `'mps'` for Metal on Apple Silicon)
6. `num_configs`: configuration number in `[0, 1, 6]`, where 0 is manual specification of cost penalty, 1 is ADAC, and 6 tests
several combinations of `k` and `cost` values
7. `nn_mode`: either `'number'` or `'distance'`, where 'number' specifies that ADAC should find a certain _number_ (k) of nearest neighbors and
'distance' specifies that ADAC should find all nearest neighbors that are within some distance (10 at a time)
8. `diameter`: defaults to -1 to automatically calculate the diameter (the greatest distance between any two states);
otherwise, provide any positive number to manually specify diameter used in normalizing the distance in ADAC
9. `gamma`: discounting factor
10. `epsilon`: convergence limit
11. `cost`: provide any number where `0 > cost >= -1` for ADAC (set `num_configs = 1`); a positive number
specifies a manual cost value (set `num_configs = 0`)
12. `policy_number`: the policy to use if `num_configs = 6` and several combinations of `k` and `cost` are generated
13. `k`: if `nn_mode = 'number'`, then `k` is the number of nearest neighbors to find; if `nn_mode = 'distance'`, then
`k` is the maximum distance of neighbors to consider
14. `build_from_file` and `save_to_file`: specify a file location to serialize and deserialize `dac_policy` objects
15. `alpha`: when using the 'number' `nn_mode`, this is the coefficient on diameter for the
maximum distance to search for nearest neighbors
16. `get_exact_diameter`: when `diameter = -1`, the standard implementation finds an approximate diameter by
taking a random sample of states and comparing them to all states; setting this to `True` forces the consideration
of all possible pairs of states to find the true maximum distance

### Troubleshooting

#### Issues with Serialization and Deserialization (.pickle and .ann Files)

_Most issues with serialization can be resolved by deleting the relevant .pickle and .ann files. This
forces the reconstruction of all `dac_policy` objects._

We use Python's Pickle module to serialize `dac_policy` objects. This is done after the MDPs have been
solved, allowing us to quickly regenerate a previously-trained model. This model can be applied to any
other configuration, provided that the map is the same and state/action/reward representations remain the
same, without retraining from the buffer data.

When constructed, if a file prefix is provided, a `dac_builder` object attempts to rebuild all of its
`dac_policy` objects from the pickled version. If, for any given intersection, the pickled version
cannot be found, we create the MDP and solve it. If a file prefix is provided to save the model, then
each `dac_policy` (one for every intersection) is pickled in succession, saved to `PREFIX_i.pickle`, where i is
the index of the intersection.

You will notice that for every intersection, several `.ann` files may be generated alongside the `.pickle`
file. This is because the `AnnoyIndex` objects used for nearest-neighbor searches are not serializable
by default. These objects are stored separately, though they are stored and loaded automatically upon
unpickling the `dac_policy` object. To do so, however, the specific file location/name is stored within the
serialized `.ann` object.

For this reason, a `FileNotFoundError` appears when attempting to load a `.ann` file after it has been moved
to a different directory or had its file name changed. Avoid making such changes, and after doing so,
all pickled files should be deleted to allow the MDPs to be created, solved, and pickled again. 

#### Issue with Value Iteration (Cannot Cast Infinity to Integer)

The tool we use for value iteration is `hiive.mdptoolbox.mdp`. Occasionally, with manually specified
data, an exception appears when calculating the upper bound on number of iterations required to
achieve convergence within some epsilon. This method takes the span of values from two consecutive
iterations (line 1563), which may be `0.0`, leading to division by zero (processed as infinity).

In this case, you may include
```python
if span == 0.0:
    self.max_iter = 1000
    print('Value Iteration Bound Reset: self.max_iter = 1000')
    return
```
to sidestep the problem. Most likely, value iteration will only run one iteration before reaching
epsilon.

## Testing a Single-Intersection Map

To run a single-intersection map, such as gharrafa, using ADAC there are several general steps:
1. Generate a buffer using [ADAC_traffic_master/run_offline_rl.py](./ADAC_traffic_master/run_offline_rl.py)
for your specific map (Gym environment) using some behavioral policy.
2. Train and evaluate the model based on the buffer using
[ADAC_traffic_master/run_offline_rl.py](./ADAC_traffic_master/run_offline_rl.py) again, with different
configurations, however.
3. After training the model, if you want to rerun the same experiment without any
modifications to the model, use [test_ADAC_model.py](ADAC_traffic_master/test_ADAC_model.py),
which will load the model and run a GUI SUMO simulation of a _cyclic_ policy and your _ADAC_ policy.

**IMPORTANT 1:** Please run all the scripts in this section from the [ADAC_traffic_master](ADAC_traffic_master)
directory (`cd ADAC_traffic_master` beforehand).

**IMPORTANT 2:** Please also note that `run_offline_rl.py` is currently configured to _only_ run gharrafa.
Several pieces of code, such as line 427 (`eval_env`) default to `gharaffa`, regardless of what
is specified in `--env`. To run other environments, set `eval_env` to construct a new
Gym environment object and test further.

### Generate a Buffer

Run [run_offline_rl.py](./ADAC_traffic_master/run_offline_rl.py) and
configure `--generate_buffer` or `--generate_buffer_with_policy` to `True`. Configure `--buffer_name`
to a suitable name and `--max_timesteps` as necessary.

Note that `--generate_buffer` generates a buffer based on a custom policy that learns by
decreasing exploration over time, whereas `--generate_buffer_with_policy` is configured to
follow a specific `CustomPolicy`, currently a constant cycle policy. To use `--generate_buffer`,
first run `run_offline_rl.py` with `--train_behavioral`, which will generate a behavioral model in
the [models](ADAC_traffic_master/models) directory.

The buffer produced will be in the [buffers](ADAC_traffic_master/buffers)
directory and consist of six serialized numpy arrays, `state`, `action`, `next_state`, `reward`,
`not_done`, and `ptr`, that store all the experiences of the buffer generation. Here, `ptr` is an
index to locate the same tuple across all six arrays.

### Running an ADAC Simulation

The [buffers](ADAC_traffic_master/buffers) directory contains a buffer, labeled
`stationary-NTFT20`, for gharrafa based on a behavioral policy. Alternatively, follow the instructions
above to generate a new buffer.

Using this buffer name, run [run_offline_rl.py](./ADAC_traffic_master/run_offline_rl.py)
specifying `--offline_algo=BCQ` for the BCQ implementation, along with other parameters to align with
your buffer. A sample command can be found in [eval-dac-policies.sh](ADAC_traffic_master/eval-dac-policies.sh)
which can be run to train ADAC using the pre-generated buffer.
(In this case, please note that this relies on `run_offline_rl.py` being configured specifically to
gharrafa since the environment `gharaffa_NTFT20` is undefined in the current version. See IMPORTANT 2 above.
Ensure that any environment name used is registered with Gym in the [\_\_init\_\_.py](ADAC_traffic_master/TrafQ/Environments/gym_gharrafa/gymGharrafa/__init__.py).)
file of your gym environment. This is naturally also where you specify any kwargs for your Gym object.

Based on the `max_timesteps` argument, the training will occur for a given number of iterations.
Upon completion, a final evaluation is run in which the GUI is enabled on SUMO, providing
visualization of the results of training. Evaluations occur with different random seeds.

The best policy (BCQ) from training is automatically stored in the `models` directory for later use
if needed.

An important argument is the `--dac_configs` argument, which is pushed into the `num_configs` parameter
of `dac_builder`.. (See above for more information on the parameters of `dac_builder`.)

#### Instructions for Running Based on the `stationary-NTFT20` Buffer

Run the shell script [eval-dac-policies.sh](ADAC_traffic_master/eval-dac-policies.sh).

### Running a Fast Cyclic and ADAC Visual Comparison for Gharrafa

For quick access in a demo scenario, use [test_ADAC_model.py](ADAC_traffic_master/test_ADAC_model.py)
to load the model files generated by `eval-dac-policies.sh`. Upon loading, the `dac_policy` objects are
constructed, solved, and pickled (for future unpickling).

The script then runs two simulations: cyclic and BCQ ADAC, one after the other, on the same map and
scenario. This allows for ease of comparison.

Moreover, the serialized objects have been generated (in the [pickled_ADAC](ADAC_traffic_master/pickled_ADAC) directory),
running `python test_ADAC_model.py` allows you to very quickly generate a SUMO GUI simulation for
both the cyclic and ADAC policies.

#### Instructions for a Cyclic and ADAC Demo

In the [ADAC_traffic_master](ADAC_traffic_master) directory, run `python test_ADAC_model.py`.

There are two requirements:
* buffers matching `stationary-NTFT20_gharaffa-NTFT20_20` in the [buffers](ADAC_traffic_master/buffers)
directory. (Missing? Generate new buffer with this name, or edit the `test_ADAC_model.py` script.)
* model files matching `BCQ_gharaffa-NTFT20_20` in the [models](ADAC_traffic_master/models) directory.
  (Missing? Run `eval-dac-policies.sh`.)

If the pickled `dac_policy` is not present in [pickled_ADAC](ADAC_traffic_master/pickled_ADAC),
the object will be constructed, solved, and serialized for fast future unpickling.

### I Want to ...

#### Alter My Current Map or Simulation Configuration

For gharrafa, locate [tl.sumocfg](ADAC_traffic_master/TrafQ/Environments/gym_gharrafa/gymGharrafa/assets/tl.sumocfg)
(similar for other maps). Here, you can change the simulation begin and end time (limited by your traffic route data).
Replacing the `route-files` file allows you to run the simulation on new traffic data.
This setting will apply across the board. You do _not_ have re-generate the buffer or rerun ADAC. If you already
have the model files saved in [models](ADAC_traffic_master/models), directly run [test_ADAC_model.py](ADAC_traffic_master/test_ADAC_model.py)
for testing on the altered map.

Default SUMO simulation settings can be changed in the `GharrafaBasicEnv` class. In particular, see line 107, which
defines the `self.argslist` for the SUMO command. Arguments that can be added can be found in [SUMO documentation](https://sumo.dlr.de/docs/Basics/Using_the_Command_Line_Applications.html).

For one-off demos, many settings, such as visualization, traffic scale, and step delay, can be changed in the GUI.

#### Create a New Map

In [ADAC_traffic_master/TrafQ/Environments](ADAC_traffic_master/TrafQ/Environments), create a new directory for a custom Gym environment.
Within this new directory, create another directory with the name `gym{MAP_NAME}`, similar to
[ADAC_traffic_master/TrafQ/Environments/gym_gharrafa/gymGharrafa](ADAC_traffic_master/TrafQ/Environments/gym_gharrafa/gymGharrafa).

In this directory, write two new Python scripts: one to define your Gym environment's class, such
as [GharrafaBasicEnv.py](ADAC_traffic_master/TrafQ/Environments/gym_gharrafa/gymGharrafa/GharrafaBasicEnv.py),
which inherits from the `gym.Env` parent class. The other script is a module identifier `__init__.py`
to register your custom environment with Gym. Follow a similar format to [\_\_init\_\_.py](ADAC_traffic_master/TrafQ/Environments/gym_gharrafa/gymGharrafa/__init__.py).
Registration allows you to call `gym.make(ENV_NAME)` to create your environment based on custom
parameters. For example, `gym.make("gymGharrafa-v2")` results in the construction of a `GharrafaBasicEnv` object with
the arguments `'GUI': True, 'Play': "action"` (i.e., SUMO GUI is enabled and external code is allowed
to control the traffic lights through the `step(action)` method).

Also in this directory is the `assets` directory which includes all sumocfg, route, net, and other xml
files to define the specific SUMO map.

#### Create a New Policy for an Existing Map

For gharrafa, in [ADAC_traffic_master/gharaffaPolicies.py](ADAC_traffic_master/gharaffaPolicies.py), define a new class
that inherits the `CustomPolicy` base class, similar to the `gharaffaConstantCyclePolicy` class.
Implement `select_action` and `reset` accordingly.


## Testing a Multi-Intersection Map

In the multi-intersection scenario, we use RESCO to bundle together multiple traffic signals.
In running ADAC on maps like `cologne3`, `cologne8`, or `corniche`, multiple `dac_policy` objects
are bundled together, one for each intersection.

_FULL DOCUMENTATION TO BE COMPLETED_
