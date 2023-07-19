# Adaptive Deep Averagers with Costs (A-DAC)

Note that the scope of this documentation is limited to what is relevant to traffic
signal control in the paper. Refer to the paper as necessary.

This codebase makes use of code from two other sources:
1. [Benchmarking Batch Deep Reinforcement Learning Algorithms (for BCQ)](https://github.com/sfujim/BCQ)
2. [Reinforcement Learning Benchmarks for Traffic Signal Control (RESCO)](https://github.com/Pi-Star-Lab/RESCO)

Please refer to the documentation provided for troubleshooting the relevant pieces of code above.

Please refer to [SUMO Documentation](https://sumo.dlr.de/docs/Installing/index.html) for issues
with SUMO (Simulation of Urban MObility).

## Table of Contents

1. [Troubleshooting SUMO](#troubleshooting-sumo)
2. [A-DAC](#a-dac)
   1. [Manually Inputted Data](#testing-adac-with-manually-inputted-data)
   2. [`dac_builder` Construction](#parameters-of-dac_builder)
   3. [Troubleshooting](#troubleshooting)
      1. [Serialization (Pickle)](#issues-with-serialization-and-deserialization-pickle-and-ann-files)
      2. [Value Iteration Error](#issue-with-value-iteration-cannot-cast-infinity-to-integer)
3. [Single-Intersection Testing](#testing-a-single-intersection-map)
   1. [Buffer Generation](#generating-a-buffer)
   2. [Running A-DAC](#running-an-adac-simulation)
   3. [Running a Visual Demo: Cyclic and A-DAC](#running-a-fast-cyclic-and-adac-visual-comparison-for-gharrafa)
   4. [How to](#how-to-)
      1. [Alter My Current Map or Simulation Configuration](#alter-my-current-map-or-simulation-configuration)
      2. [Create a New Map](#create-a-new-map)
      3. [Create a New Policy](#create-a-new-policy-for-an-existing-map)
4. [Multi-Intersection Testing](#testing-a-multi-intersection-map)
   1. [Buffer Generation](#generating-a-multi-intersection-buffer)
   2. [Non-ADAC](#running-a-non-adac-policy-agent-on-resco)
   3. [ADAC](#running-adac-or-dac-on-resco)
   4. [Altering Simulation Configuration](#modifying-sumo-simulation-settings-including-visualization)
   5. [Creating or Changing Policy Agents, States, or Rewards](#modifying-or-defining-a-new-policy-agent-state-representation-or-reward-representation)
   6. [Creating a New Map](#defining-a-new-map)

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
4. At this stage, we are done training ADAC. Calling `select_action` with a state vector on the `dac_policy` will return
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

### Generating a Buffer

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

_Please note that buffer generation may not overwrite a prior buffer._

### Running an ADAC Simulation

The ADAC implementation for a single-intersection allows the user to test multiple offline algorithm classes, such as
BCQ (Batch-Constrained Q-Learning). The default is to use BCQ.

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

The best policy from training is automatically stored (as a BCQ model) in the `models` directory for later use
if needed.

An important argument is the `--dac_configs` argument, which is pushed into the `num_configs` parameter
of `dac_builder`. (See above for more information on the parameters of `dac_builder`.)

#### Instructions for Running Based on the `stationary-NTFT20` Buffer

Run the shell script [eval-dac-policies.sh](ADAC_traffic_master/eval-dac-policies.sh).

### Running a Fast Cyclic and ADAC Visual Comparison for Gharrafa

For quick access in a demo scenario, use [test_ADAC_model.py](ADAC_traffic_master/test_ADAC_model.py)
to load the model files generated by `eval-dac-policies.sh`. Upon loading, the `dac_policy` objects are
constructed, solved, and pickled (for future unpickling).

The script then runs two simulations: cyclic and BCQ ADAC, one after the other, on the same map and
scenario. This allows for ease of comparison.

Moreover, once the serialized objects have been generated (in the [pickled_ADAC](ADAC_traffic_master/pickled_ADAC) directory),
running `python test_ADAC_model.py` allows you to very quickly generate a SUMO GUI simulation for
both the cyclic and ADAC policies.

#### Instructions for a Cyclic and ADAC Demo

In the [ADAC_traffic_master](ADAC_traffic_master) directory, run `python test_ADAC_model.py`.

There are two requirements:
* Buffers matching `stationary-NTFT20_gharaffa-NTFT20_20` in the [buffers](ADAC_traffic_master/buffers)
directory.
  * Missing? Generate new buffer with this name, or edit the `test_ADAC_model.py` script.
* Model files matching `BCQ_gharaffa-NTFT20_20` in the [models](ADAC_traffic_master/models) directory. 
  * Missing? Run `eval-dac-policies.sh`.

If the pickled `dac_policy` is not present in [pickled_ADAC](ADAC_traffic_master/pickled_ADAC),
the object will be constructed, solved, and serialized for fast future unpickling.

### How to ...

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
which inherits from the `gym.Env` parent class. Here, the possible phases along with monitored edges (lanes) must be
manually identified for the intersection.

The other script is a module identifier `__init__.py` to register your custom environment with Gym.
Follow a similar format to [\_\_init\_\_.py](ADAC_traffic_master/TrafQ/Environments/gym_gharrafa/gymGharrafa/__init__.py).
Registration allows you to call `gym.make(ENV_NAME)` to create your environment based on custom
parameters. For example, `gym.make("gymGharrafa-v2")` results in the construction of a `GharrafaBasicEnv` object with
the arguments `'GUI': True, 'Play': "action"` (i.e., SUMO GUI is enabled and external code is allowed
to control the traffic lights through the `step(action)` method).

Also in this directory is the `assets` directory which includes all sumocfg, route, net, and other xml
files to define the specific SUMO map.

**N.B.** For gharrafa, there are two separate environments, the `GharrafaBasicEnv` class and the `GharrafaEnv` class
(found in [ADAC_traffic_master/gharaffaEnv.py](ADAC_traffic_master/gharaffaEnv.py)). The latter is a modified copy of
the former. The latter is the environment used in the [run_offline_rl.py](ADAC_traffic_master/run_offline_rl.py) script,
whereas the former is used in the [test_ADAC_model.py](ADAC_traffic_master/test_ADAC_model.py) script.

#### Create a New Policy for an Existing Map

For gharrafa, in [ADAC_traffic_master/gharaffaPolicies.py](ADAC_traffic_master/gharaffaPolicies.py), define a new class
that inherits the `CustomPolicy` base class, similar to the `gharaffaConstantCyclePolicy` class.
Implement `select_action` and `reset` accordingly.

## Testing a Multi-Intersection Map

In the multi-intersection scenario, we use RESCO to bundle together multiple traffic signals.
When running ADAC on maps like `cologne3`, `cologne8`, or `corniche`, multiple `dac_policy` objects
are bundled together, one for each intersection.

Note that ADAC is not a distinct policy agent, like `MAXPRESSURE`, `CYCLIC`, or `STOCHASTIC`, but instead overlays a
"base agent" that was used to generate the buffer data. When running ADAC on RESCO, a `dac_policy` object is constructed
and its MDP solved for each intersection, which provides the `select_action` method to choose an action based on a state.

The two general steps to running ADAC are largely the same:
1. Generate a buffer using [resco_buffer_generator.py](resco_buffer_generator.py) by selecting a behavioral policy.
2. Run [resco_adac_v4.0.py](resco_adac_v4.0.py) with the correct configurations. This will process the buffer and create
the ADAC policy, and run a number of simulations with SUMO under different random seeds.

To run a non-ADAC policy, skip the first step.

**N.B.** By default, GUI is false and SUMO default statistics are printed instead. Set `--gui` to true while running
[resco_adac_v4.0.py](resco_adac_v4.0.py) to see a GUI for each trial.

### Configuring Current SUMO Environments

Each map is defined by its own directory in the [resco_benchmark/environments](resco_benchmark/environments) directory.
Within the directory are all the SUMO files, including the sumocfg, net, and route files. The net file defines the map
itself. The route file defines the traffic routing in the simulation (this can be replaced with different traffic
data). Both files are referenced in the sumocfg file. The sumocfg file also defines start and end time, along with
the smallest unit of time in the simulation (1 for one update per simulated second). See, for example,
[resco_benchmark/environments/corniche/corniche_base.sumocfg](resco_benchmark/environments/corniche/corniche_base.sumocfg).

The second location where maps are defined is the [resco_benchmark/config/map_config.py](resco_benchmark/config/map_config.py)
file, which stores the `map_configs` dictionary of map settings. Here, the location of the sumocfg file needs to be
provided, along with start and end times that match the sumocfg file. This is also where `step_length` and `yellow_length`
are defined: `step_length` provides the duration of time between each signal update step (10 seconds by default), while
`yellow_length` defines the length of yellow light times that occur between all green to red signal changes.

Note that setting the start and end times are global for all SUMO simulations, including buffer generation. By default,
the `corniche` environment only runs one hour from the time `1400` to `5000` (in seconds). This means that the buffer
is only generated on this time, and running ADAC is only done over this one hour (however, the traffic is still varied
due to different random seeds.) It is, of course, appropriate to generate the buffer for ADAC using a certain time configuration
and run and test ADAC on a completely different time configuration. The only things that must be kept constant are
the map and road network, along with state and action representations. 
You do _not_ have re-generate the buffer or rerun ADAC. If you already
have the `dac_policy` objects pickled, directly run [resco_adac_v4.0.py](resco_adac_v4.0.py) for testing on the altered map.

For `corniche`, there is traffic data in the route file for up to 24 hours (`0` to `86400`).

### Generating a Multi-Intersection Buffer

Using [resco_buffer_generator.py](resco_buffer_generator.py), a buffer of a certain size can be generated for a specific
map according to a behavioral policy.

Simply run the script using certain arguments (below) and buffer files will be generated in the
[resco_benchmark/Buffer](resco_benchmark/Buffer) directory, consisting of six serialized numpy arrays for each
intersection (the name of the intersection is placed at the start of each file name).

**N.B.** Buffer generation does not currently store the exact map, policy, or any other configuration used in the file
names. For this reason, prior to generating a new buffer, remove the old buffer from the [resco_benchmark/Buffer](resco_benchmark/Buffer)
directory. _Note also that buffer generation may not overwrite a prior buffer._

The pre-generated buffer in this repository is for `corniche` and comes from 24 episodes of `STOCHASTICWAVE` of one hour
duration (`1400` to `5000`) each, with random seeds `[100, 102, 104, ..., 146]`.

#### Script Arguments for RESCO Buffer Generation

1. `--agent`: behavioral policy used in buffer generation. See RESCO documentation for how most of these agents are
implemented. There are two custom agents: `CYCLIC`, which behaves according to the default traffic cycle (defined in
the .net.xml file of the map) and changes every 2 signal change steps (2 * 10 seconds = 20 seconds by default); and
`STOCHASTICWAVE`, which behaves identically to `STOCHASTIC` (random actions) but uses the `states.wave` representation
to store each state (found in [resco_benchmark/states.py](resco_benchmark/states.py)) instead of the standard
`states.mplight`. See below for how to modify and create new agents.
2. `--eps`: the number of episodes to generate the buffer for. Each episode is one complete simulation from start to end
time, as defined in map_config.py and the sumocfg file. With each new episode, a new random seed is used for some
variation in traffic, though the general trend of traffic remains the same (e.g., rush hour still looks like rush hour,
as defined in the route file, irrespective of the seed used). Currently, the script can only handle up to 450 episodes,
but this is easily expanded by changing line 111 to initialize more values in the `random_seeds` list.
3. `--map`: map to run. See below for how to create new maps.
4. `--gui`: specifies if you would like to see buffer generation in SUMO's GUI.
5. `--traffic`: an integer that scales traffic from the base traffic level defined by the environment's route file. Note
that the traffic data for `corniche` was collected only through taxis, so scaling the traffic by a factor of at least 10
more closely simulates reality. Ideally, when running ADAC, the traffic level used to collect the buffer data should be
similar to the traffic used when testing the resulting model.
6. `--emissions`: a boolean denoting whether SUMO should generate an emissions file for each episode, which
denotes the emissions output of every single vehicle in the simulation at every time step. _Note that these files can
easily take up several GB of space each, depending on the amount of traffic._ The resulting emissions file can be found
in the [resco_benchmark/results](resco_benchmark/results) directory, under the directory matching your configurations.
The [emissions_stats.ipynb](resco_benchmark/results/emissions_stats.ipynb) notebook provided allows you to sum the CO2
output from all the vehicles.

Several arguments are not covered above (such as those for manual multithreading) and do not pertain to the core
functionality of the buffer generation procedure. 

### Running a Non-ADAC Policy Agent on RESCO

Run the [resco_adac_v4.0.py](resco_adac_v4.0.py) script using either custom arguments or by specifying the core settings
in a `settings` list (lines 18 to 25) to run several settings consecutively. If using a `settings` list, note that
the settings should be given in the form of a triple: `[agent, which, how]`.

Please see the section directly above for descriptions of the standard RESCO arguments. There are several other
ADAC-only parameters that are relevant:

1. `--which`: for non-ADAC, set to `"NotADAC"`.
2. `--how`: for non-ADAC, always set to `"Nil"`.

For running a non-ADAC agent, keep all other arguments not listed here or [above](#script-arguments-for-resco-buffer-generation) at their default values.

Running a Non-ADAC agent on RESCO behaves in the same way as buffer generation, though no buffer files will be generated.

At some point, you may encounter a `ValueError` with the message
`agt_config['episodes'] should be greater than 0. Set args.eps >= 2.`. This occurs when running certain policy agents
that use the `SharedAgent` class, such as `IDQN`, and you have set `--eps=1`. Due to a calculation involving scaling
the number of episodes by `0.8`, the number of episodes configured for the agent would be `0`. Instead, set `--eps=2`.

#### Example: Running `MAXPRESSURE` on `corniche`

Suppose we wanted to run `MAXPRESSURE` on `corniche` using 15x traffic with GUI turned on for 3 episodes using the
seeds `[12, 24, 36]`.

1. In line 16, edit `random_seeds = [12, 24, 36]`.
2. Ensure the for loop in lines 18 to 25 only runs once (include only one setting; this does not have to be
`["MAXPRESSURE", "NotADAC", "Nil"]` since we are overriding the coded setting using our command arguments).
3. In the command line, we would run
`python resco_adac_v4.0.py --agent=MAXPRESSURE --eps=3 --map=corniche --gui=True --which=NotADAC --how=Nil --traffic=15`.

### Running ADAC (or DAC) on RESCO

Run the [resco_adac_v4.0.py](resco_adac_v4.0.py) script using either custom arguments or by specifying the core settings
in a `settings` list (lines 18 to 25) to run several settings consecutively.

See [above](#script-arguments-for-resco-buffer-generation) for information on standard RESCO arguments.
Several parameters are ADAC-specific and are given below.

1. `--agent`: set to the same agent used to generate the buffer.
2. `--which`: for ADAC or DAC, set to `"ADAC"`.
3. `--how`: set to `"Nil"`, `"Average"`, or `"AverageCat"` for how the model should consider **neighboring intersections**.
Nil does not consider an intersection's neighbors in its own state. Average takes the average state of the intersection
and all of its neighbors as the state vector for the intersection. AverageCat takes the average state of all of an
intersection's neighbors and concatenates this average neighbor state to the state vector of the intersection.
4. `--max_wait`: the maximum wait time in seconds that any vehicle can wait at an intersection before its light
turns green (note that this is only enforced when `--which=ADAC`).
5. `--cost`: set to a value such that `-1 <= cost < 0` for ADAC. Otherwise, set to a positive value to run DAC on a
specific cost value, e.g., `C=2`.

When running ADAC on RESCO, the first step is to load the buffer files into ADAC. (Note that even when the DAC model
is serialized, a buffer is still necessary.)

At this point, the buffer is processed and MDPs (`dac_policy` objects) are constructed using the `dac_builder`. Here,
the settings for building the MDPs are specified, including `device`, `gamma`, and where to load or save the serialized
`dac_policy` objects. By default, the pickled `dac_policy` objects are stored in the [resco_benchmark/pickled_ADAC](resco_benchmark/pickled_ADAC)
directory.

When running the trials, each intersection is processed with its state vector, and returns some action selected (given
the possible actions for each intersection) as an index from `0` to `n_i - 1`, where `n_i` is the number of possible
actions for intersection `i`.

The random seeds used for each episode are currently hard-coded to be `[3000, 3005, ..., 3995]`. For proper testing,
these values should be distinct from the seeds used in buffer generation.

#### Example: Running ADAC (Average_Cat) on `corniche` (10x Traffic) Using a `STOCHASTICWAVE` Buffer with GUI Enabled

1. Run `python resco_adac_v4.0.py --agent=STOCHASTICWAVE --eps=1 --map=corniche --gui=True --which=ADAC --how=Average_Cat
--traffic=10`.
2. The script expects a buffer that corresponds to `STOCHASTICWAVE` on `corniche` in the buffer directory.
3. Prior to building each `dac_policy` object (MDP), the `dac_builder` will look for a .pickle file with the name
`corniche_STOCHASTICWAVE_traffic10_costADAC_Average_Cat_{INTERSECTIONLABEL}.pickle` in the
[resco_benchmark/pickled_ADAC](resco_benchmark/pickled_ADAC) directory, along with corresponding .ann files if the
pickle file is found. This will be done for each intersection. If any one of these serialized object files is not found,
the procedure is to build the MDP, solve the MDP using value iteration, and pickle it.
4. Upon the completion of MDP building, we are ready to begin the simulation trials. Ensure `--gui=True` to see the
GUI. Otherwise, some brief statistics will be printed at the end of each episode.

### Modifying SUMO Simulation Settings (Including Visualization)

Note that several settings, such as start and end time or simulation step length, are controlled in the .sumocfg and
`map_config.py` files for each map individually.

Default SUMO simulation settings can be changed in the file [resco_benchmark/multi_signal.py](resco_benchmark/multi_signal.py).
In particular, see lines 133 to 141, which define the arguments for the SUMO command.
Arguments that can be added can be found in [SUMO documentation](https://sumo.dlr.de/docs/Basics/Using_the_Command_Line_Applications.html).

For one-off demos, many settings, such as visualization, traffic scale, and step delay, can be changed in the GUI.

### Modifying or Defining a New Policy Agent, State Representation, or Reward Representation

The general steps to define a new policy agent is to create a new .py file containing a new class that inherits either
from `IndependentAgent` or `SharedAgent` in the [resco_benchmark/agents](resco_benchmark/agents) directory. This
should initialize individual agents (another class inheriting from the `Agent` base class) for each intersection.
In the individual agent class, `observe` and `act` should be defined to process new states and select an action.
See [resco_benchmark/agents/cyclic.py](resco_benchmark/agents/cyclic.py) for an example.

This is also where modifications to policy agents can be made. For example, to change cyclic behavior to only change the
cycle every 4 signal update steps (10 seconds each by default), change line 20 of [resco_benchmark/agents/cyclic.py](resco_benchmark/agents/cyclic.py)
to `return (self.counter // 4) % self.num_actions`.

The signal update step length, on the other hand, must be updated in the `map_configs` dictionary. See
[above](#configuring-current-sumo-environments).

State representations (how states are represented in all of RESCO) are defined in
[resco_benchmark/states.py](resco_benchmark/states.py). Each agent chooses which state to use according to
[resco_benchmark/config/agent_config.py](resco_benchmark/config/agent_config.py).

Likewise, [resco_benchmark/config/agent_config.py](resco_benchmark/config/agent_config.py) also defines which reward
representation to use for each agent. Reward representations can be modified and created in
[resco_benchmark/rewards.py](resco_benchmark/rewards.py)

See RESCO documentation for more details.

### Defining a New Map

Creating a new map in RESCO is a tedious task. The first step is to store the SUMO files (including sumocfg, net, route, and
other xml files) in a separate directory within [resco_benchmark/environments](resco_benchmark/environments). Then add a new
item to `map_configs` in [resco_benchmark/config/map_config.py](resco_benchmark/config/map_config.py) to point to the sumocfg
file, along with other configurations.

The next step is to edit [resco_benchmark/config/signal_config.py](resco_benchmark/config/signal_config.py) to include
information regarding valid phase pairs (two traffic movements green at the same time) for all intersections, along with
the individual phase pair actions that are valid for each intersection. The lanes corresponding to each traffic movement
must also be specified by name in the `lane_sets` list. The neighboring intersections (used for, e.g., `Average_Cat` in
ADAC) are defined by cardinal directions from each intersection.

See RESCO documentation, in particular [Environment Configuration.md](resco_benchmark/config/Environment%20Configuration.md), for more details.

The way to interpret traffic movements is to consider which way a vehicle is face when entering and leaving an intersection.
For example, N-W is a left turn (a vehicle enters Northbound, but leaves Westbound, making a left turn in the intersection).
E-E is interpreted as going straight across the intersection (starting from the West side, going East across).

#### Limitations of RESCO

Note that there are 12 possible traffic movements:
'S-W', 'S-S', 'S-E', 'W-N', 'W-W', 'W-S', 'N-E', 'N-N', 'N-W', 'E-S', 'E-E', 'E-N'.
In other words, there is no configuration in RESCO for U-turns (e.g., 'S-N' or 'E-W').
In the `corniche` environment U-turns are not individually considered in the model but are instead bundled into left
turns.

Moreover, the phase pair system only allows two traffic movements to be green at the same time, so it is not possible
for the model to consider three traffic movements at the same time (for example, N-N + N-W + E-S simultaneously).
