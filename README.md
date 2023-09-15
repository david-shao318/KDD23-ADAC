# A-DAC Traffic Signal Control

https://github.com/david-shao318/KDD23-ADAC/assets/57266876/569aef7e-f481-4211-aab1-31ac20cf8d62

## Offline Model-Based Reinforcement Learning for Traffic Signal Control

### Associated Paper

Mayuresh Kunjir, Sanjay Chawla, Siddarth Chandrasekar, Devika Jay, and Balaraman Ravindran. 2023. "Optimizing Traffic Control with Model-Based Learning: A Pessimistic Approach to Data-Efficient Policy Inference." In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '23)*. Association for Computing Machinery, New York, NY, USA, 1176–1187. [https://doi.org/10.1145/3580305.3599459](https://doi.org/10.1145/3580305.3599459)

### Abstract

Traffic signal control is an important problem in urban mobility
with a significant potential for economic and environmental impact.
While there is a growing interest in Reinforcement Learning (RL)
for traffic signal control, the work so far has focussed on learning
through simulations which could lead to inaccuracies due to
simplifying assumptions. Instead, real experience data on traffic is
available and could be exploited at minimal costs. Recent progress
in offline or batch RL has enabled just that. Model-based offline RL
methods, in particular, have been shown to generalize from the
experience data much better than others.
We build a model-based learning framework that infers a Markov
Decision Process (MDP) from a dataset collected using a cyclic
traffic signal control policy that is both commonplace and easy
to gather. The MDP is built with pessimistic costs to manage
out-of-distribution scenarios using an adaptive shaping of rewards
which is shown to provide better regularization compared to the
prior related work in addition to being PAC-optimal. Our model is
evaluated on a complex signalised roundabout and a large
multi-intersection environment, demonstrating that highly performant
traffic control policies can be built in a data-efficient manner.

### Installation
1. Use [environment.yml](./environment.yml) to create a conda environment. (For example, `conda env create --name ADAC --file=environment.yml`.)
2. Install [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) for traffic simulation. (Also visit [this page](https://sumo.dlr.de/docs/Installing/index.html).)
3. Run `python3 -m pip install .` to set up RESCO package.

### Getting Started

If properly installed, you should be able to run ADAC using a serialized model in a multi-intersection environment (Doha Corniche) with SUMO in a GUI. Run `python3 resco_adac_v4.0.py --eps=1 --gui=True`.

### Full Documentation

**Please read [DOCS.md](DOCS.md) for more detailed documentation.**

## Multi-Intersection

Testing a multi-intersection environment involves the [RESCO](https://github.com/Pi-Star-Lab/RESCO) benchmarking tool (J. Ault & G. Sharon, Reinforcement Learning Benchmarks for Traffic Signal Control, CC BY-NC-SA 4.0). We have adapted several files to run the ADAC agent (along with other agents) and have introduced a new map, `corniche`, modeled from the Doha Corniche.

We test primarily on the `cologne3`, `cologne8`, and custom-defined `corniche` environments part of our version of RESCO.

The current files and buffer data are configured to run the `corniche` environment at 10× traffic.

### Data

Use either the generated stochastic buffer [data](./resco_benchmark/Buffer/) or a separately generated buffer with [resco_buffer_generator.py](./resco_buffer_generator.py).

The pre-generated [buffer](./resco_benchmark/Buffer/) has 24 hours worth of stochastic data (using the `STOCHASTICWAVE` agent) between the times `1400` and `5000` in the `corniche` environment.

### Policy Building and Evaluation

After generating a compatible buffer or using the pre-generated buffer, use the configurable [resco_adac_v4.0.py](./resco_adac_v4.0.py) script to test various behavioral and RL algorithms, including `CYCLIC`, `STOCHASTIC`, `MAXPRESSURE`, `IDQN`, and ADAC, on standard RESCO maps and the corniche environment.

For training ADAC, set the `--agent` to the behavioral agent used to train the buffer data. Set `--which` to `ADAC` (set to `NotADAC` for all other policies). Set `--how` to the chosen method of considering neighboring intersections (`Nil` does not consider neighbors, `Average_Cat` takes the average of all neighbors and concatenates their state to each intersection's state, as described in the paper).

## Single-Intersection

The [ADAC](./ADAC_traffic_master/ADAC/) and [TrafQ](./ADAC_traffic_master/TrafQ/) folders contain the core functionality of ADAC. The [ADAC_traffic_master](./ADAC_traffic_master/) folder contains other files for testing a single-intersection environment using ADAC.

One standard single-intersection environment we test is based on Al Gharrafa roundabout in Doha.

### Data

Folder [ADAC_traffic_master/buffers](./ADAC_traffic_master/buffers/) provides a small data set collected from cyclic traffic signal control policy for the `gharrafa` environment.

To generate data sets with different sizes and behavioral policy, check the functionality provided in [run_offline_rl.py](./ADAC_traffic_master/run_offline_rl.py).

### Policy Building and Evaluation

Use the script [eval-dac-policies.sh](./ADAC_traffic_master/eval-dac-policies.sh) to test model-based offline RL solutions using the data set provided in folder buffers.

After generating the model files, use the script [test_ADAC_Model.py](./ADAC_traffic_master/test_ADAC_Model.py) to run a SUMO GUI simulation.

## Note on Pickling

On both the single- and multi-intersection environments, upon the completion of model training for ADAC, the models are automatically serialized into the [ADAC_traffic_master/pickled_ADAC](./ADAC_traffic_master/pickled_ADAC/) (for single-intersection) and [resco_benchmark/pickled_ADAC](./resco_benchmark/pickled_ADAC/) (for multi-intersection) folders respectively.

Without deleting the '.pickle' files, upon re-rerunning the same agent and configuration, the trained model is deserialized automatically. In this way, changes to, e.g., traffic scale and time can be made without retraining the models from buffer data. AnnoyIndex objects from Spotify's [Annoy](https://github.com/spotify/annoy) library are serialized and deserialized separately in '.ann' files.

_Please note that pickled files can execute arbitrary code. It may be advisable to remove all '.pickle' and '.ann' files to retrain the models from the buffer data._
