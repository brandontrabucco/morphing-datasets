from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from multiprocessing import Process
from multiprocessing import set_start_method
import os
import gym
import numpy as np
import pickle as pkl
import argparse


def get_designs(num_designs,
                num_legs,
                domain='ant',
                method="uniform",
                verbose=True,
                noise_std=0.125):
    """Sample many designs uniformly from the design space
    specified by an agents morphology

    Arg:

    num_designs: int
        the number of designs that will be sampled from the design space
    num_legs: int
        the number of legs that will be sampled from the design space
    method: str in ["uniform", "centered"]
        the sampling method to use; centered is typically easier
    verbose: bool
        if true will the designs that error during sampling
    noise_std: float
        if method == "centered" the standard deviation of the noise

    Returns:

    designs: list
        a list of designs sampled from the design space of an agent
    """

    # check the target domain and import the appropriate agent
    if domain == "ant":
        from morphing_agents.mujoco.ant.env import MorphingAntEnv as Env
        from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN
        from morphing_agents.mujoco.ant.designs import sample_uniformly
        from morphing_agents.mujoco.ant.elements import LEG
        from morphing_agents.mujoco.ant.elements import LEG_UPPER_BOUND
        from morphing_agents.mujoco.ant.elements import LEG_LOWER_BOUND
    elif domain == "dog":
        from morphing_agents.mujoco.dog.env import MorphingDogEnv as Env
        from morphing_agents.mujoco.dog.designs import DEFAULT_DESIGN
        from morphing_agents.mujoco.dog.designs import sample_uniformly
        from morphing_agents.mujoco.dog.elements import LEG
        from morphing_agents.mujoco.dog.elements import LEG_UPPER_BOUND
        from morphing_agents.mujoco.dog.elements import LEG_LOWER_BOUND
    else:
        from morphing_agents.mujoco.dkitty.env import MorphingDKittyEnv as Env
        from morphing_agents.mujoco.dkitty.designs import DEFAULT_DESIGN
        from morphing_agents.mujoco.dkitty.designs import sample_uniformly
        from morphing_agents.mujoco.dkitty.elements import LEG
        from morphing_agents.mujoco.dkitty.elements import LEG_UPPER_BOUND
        from morphing_agents.mujoco.dkitty.elements import LEG_LOWER_BOUND

    ub = np.array(list(LEG_UPPER_BOUND))
    lb = np.array(list(LEG_LOWER_BOUND))
    scale = (ub - lb) / 2

    designs = []
    while len(designs) < num_designs:

        try:
            # sample designs blindly from the design space
            if method == 'uniform':
                d = sample_uniformly(num_legs=num_legs)

            # sample designs centered at a gold standard
            elif method == 'centered':
                d = [LEG(*np.clip(
                    np.array(leg) +
                    np.random.normal(0, scale / noise_std),
                    lb,
                    ub)) for leg in DEFAULT_DESIGN]

            # otherwise use the default design
            else:
                d = DEFAULT_DESIGN

            # check if that designed raised an error when built
            # some designs are invalid and checking them is non-trivial
            Env(fixed_design=d)
            designs.append(d)

        except:
            if verbose:
                print(f"resampling design that errored: {d}")

    return designs


def start_gpu_process(process_id,
                      gpu_id,
                      designs,
                      domain='ant',
                      data_folder="data/",
                      max_episode_steps=1000,
                      n_envs=4,
                      total_timesteps=1000000):
    """Train many policies using Proximal Policy Optimization on a set of
    morphologies and save training statistics

    Arg:

    process_id: int
        the integer id of the process running here
    gpu_id: int
        the integer id of the gpu allocated to this process
    designs: list
        a list of designs sampled from the design space of an agent
    data_folder: str
        the folder where training statistics are saved
    max_episode_steps: int
        the maximum number of steps per episode
    n_envs: int
        the number of parallel environments for sampling
    total_timesteps: int
        the number of environments steps to collect while training
    """

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    # check the target domain and import the appropriate agent
    if domain == "ant":
        env_name = "MorphingAntEnv"
    elif domain == "dog":
        env_name = "MorphingDogEnv"
    else:
        env_name = "MorphingDKittyEnv"

    for i, spec in enumerate(designs):

        # register my custom gym env
        gym.envs.register(
            id=f'{env_name}-{i}-v0',
            entry_point=f'morphing_agents.mujoco.{domain}.env:{env_name}',
            max_episode_steps=max_episode_steps,
            kwargs={'fixed_design': spec, 'expose_design': False})

        # multiprocess environment
        env = make_vec_env(f'{env_name}-{i}-v0', n_envs=n_envs)

        # make a ppo agent
        agent = PPO2(
            MlpPolicy, env,
            verbose=1,
            full_tensorboard_log=False,
            tensorboard_log=os.path.join(
                data_folder, f"p-{process_id}/{domain}-log-{i}"))

        # train for many gradient descent steps
        agent.learn(total_timesteps=total_timesteps)
        agent.save(os.path.join(
            data_folder, f"p-{process_id}/{domain}-log-{i}/weights"))

        # save the robot morphology in the design folder
        with open(os.path.join(
                data_folder, f"p-{process_id}/{domain}-log-{i}/design.pkl"),
                "wb") as f:
            pkl.dump(spec, f)


def split(some_list,
          parts):
    """Split a list into a number of smaller lists of approximately
    equal length
    """

    avg = len(some_list) / float(parts)
    out = []
    last = 0.0

    while last < len(some_list):
        out.append(some_list[int(last):int(last + avg)])
        last += avg

    return out


if __name__ == "__main__":

    # on startup ensure all processes are started using the spawn method
    # see https://github.com/tensorflow/tensorflow/issues/5448
    set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser('GenerateDataset')
    parser.add_argument('--local-dir',
                        type=str,
                        default='./data')
    parser.add_argument('--num-legs',
                        type=int,
                        default=4)
    parser.add_argument('--dataset-size',
                        type=int,
                        default=10)
    parser.add_argument('--num-parallel',
                        type=int,
                        default=1)
    parser.add_argument('--num-gpus',
                        type=int,
                        default=1)
    parser.add_argument('--n-envs',
                        type=int,
                        default=4)
    parser.add_argument('--max-episode-steps',
                        type=int,
                        default=1000)
    parser.add_argument('--total-timesteps',
                        type=int,
                        default=1000000)
    parser.add_argument('--method',
                        type=str,
                        default='uniform',
                        choices=['uniform', 'centered'])
    parser.add_argument('--domain',
                        type=str,
                        default='ant',
                        choices=['ant', 'dog', 'dkitty'])
    args = parser.parse_args()

    design_list = get_designs(args.dataset_size,
                              args.num_legs,
                              domain=args.domain,
                              method=args.method,
                              verbose=True,
                              noise_std=0.125)

    design_chunks = split(design_list,
                          args.num_parallel)

    ps = [Process(target=start_gpu_process,
                  args=(process_id,
                        process_id % args.num_gpus,
                        design_chunks[process_id]),
                  kwargs=dict(domain=args.domain,
                              data_folder=args.local_dir,
                              max_episode_steps=args.max_episode_steps,
                              n_envs=args.n_envs,
                              total_timesteps=args.total_timesteps))
          for process_id in range(args.num_parallel)]

    for p in ps:
        p.start()

    for p in ps:
        p.join()
