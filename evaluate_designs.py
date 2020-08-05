from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from multiprocessing import Process
from multiprocessing import set_start_method
import os
import shutil
import gym
import numpy as np
import tensorflow as tf
import pickle as pkl
import argparse


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
            MlpPolicy,
            env,
            n_steps=max_episode_steps,
            ent_coef=0.001,
            learning_rate=0.0003,
            vf_coef=0.5,
            max_grad_norm=0.5,
            lam=0.95,
            nminibatches=10,
            noptepochs=10,
            cliprange=0.2,
            verbose=1,
            full_tensorboard_log=False,
            tensorboard_log=os.path.join(
                data_folder, f"p-{process_id}/{domain}-log-{i}"))

        # train for many gradient descent steps
        agent.learn(total_timesteps=total_timesteps)

        # calculate the agent score
        score = np.zeros([1])
        count = 0
        f = tf.io.gfile.glob(os.path.join(
            data_folder, f"p-{process_id}/{domain}-log-{i}/PPO2_1/events.out.*"))[0]
        for e in tf.compat.v1.train.summary_iterator(f):
            for v in e.summary.value:
                if v.tag == 'episode_reward':
                    score += v.simple_value
                    count += 1

        # save that score to the log folder
        np.save(os.path.join(
            data_folder, f"p-{process_id}/{domain}-log-{i}/score.npy"), score)


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

    parser = argparse.ArgumentParser('EvaluateDesigns')
    parser.add_argument('--local-dir',
                        type=str,
                        default='./data')
    parser.add_argument('--designs-file',
                        type=str,
                        default='designs.pkl')
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
                        default=500)
    parser.add_argument('--total-timesteps',
                        type=int,
                        default=1000000)
    parser.add_argument('--domain',
                        type=str,
                        default='ant',
                        choices=['ant', 'dog', 'dkitty'])
    args = parser.parse_args()

    with open(args.designs_file, 'rb') as f:
        design_list = pkl.load(f)

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

    # merge all previously evaluated scores
    all_scores = []
    for i in range(args.num_parallel):
        for j in range(len(design_chunks[i])):
            all_scores.append(np.load(os.path.join(
                args.local_dir, f"p-{i}/{args.domain}-log-{j}/score.npy")))
        shutil.rmtree(os.path.join(args.local_dir, f"p-{i}"))

    # save the combined scores to a new file
    all_scores = np.concatenate(all_scores, axis=0).reshape([-1, 1])
    np.save(os.path.join(
        args.local_dir, 'score.npy'), all_scores)
