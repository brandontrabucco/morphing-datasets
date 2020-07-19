from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from multiprocessing import Process
from multiprocessing import set_start_method
import os
import numpy as np
import argparse


def start_gpu_process(process_id,
                      gpu_id,
                      env_name='Hopper-v2',
                      data_folder="data/",
                      max_episode_steps=1000,
                      n_envs=4,
                      total_timesteps=1000000,
                      max_save_steps=10000):
    """Train many policies using Proximal Policy Optimization on a set of
    morphologies and save training statistics

    Arg:

    process_id: int
        the integer id of the process running here
    gpu_id: int
        the integer id of the gpu allocated to this process
    env_name: str
        the name of an environment registered with gym that will be trained with
    data_folder: str
        the folder where training statistics are saved
    max_episode_steps: int
        the maximum number of steps per episode
    n_envs: int
        the number of parallel environments for sampling
    total_timesteps: int
        the number of environments steps to collect while training
    max_save_steps: int
        the number of environments steps per save operation
    """

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    # multiprocess environment
    env = make_vec_env(env_name, n_envs=n_envs)
    eval_env = make_vec_env(env_name, n_envs=1)

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
            data_folder, f"p-{process_id}/{env_name}"))

    for i in range(total_timesteps // max_save_steps):

        # train for many gradient descent steps
        agent.learn(total_timesteps=max_save_steps,
                    reset_num_timesteps=False)
        agent.save(os.path.join(
            data_folder, f"p-{process_id}/{env_name}/weights-{i}"))

        # calculate the expected return for this policy
        avg_return = []
        for _ in range(10):
            obs, done = eval_env.reset(), False
            total_return = 0
            while not done:
                action, _states = agent.predict(obs)
                obs, reward, done, info = eval_env.step(action)
                total_return += reward
            avg_return.append(total_return)
        avg_return = np.mean(avg_return)
        np.save(os.path.join(
            data_folder, f"p-{process_id}/{env_name}/return-{i}"), avg_return)

    remaining_steps = total_timesteps % max_save_steps
    if remaining_steps > 0:

        # train for many gradient descent steps
        agent.learn(total_timesteps=remaining_steps,
                    reset_num_timesteps=False)
        agent.save(os.path.join(
            data_folder, f"p-{process_id}/{env_name}/weights-{i + 1}"))

        # calculate the expected return for this policy
        avg_return = []
        for _ in range(10):
            obs, done = eval_env.reset(), False
            total_return = 0
            while not done:
                action, _states = agent.predict(obs)
                obs, reward, done, info = eval_env.step(action)
                total_return += reward
            avg_return.append(total_return)
        avg_return = np.mean(avg_return)
        np.save(os.path.join(
            data_folder, f"p-{process_id}/{env_name}/return-{i + 1}"), avg_return)


if __name__ == "__main__":

    # on startup ensure all processes are started using the spawn method
    # see https://github.com/tensorflow/tensorflow/issues/5448
    set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser('GenerateDataset')
    parser.add_argument('--local-dir',
                        type=str,
                        default='./data')
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
    parser.add_argument('--max-save-steps',
                        type=int,
                        default=10000)
    parser.add_argument('--env-name',
                        type=str,
                        default='Hopper-v2')
    args = parser.parse_args()

    ps = [Process(target=start_gpu_process,
                  args=(process_id,
                        process_id % args.num_gpus),
                  kwargs=dict(env_name=args.env_name,
                              data_folder=args.local_dir,
                              max_episode_steps=args.max_episode_steps,
                              n_envs=args.n_envs,
                              total_timesteps=args.total_timesteps,
                              max_save_steps=args.max_save_steps))
          for process_id in range(args.num_parallel)]

    for p in ps:
        p.start()

    for p in ps:
        p.join()
