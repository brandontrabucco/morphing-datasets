from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import os
import gym
import pickle as pkl
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser('GenerateDataset')
    parser.add_argument('--local-dir',
                        type=str,
                        default='./data/p-0/ant-log-0/')
    parser.add_argument('--max-episode-steps',
                        type=int,
                        default=500)
    parser.add_argument('--domain',
                        type=str,
                        default='ant',
                        choices=['ant', 'dog', 'dkitty'])
    args = parser.parse_args()

    # check the target domain and import the appropriate agent
    if args.domain == "ant":
        env_name = "MorphingAntEnv"
    elif args.domain == "dog":
        env_name = "MorphingDogEnv"
    else:
        env_name = "MorphingDKittyEnv"

    # load the robot morphology in the design folder
    with open(os.path.join(args.local_dir, "design.pkl"), "rb") as f:
        spec = pkl.load(f)
        print(spec)

    # register my custom gym env
    gym.envs.register(
        id=f'{env_name}-v0',
        entry_point=f'morphing_agents.mujoco.{args.domain}.env:{env_name}',
        max_episode_steps=args.max_episode_steps,
        kwargs={'fixed_design': spec, 'expose_design': False})

    # multiprocess environment
    env = make_vec_env(f'{env_name}-v0', n_envs=1)

    # make a ppo agent
    model = PPO2.load(os.path.join(args.local_dir, "weights"))

    # Enjoy trained agent
    for i in range(10):
        obs, done = env.reset(), False
        returns = 0.0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render(mode='human')
            returns += reward
        print(returns)
