import gym
import numpy as np
from td3_torch import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    load_checkpoint = False
    continue_traning = True
    
    if continue_traning:
        ex_env_name = "BipedalWalker_td3" 
        # now_env_name = "BipedalWalker_td3" 
        # env = gym.make('BipedalWalker-v3')
        now_env_name = "BipedalWalkerHardcore_td3" 
        env = gym.make('BipedalWalkerHardcore-v3')
    else:
        env_name = "BipedalWalker_td3" 
        ex_env_name = now_env_name = env_name
        env = gym.make('BipedalWalker-v3')
    agent = Agent(now_env_name, ex_env_name, input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0], load_checkpoint=load_checkpoint)
    n_games = 200
    filename = now_env_name + '_ContinueTraining.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    every = 1

    if load_checkpoint or continue_traning:
        agent.load_models()

    for i in range(n_games):
        if i % every == 0:
            render = True
        else:
            render = False

        observation = env.reset()
        done = False
        score = 0
        while not done:
            if render == True:
                env.render()
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn(i)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-10:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)