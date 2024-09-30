#!/usr/bin/env python

import gym
import numpy as np
import argparse
from agent import AgentBasic, AgentRandom, AgentLearning
import stats


def environment_info(env):
    print('************** Environment Info **************')
    print('Observation space: {}'.format(env.observation_space))
    print('Observation space high values: {}'.format(env.observation_space.high))
    print('Observation space low values: {}'.format(env.observation_space.low))
    print('Action space: {}'.format(env.action_space))
    print()


def basic_guessing_policy(env, agent):
    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        for step in range(1000):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        totals.append(episode_rewards)

    print('************** Reward Statistics **************')
    print('Average: {}'.format(np.mean(totals)))
    print('Standard Deviation: {}'.format(np.std(totals)))
    print('Minimum: {}'.format(np.min(totals)))
    print('Maximum: {}'.format(np.max(totals)))


def random_guessing_policy(env, agent):
    totals = []
    for episode in range(500):
        episode_rewards = 0
        obs = env.reset()
        for step in range(1000):
            action = agent.act()
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if done:
                break
        totals.append(episode_rewards)

    print('Average: {}'.format(np.mean(totals)))
    print('Standard Deviation: {}'.format(np.std(totals)))
    print('Minimum: {}'.format(np.min(totals)))
    print('Maximum: {}'.format(np.max(totals)))


def q_learning(env, agent):
    valid_actions = [0, 1]
    tolerance = 0.001
    training = True
    training_totals = []
    testing_totals = []
    history = {'epsilon': [], 'alpha': []}
    for episode in range(800):
        episode_rewards = 0
        obs = env.reset()
        if agent.epsilon < tolerance:
            agent.alpha = 0
            agent.epsilon = 0
            training = False
        agent.epsilon = agent.epsilon * 0.99
        for step in range(200):
            state = agent.create_state(obs)
            agent.create_Q(state, valid_actions)
            action = agent.choose_action(state)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            if step != 0:
                agent.learn(state, action, prev_reward, prev_state, prev_action)
            prev_state = state
            prev_action = action
            prev_reward = reward
            if done:
                break
        if training:
            training_totals.append(episode_rewards)
            agent.training_trials += 1
            history['epsilon'].append(agent.epsilon)
            history['alpha'].append(agent.alpha)
        else:
            testing_totals.append(episode_rewards)
            agent.testing_trials += 1
            if agent.testing_trials == 100:
                break
    return training_totals, testing_totals, history


def main():
    env = gym.make('CartPole-v0')
    env.seed(21)
    environment_info(env)
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--agent', help='define type of agent you want')
    args = parser.parse_args()
    if args.agent == 'basic':
        agent = AgentBasic()
        basic_guessing_policy(env, agent)
    elif args.agent == 'random':
        agent = AgentRandom(env.action_space)
        random_guessing_policy(env, agent)
    elif args.agent == 'q-learning':
        agent = AgentLearning(env, alpha=0.9, epsilon=1.0, gamma=0.9)
        training_totals, testing_totals, history = q_learning(env, agent)
        stats.display_stats(agent, training_totals, testing_totals, history)
        stats.save_info(agent, training_totals, testing_totals)
        if np.mean(testing_totals) >= 195.0:
            print("Environment SOLVED!!!")
        else:
            print("Environment not solved.", "Must get average reward of 195.0 or", "greater for 100 consecutive trials.")
    else:
        agent = AgentBasic()
        basic_guessing_policy(env, agent)


if __name__ == '__main__':
    main()
