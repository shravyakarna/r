#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window_size=10):
    sum_vec = np.cumsum(np.insert(data, 0, 0))
    moving_ave = (sum_vec[window_size:] - sum_vec[:-window_size]) / window_size
    return moving_ave


def display_stats(agent, training_totals, testing_totals, history):
    print('******* Training Stats *********')
    print('Average: {}'.format(np.mean(training_totals)))
    print('Standard Deviation: {}'.format(np.std(training_totals)))
    print('Minimum: {}'.format(np.min(training_totals)))
    print('Maximum: {}'.format(np.max(training_totals)))
    print('Number of training episodes: {}'.format(agent.training_trials))
    print()
    print('******* Testing Stats *********')
    print('Average: {}'.format(np.mean(testing_totals)))
    print('Standard Deviation: {}'.format(np.std(testing_totals)))
    print('Minimum: {}'.format(np.min(testing_totals)))
    print('Maximum: {}'.format(np.max(testing_totals)))
    print('Number of testing episodes: {}'.format(agent.testing_trials))
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(311)
    ax1.plot([num + 1 for num in range(agent.training_trials)],
             history['epsilon'],
             color='b',
             label='Exploration Factor (Epsilon)')
    ax1.plot([num + 1 for num in range(agent.training_trials)],
             history['alpha'],
             color='r',
             label='Learning Factor (Alpha)')
    ax1.set(title='Paramaters Plot',
            ylabel='Parameter values',
            xlabel='Trials')

    ax2 = fig.add_subplot(312)
    ax2.plot([num + 1 for num in range(agent.training_trials)],
             training_totals,
             color='m',
             label='Training',
             alpha=0.4, linewidth=2.0)
    total_trials = agent.training_trials + agent.testing_trials
    ax2.plot([num + 1 for num in range(agent.training_trials, total_trials)],
             testing_totals,
             color='k',
             label='Testing', linewidth=2.0)
    ax2.set(title='Reward per trial',
            ylabel='Rewards',
            xlabel='Trials')

    ax3 = fig.add_subplot(313)
    window_size = 10
    train_ma = moving_average(training_totals, window_size=window_size)
    train_epi = [num+1 for num in range(agent.training_trials-(window_size-1))]
    ax3.plot(train_epi, train_ma,
             color='m',
             label='Training',
             alpha=0.4, linewidth=2.0)
    test_ma = moving_average(testing_totals, window_size=window_size)
    total_trials = total_trials - (window_size*2) + 2
    test_epi = [num+1 for num in range(agent.training_trials-(window_size-1), total_trials)]
    ax3.plot(test_epi, test_ma,
             color='k',
             label='Testing', linewidth=2.0)
    ax3.set(title='Rolling Average Rewards',
            ylabel='Reward',
            xlabel='Trials')

    fig.subplots_adjust(hspace=0.5)
    ax1.legend(loc='best')
    ax2.legend(loc='best')
    ax3.legend(loc='best')


def save_info(agent, training_totals, testing_totals):
    with open('CartPole-v0_stats.txt', 'w') as file_obj:
        file_obj.write('/-------- Q-Learning --------\\\n')
        file_obj.write('\n/---- Training Stats ----\\\n')
        file_obj.write('Average: {}\n'.format(np.mean(training_totals)))
        file_obj.write('Standard Deviation: {}\n'.format(np.std(training_totals)))
        file_obj.write('Minimum: {}\n'.format(np.min(training_totals)))
        file_obj.write('Maximum: {}\n'.format(np.max(training_totals)))
        file_obj.write('Number of training episodes: {}\n'.format(agent.training_trials))
        file_obj.write('\n/---- Testing Stats ----\\\n')
        file_obj.write('Average: {}\n'.format(np.mean(testing_totals)))
        file_obj.write('Standard Deviation: {}\n'.format(np.std(testing_totals)))
        file_obj.write('Minimum: {}\n'.format(np.min(testing_totals)))
        file_obj.write('Maximum: {}\n'.format(np.max(testing_totals)))
        file_obj.write('Number of testing episodes: {}\n'.format(agent.testing_trials))
        file_obj.write('\n/---- Q-Table ----\\')
        for state in agent.Q_table:
            file_obj.write('\n State: ' + str(state) +
                           '\n\tAction: ' + str(agent.Q_table[state]))
    plt.savefig('plots.png')
    plt.show()
