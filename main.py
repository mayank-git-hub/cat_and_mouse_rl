from brain import brain
from environment import env
from agent import agent
import argparse
import numpy as np
import matplotlib.pyplot as plt
def train_settings():

    # Training settings
    global parser
    global args

    parser = argparse.ArgumentParser(description='Cat&Mouse')
    parser.add_argument('--arena-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 15)')
    parser.add_argument('--random-steps', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--seed', type=int, default=600, metavar='S',
                        help='random seed (default: 600)')
    args = parser.parse_args()

def seed():

    np.random.seed(args.seed)


def animate(agent_):

    copy_maze = np.copy(agent_.env.maze)

    plt.imshow(copy_maze)
    plt.pause(0.01)
    plt.clf()

    for __ in range(100):

        agent_.clean_start()

        while True:

            # print(agent_.s)

            s_ = agent_.brain.exploit(agent_.s, agent_.env.poss)
            agent_.s = s_
            copy_maze = np.copy(agent_.env.maze)
            copy_maze[np.unravel_index(agent_.s, [args.arena_size, args.arena_size], order='C')] = 5
            plt.imshow(copy_maze)
            plt.pause(0.1)
            plt.clf()

            if agent_.env.maze.ravel()[s_] != 0:
                break




if __name__ == '__main__':

    train_settings()

    seed()

    brain_ = brain(size= args.arena_size, gamma = 0.9, l_r = 0.9)
    env_ = env(size = args.arena_size, cat_r=[-10, -20], cheese_r=[10, 20])

    agent_ = agent(env = env_, brain = brain_)

    plt.imshow(env_.maze)
    plt.pause(1)

    for i in range(args.random_steps):

        agent_.step()

        if i % 10 == 0:

            plt.imshow(agent_.brain.q_mat)
            plt.pause(0.01)
            plt.clf()

    animate(agent_)