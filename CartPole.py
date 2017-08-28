import gym
import random 
import numpy as np
import tflearn 
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter


LR = 1e-3
env = gym.make('CartPole-v0')
env.reset()
goal_step = 500
socre_requirement = 10
initial_games = 10000

def some_random_game():
    for epoise in range(5):
        env.reset()
        for t in range(goal_step):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            if done :
                break

def init_population():
    train_data = []
    scores = []
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_step):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])

            prev_observation = observation

            score += reward
            if done:
                break

        if score >= socre_requirement:
            accepted_scores.append(score)

            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                train_data.append([data[0], output])
    env.reset()
    scores.append(score)

    print('Average accepted score:',mean(accepted_scores))
    print('Median score for accepted scores:',median(accepted_scores))
    print(Counter(accepted_scores))

    return train_data

def nerual_network(input_size):
    
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')

    network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def train_model(train_data, model=None):

    X = np.array([i[0] for i in train_data]).reshape(-1, len(train_data[0][0]), 1)
    y = np.array([i[1] for i in train_data])

    if not model:
        model = nerual_network(len(X[0]))

    model.fit({'input': X}, {'targets': y}, n_epoch=100, snapshot_step=500, show_metric=True, run_id='openai_learning')

    return model

training_data = init_population()
model = train_model(training_data)































    


if __name__ == "__main__":
    some_random_game()
