import gym 
import numpy as np

parameter = np.random.rand(4) * 2 - 1

def run_episode(env, paramters):
    observation = env.reset()
    totalrewards = 0
    
    for _ in range(2000):
        env.render()
        action = 0 if np.matmul(paramters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalrewards += reward

        if done:
            break

    print("This Session end with ", totalrewards)
    env.reset()
    return totalrewards

def Random_Search():
    best_parameter = None
    best_reward = 0
    for _ in range(10000):
        paramter = np.random.rand(4) * 2 - 1
        env = gym.make('CartPole-v0')
        reward = run_episode(env, paramter)
        print(reward)

        if reward > best_reward:
            best_reward = reward
            beat_parameter = paramter

            if best_reward >= 150:
                break
    env.reset()

def Hill_Climbing():
    noise_scaling = 0.1
    paramters = np.random.rand(4) * 2 - 1
    best_reward = 0
    for _ in range(10000):
        newparams = paramters + (np.random.rand(4) * 2 - 1) * noise_scaling
        reward = 0
        env = gym.make('CartPole-v0')
        reward = run_episode(env, newparams)
        if reward > best_reward:
            best_reward = reward 
            paramter = newparams 
            if reward == 200:
                break

    env .reset()

if __name__ == '__main__':
    ##Random_Search()
    Hill_Climbing()




