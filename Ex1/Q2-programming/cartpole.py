import gymnasium as gym
import random as random
import numpy as np
import matplotlib.pyplot as plt

#random.seed(0)
#np.random.seed(0)

def main():
    env = gym.make("CartPole-v1")
    num_episodes_to_perfection = [] # number of episodes to reach 500 reward
    for _ in range(1000):
        Ws = []
        rewards = []
        found_perfect = False
        for episode_index in range(10000):
            observation = env.reset()[0]
            w = np.random.uniform(-1, 1, 4)
            Ws.append(w)
            total_reward = 0

            for step_index in range(500):

                #env.render()
                
                action = generate_action(observation, w)
                
                observation, reward, terminated, truncated, info = env.step(action)         
                
                total_reward += reward
                
                if terminated or truncated:
                    rewards.append(total_reward)
                    if total_reward == 500:
                        num_episodes_to_perfection.append(episode_index+1)
                        print("Found perfect policy at episode", episode_index+1)
                        found_perfect = True
                    break    
                
            if found_perfect:
                break
        
    env.close()                

    # plot histogram of number of episodes to reach 500 reward
    print("Average number of episodes to reach perfect policy:", np.mean(num_episodes_to_perfection))
    print(len(num_episodes_to_perfection), "out of 1000 experiments reached perfect policy")
    plt.hist(num_episodes_to_perfection, bins=35)
    plt.xlabel("Episodes to reach perfect policy")
    plt.ylabel("Frequency")
    plt.xlim(0, 200)
    plt.text(100, 100, "Average to episodes reach 500: "+str(np.mean(num_episodes_to_perfection)))
    plt.savefig("cartPole.png")
    plt.show()
    

    


def generate_action(observation, w):
    return 1 if np.dot(observation, w) >= 0 else 0

if __name__ == "__main__":
    main()
    
