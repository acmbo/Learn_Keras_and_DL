'''

Keras & Q-Learning in Open Ai Gym

Siehe:
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c

'''

from collections import deque
import gym
from tensorflow import keras
import numpy as np
import random


class DQN:
    def __init__(self, env):
        self.env = env # Environment
        self.memory = deque(maxlen=2000)    # Memory aus Trials im Environment
                                            # Sinvoll, weil später ein zufälliges Trail zum lernen genutzt wird
                                            # um keinen Bias zu bekommen
        self.gamma = 0.95   #Deprication Faktor für Lernen
        self.epsilon = 1.0  #Wahl zwischen Exploration und Exploitation
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01

        self.model = self.create_model()
        # "hack" implemented by DeepMind to improve convergence
        self.target_model = self.create_model()

    def create_model(self):
        model = keras.models.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(keras.layers.Dense(24, input_dim=state_shape[0],
                        activation="relu"))
        model.add(keras.layers.Dense(48, activation="relu"))
        model.add(keras.layers.Dense(24, activation="relu"))
        model.add(keras.layers.Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])


    def replay(self):
        batch_size=32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory,batch_size)

        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)

        if done:
            target[0][action] = reward
        else:
            Q_future = max(self.target_model.predict(new_state)[0])
            target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state,target,epochs=1,verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    def act(self,state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])


def main():
    env = gym.make("MountainCar-v0")
    gamma = 0.9
    epsilon = .95
    trials = 100
    trial_len = 500
    updateTargetNetwork = 1000
    dqn_agent = DQN(env=env)
    steps = []
    for trial in range(trials):
        print('Trial:',trial)
        cur_state = env.reset().reshape(1, 2)
        for step in range(trial_len):
            print('step in Trail: ',step)
            action = dqn_agent.act(cur_state)
            #env.render()
            new_state, reward, done, _ = env.step(action)
            reward = reward if not done else -20
            #print(reward)
            new_state = new_state.reshape(1, 2)
            dqn_agent.remember(cur_state, action,
                               reward, new_state, done)

            dqn_agent.replay()
            dqn_agent.target_train()
            cur_state = new_state
            if done:
                break
        if step >= 199:
            print("Failed to complete trial")
        else:
            print("Completed in {} trials".format(trial))
            break



print('starting..')
main()