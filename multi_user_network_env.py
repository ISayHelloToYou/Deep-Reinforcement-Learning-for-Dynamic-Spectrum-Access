import numpy as np
import random
import sys
import os


# TIME_SLOTS = 1
NUM_CHANNELS = 2
NUM_USERS = 3
NUM_SIZE = 3
ATTEMPT_PROB = 0.6
# GAMMA = 0.90

class env_network:
    def __init__(self,num_users,num_channels,attempt_prob, num_size, table):
        self.ATTEMPT_PROB = attempt_prob
        self.NUM_USERS = num_users
        self.NUM_CHANNELS = num_channels
        self.NUM_SIZE = num_size
        self.REWARD = -1

        self.table = table

        #self.channel_alloc_freq = 
        self.action_space = np.arange((self.NUM_CHANNELS+1)*self.NUM_SIZE)
        self.channel_space = np.arange(self.NUM_CHANNELS+1)
        self.size_space = np.arange(self.NUM_SIZE)
        self.users_action = np.zeros([self.NUM_USERS],np.int32)
        self.users_observation = np.zeros([self.NUM_USERS],np.int32)
    def reset(self):
        pass
    def sample(self):#生成action，维度为
        # table = np.array([[0,1,2],[3,4,5],[6,7,8]])
        mode = []
        mode1 = []
        x =  np.random.choice(self.channel_space,size=self.NUM_USERS)
        y =  np.random.choice(self.size_space,size=self.NUM_USERS)
        res = list(zip(x,y))
        index = np.array(res)
        # array([[0, 1],
               # [0, 2],
               # [2, 0]])
        for i in range(len(index)):
            mode.append(self.table[index[i][0]][index[i][1]])
        mode1 = np.array(mode)
        return mode1 #array([1,2,6]) shape=(3,)

    def inx(self, mode):
        x, y = np.where(self.table == mode)
        index = list(zip(x,y))
        inx = np.array(index)


    def step(self, action):#形参action是通过sample（）得到的，是模式的集合
        # print
        assert (action.size) == self.NUM_USERS
        channel_alloc_frequency = np.zeros([self.NUM_CHANNELS + 1], np.int32)  # 0 for no chnnel access
        # size_alloc_frequency = np.zeros([self.NUM_SIZE], np.int32)
        obs = []
        reward = np.zeros([self.NUM_USERS])
        j = 0
        for each in action:
            prob = random.uniform(0, 1)

            x, y = np.where( self.table == each)
            index = np.array(list(zip(x,y)))

            if prob <= self.ATTEMPT_PROB:
                self.users_action[j] = each  # action
                channel_alloc_frequency[index[0][0]] += 1
                # size_alloc_frequency[index[0][1]] += 1

            j += 1

        for i in range(1, len(channel_alloc_frequency)):
            if channel_alloc_frequency[i] > 1:
                channel_alloc_frequency[i] = 0

        for i in range(len(action)):
            x, y = np.where(self.table == self.users_action[i])
            index = np.array(list(zip(x, y)))

            self.users_observation[i] = channel_alloc_frequency[index[0][0]]
            if self.users_action[i] == 0:  # accessing no channel
                self.users_observation[i] = 0
            if self.users_action[i] == 1:  # accessing no channel
                self.users_observation[i] = 0
            if self.users_action[i] == 2:  # accessing no channel
                self.users_observation[i] = 0

            if self.users_observation[i] == 1:
                reward[i] = np.exp(-1 * index[0][1])
            if self.users_observation[i] == 0:
                reward[i] = 0
            obs.append((self.users_observation[i], reward[i]))

        residual_channel_capacity = channel_alloc_frequency[1:]
        residual_channel_capacity = 1 - residual_channel_capacity
        obs.append(residual_channel_capacity)
        return obs





