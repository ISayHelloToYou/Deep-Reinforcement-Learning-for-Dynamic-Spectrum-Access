import numpy as np
import random
import sys
import os


# TIME_SLOTS = 1
NUM_CHANNELS = 2
NUM_USERS = 3
NUM_SIZE = 3
ATTEMPT_PROB = 0.6
CHANNEL_CAPACITY = 7
# GAMMA = 0.90

class env_network:
    def __init__(self,num_users,num_channels, num_size,attempt_prob, table, channel_capacity):
        self.ATTEMPT_PROB = attempt_prob
        self.NUM_USERS = num_users
        self.NUM_CHANNELS = num_channels
        self.NUM_SIZE = num_size
        self.CHANNEL_CAPACITY = channel_capacity
        self.table = table
        self.REWARD = -1

        # self.action_space = np.arange((self.NUM_CHANNELS+1)*self.NUM_SIZE)
        self.action_space = np.arange((NUM_CHANNELS+1) * NUM_SIZE**NUM_SIZE)
        self.channel_space = np.arange(self.NUM_CHANNELS+1)
        self.size_space = np.arange(self.NUM_SIZE) + 1
        self.users_action = np.zeros([self.NUM_USERS],np.int32)
        self.users_observation = np.zeros([self.NUM_USERS],np.int32)

    def reset(self):
        pass

    def sample(self):
        #生成action，维度为(num_users,)
        mode = []
        index_list = []
        check_index = []

        x =  list(np.random.choice(self.channel_space,size=self.NUM_USERS))
        Y =  np.random.choice(self.size_space,size=3)#每次来的包的个数是固定的，但是大小是可变的
        y =  [list(np.random.choice(Y, size =3)) for i in range(self.NUM_USERS)]

        for i in range(1, self.NUM_SIZE + 1):
            for j in range(1, self.NUM_SIZE + 1):
                for k in range(1, self.NUM_SIZE + 1):
                    check_index.append([i,j,k])

        for i in range(len(y)):
            index_list.append(check_index.index(y[i]))

        res = list(zip(x, index_list))
        index = np.array(res)
        # array([[0, 1],
               # [0, 2],
               # [2, 0]])
        for i in range(len(index)):
            mode.append(self.table[index[i][0]][index[i][1]])
        mode1 = np.array(mode)
        return mode1 #array([1,2,6]) shape=(3,)


    # def step(self, action):#形参action是通过sample（）得到的，是模式的集合
    #     # print
    #     assert (action.size) == self.NUM_USERS
    #     channel_alloc_frequency = np.zeros([self.NUM_CHANNELS + 1], np.int32)  # 0 for no chnnel access
    #     # size_alloc_frequency = np.zeros([self.NUM_SIZE], np.int32)
    #     obs = []
    #     reward = np.zeros([self.NUM_USERS])
    #     j = 0
    #     for each in action:
    #         prob = random.uniform(0, 1)
    #
    #         x, y = np.where( self.table == each)
    #         index = np.array(list(zip(x,y)))
    #
    #         if prob <= self.ATTEMPT_PROB:
    #             self.users_action[j] = each  # action
    #             channel_alloc_frequency[index[0][0]] += 1
    #             # size_alloc_frequency[index[0][1]] += 1
    #
    #         j += 1
    #
    #     for i in range(1, len(channel_alloc_frequency)):
    #         if channel_alloc_frequency[i] > 1:
    #             channel_alloc_frequency[i] = 0
    #
    #     for i in range(len(action)):
    #         x, y = np.where(self.table == self.users_action[i])
    #         index = np.array(list(zip(x, y)))
    #
    #         self.users_observation[i] = channel_alloc_frequency[index[0][0]]
    #         if self.users_action[i] == 0:  # accessing no channel
    #             self.users_observation[i] = 0
    #         if self.users_action[i] == 1:  # accessing no channel
    #             self.users_observation[i] = 0
    #         if self.users_action[i] == 2:  # accessing no channel
    #             self.users_observation[i] = 0
    #
    #         if self.users_observation[i] == 1:
    #             reward[i] = 1 + np.exp(-1 * index[0][1])
    #         if self.users_observation[i] == 0:
    #             reward[i] = 0
    #         obs.append((self.users_observation[i], reward[i]))
    #
    #     residual_channel_capacity = channel_alloc_frequency[1:]
    #     residual_channel_capacity = 1 - residual_channel_capacity
    #     obs.append(residual_channel_capacity)
    #     return obs
    def step(self, action):#形参action是通过sample（）得到的，是模式的集合
        # print
        assert (action.size) == self.NUM_USERS
        channel_alloc_frequency = np.zeros([self.NUM_CHANNELS + 1], np.int32)  # 0 for no chnnel access
        # size_alloc_frequency = np.zeros([self.NUM_SIZE], np.int32)
        obs = []
        reward = np.zeros([self.NUM_USERS])
        j = 0

        check_index = []
        for x in range(1, self.NUM_SIZE + 1):
            for y in range(1, self.NUM_SIZE + 1):
                for z in range(1, self.NUM_SIZE + 1):
                    check_index.append([x, y, z])

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
            index = np.array(list(zip(x, y)))#array([[×, ×]]

            self.users_observation[i] = channel_alloc_frequency[index[0][0]]
            if self.users_action[i] >=0 and self.users_action[i] < self.NUM_SIZE**self.NUM_SIZE :  # accessing no channel,table的第一行
                self.users_observation[i] = 0

            if self.users_observation[i] == 1:
                if sum(check_index[index[0][1]]) <= self.CHANNEL_CAPACITY:
                    reward[i] = 1
                else:
                    reward[i] = 1 + np.power(np.e, (-1 * sum(np.random.choice(check_index[index[0][1]], size = 2))))
            else:
                reward[i] = 0
            obs.append((self.users_observation[i], reward[i]))

        # residual_channel_capacity = channel_alloc_frequency[1:]
        # residual_channel_capacity = 1 - residual_channel_capacity
        # obs.append(residual_channel_capacity)
        obs.append(channel_alloc_frequency)
        return obs





