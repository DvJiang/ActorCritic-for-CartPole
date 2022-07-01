import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib import animation 

# 设置全局随机变量种子
np.random.seed(1)
tf.random.set_seed(1)

# 动画保存
def display_frames_as_gif(frames,name):
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=1)
    anim.save('D:\\'+name+'.gif', writer='ffmpeg', fps=30)

# 关于Actor的部分
class Actor:
    def __init__(self,learning_rate):
        self.lr = learning_rate
        self.model = self.model_init()
    
    # 模型建立及设定
    def model_init(self):
        # 输入层，ob的特征有4个
        input_layer = tf.keras.layers.Input(shape=(4,))

        # 中间层，取20个神经元，激活函数用的relu
        layer = keras.layers.Dense(
            units=20,
            activation=keras.activations.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )(input_layer)

        # 输出层，输出只有两个动作的概率（左和右），激活函数为softmax
        output_layer = keras.layers.Dense(
            units=2,
            activation=keras.activations.softmax,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )(layer)

        # 设置学习率
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # 建立模型 损失函数选择交叉熵
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=self.optimizer)

        return model

    # 动作选择
    def choose_action(self,state):# 有p的概率选择0
        p = self.model(state)[0].numpy()
        rand_one = np.random.rand()
        if (rand_one > p[0]):
            return 1
        else:
            return 0

    # 模型训练
    def fit(self, state, action, weight):
        self.model.fit(state, np.array([action]), verbose=0, sample_weight=weight)

# 关于critic的部分
class Critic:
    def __init__(self,learning_rate,gama,iter_t):
        self.iter = 0 #计算副本更新周期
        self.iter_t = iter_t # 副本更新周期
        self.lr = learning_rate
        self.gama = gama
        self.model = self.model_init()
        self.model_ = self.model # 创建副本 周期更新
    
    # 模型建立及设定
    def model_init(self):
        # 输入层，ob的特征有4个
        input_layer = tf.keras.layers.Input(shape=(4,))

        # 中间层，取20个神经元，激活函数用的relu
        layer = keras.layers.Dense(
            units=20,
            activation=keras.activations.relu,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )(input_layer)

        # 输出层
        output_layer = keras.layers.Dense(
            units=1,
            kernel_initializer=tf.random_normal_initializer(0., .1),
            bias_initializer=tf.constant_initializer(0.1),
        )(layer)

        # 设置学习率
        self.optimizer = keras.optimizers.Adam(learning_rate=self.lr)

        # 建立模型
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer=self.optimizer)

        return model

    def fit(self, state, reward, state_):
        self.iter += 1
        value = reward + self.model_(state_) * self.gama # 使用副本，使得数据收敛更容易
        td_error = value - self.model(state)

        self.model.fit(state, value, verbose=0)
        if(self.iter >= self.iter_t):
            self.iter = 0
            self.model_ = self.model

        return td_error

def main():
    env = gym.make('CartPole-v0')
    frames1 = []
    frames2 = []
    time = []
    actor = Actor(1e-3)
    critic = Critic(1e-2,0.95,5)

    print("-------------------  start trying")
    print(animation.writers.list())
    for epi in range(200):
        rewards = []
        observation = env.reset(seed = 1)
        observation = observation[np.newaxis, :]

        for t in range(1000):
            #env.render()# 是否渲染
            # if (epi >= 0 and epi <= 9):frames1.append(env.render(mode = 'rgb_array'))
            if (epi >= 189 and epi <= 199):frames2.append(env.render(mode = 'rgb_array'))

            # 选择
            action = actor.choose_action(observation)
            observation_, reward, done, info = env.step(action)# action 0左1右 
            observation_ = observation_[np.newaxis, :]

            # 训练
            if (done and t < 199): reward = -20
            if (done and t >= 199): reward = 20
            rewards.append(reward)
            td_error = critic.fit(observation, reward, observation_)
            actor.fit(observation, action, td_error)
            observation = observation_

            if done:
                print("Episode {}: {} timesteps".format(epi,t+1))
                time.append(t)
                break
    env.close()

    # 展示
    x = np.linspace(0,199,200)
    plt.plot(x,np.array(time), label='time')
    plt.show()
    # display_frames_as_gif(frames1,'1-10-result')
    display_frames_as_gif(frames2,'190-200-result')



if __name__ == "__main__":
    main()