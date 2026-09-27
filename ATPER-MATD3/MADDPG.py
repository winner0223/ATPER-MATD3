import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib
import os
import random
from collections import deque
import time


matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['axes.unicode_minus'] = False  

class OUActionNoise:


    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):

        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class ReplayBuffer:


    def __init__(self, capacity, state_dim, action_dim):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.action_dim = action_dim

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)

        states = np.array([sample[0] for sample in samples], dtype=np.float32)
        actions = np.array([sample[1] for sample in samples], dtype=np.float32)
        rewards = np.array([sample[2] for sample in samples], dtype=np.float32)
        next_states = np.array([sample[3] for sample in samples], dtype=np.float32)
        dones = np.array([sample[4] for sample in samples], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DDPG:
    """
    Deep Deterministic Policy Gradient算法
    """

    def __init__(self, state_dim, action_dim, action_high, action_low):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.action_low = action_low

        # Actor网络参数
        self.actor_lr = 0.001

        # Critic网络参数
        self.critic_lr = 0.002

        # 目标网络软更新参数
        self.tau = 0.005

        # 折扣因子
        self.gamma = 0.99

        # 构建Actor和Critic网络
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # 构建目标Actor和Critic网络
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()

        # 复制权重到目标网络
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        # 优化器
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

        # 定义噪声生成器
        std_dev = 0.2
        self.noise = OUActionNoise(mean=np.zeros(action_dim), std_deviation=float(std_dev) * np.ones(action_dim))

    def _build_actor(self):
        """
        构建Actor网络 - 策略网络，输入状态，输出动作
        """
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)

        # 使用tanh激活函数输出[-1,1]范围的动作
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)

        # 将输出缩放到动作空间范围
        scaled_outputs = outputs * (self.action_high - self.action_low) / 2.0 + (
                    self.action_high + self.action_low) / 2.0

        model = models.Model(inputs=inputs, outputs=scaled_outputs)
        return model

    def _build_critic(self):
        """
        构建Critic网络 - 价值网络，输入状态和动作，输出Q值
        """
        # 状态输入
        state_input = layers.Input(shape=(self.state_dim,))
        state_out = layers.Dense(256, activation='relu')(state_input)

        # 动作输入
        action_input = layers.Input(shape=(self.action_dim,))
        action_out = layers.Dense(64, activation='relu')(action_input)

        # 合并状态和动作
        concat = layers.Concatenate()([state_out, action_out])

        x = layers.Dense(128, activation='relu')(concat)
        x = layers.Dense(64, activation='relu')(x)

        # 输出单个Q值
        outputs = layers.Dense(1)(x)

        model = models.Model(inputs=[state_input, action_input], outputs=outputs)
        return model

    def _build_attention_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(self.feature_dim, activation='relu')(inputs)

        seq_length = 4
        if self.feature_dim % seq_length != 0:
            raise ValueError(f"feature_dim ({self.feature_dim}) 必须能被 seq_length ({seq_length}) 整除")
        seq_feature_dim = self.feature_dim // seq_length

        x_reshaped = layers.Reshape((seq_length, seq_feature_dim))(x)

        attention_output, self.attention_weights = MultiHeadAttention(
            d_model=seq_feature_dim, num_heads=self.attention_heads
        )(x_reshaped, x_reshaped, x_reshaped)

        attention_output = layers.add([x_reshaped, attention_output])
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

        # Replace Flatten with explicit Reshape to guarantee shape is known
        x = layers.Reshape((seq_length * seq_feature_dim,))(attention_output)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)

        outputs = outputs * (self.action_high - self.action_low) / 2.0 + (self.action_high + self.action_low) / 2.0

        return models.Model(inputs=inputs, outputs=outputs, name=f"{self.agent_name}_attention_actor")

    def _build_attention_critic(self):
        state_input = layers.Input(shape=(self.state_dim,))
        state_out = layers.Dense(self.feature_dim, activation='relu')(state_input)
        action_input = layers.Input(shape=(self.action_dim,))
        action_out = layers.Dense(32, activation='relu')(action_input)
        combined = layers.Concatenate()([state_out, action_out])

        combined_dim = self.feature_dim + 32
        seq_length = 4
        seq_feature_dim = combined_dim // seq_length

        combined = layers.Dense(seq_length * seq_feature_dim, activation='relu')(combined)
        combined_reshaped = layers.Reshape((seq_length, seq_feature_dim))(combined)

        attention_output, _ = MultiHeadAttention(
            d_model=seq_feature_dim, num_heads=self.attention_heads)(combined_reshaped, combined_reshaped,
                                                                     combined_reshaped)
        attention_output = layers.add([combined_reshaped, attention_output])
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

        # Replace Flatten with explicit Reshape
        x = layers.Reshape((seq_length * seq_feature_dim,))(attention_output)

        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1)(x)
        return models.Model(inputs=[state_input, action_input], outputs=outputs,
                            name=f"{self.agent_name}_attention_critic")

    def policy(self, state, add_noise=True):
        """
        根据状态选择动作
        """
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
        action = self.actor(state)[0].numpy()

        if add_noise:
            action += self.noise()

        # 裁剪动作到有效范围
        action = np.clip(action, self.action_low, self.action_high)

        return action

    def learn(self, replay_buffer, batch_size):
        """
        从经验回放缓冲区中学习
        """
        # 从回放缓冲区采样
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # 计算目标Q值
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic([next_states, target_actions])

            # 贝尔曼方程
            target_q = rewards + (1 - dones) * self.gamma * target_q_values

            # 当前Q值
            current_q = self.critic([states, actions])

            # 均方误差损失
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))

        # 更新Critic网络
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            # 当前策略的动作
            new_actions = self.actor(states)

            # 当前策略的Q值
            q_values = self.critic([states, new_actions])

            # 负Q值作为策略梯度损失
            actor_loss = -tf.reduce_mean(q_values)

        # 更新Actor网络
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # 软更新目标网络
        self._update_target_networks()

        return critic_loss.numpy(), actor_loss.numpy()

    def _update_target_networks(self):
        """
        软更新目标网络
        """
        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(a * (1 - self.tau) + b * self.tau)

        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(a * (1 - self.tau) + b * self.tau)

# 星地车联网环境模拟器
class SatVehicleEnvironment:


    def __init__(self, num_vehicles=3, num_mec=1, num_satellites=2):
        # 环境参数
        self.num_vehicles = num_vehicles
        self.num_mec = num_mec
        self.num_satellites = num_satellites
        self.num_nodes = num_vehicles + num_mec + num_satellites

        # 资源参数初始化
        self.vehicle_compute_capacity = np.random.uniform(0.5, 1.5, num_vehicles)  # GHz
        self.mec_compute_capacity = np.full(num_mec, 5.0)  # GHz
        self.satellite_compute_capacity = np.random.uniform(2.0, 3.0, num_satellites)  # GHz

        # 通信带宽参数
        self.vehicle_bandwidth = np.random.uniform(1.0, 5.0, num_vehicles)  # Mbps
        self.mec_bandwidth = np.full(num_mec, 20.0)  # Mbps
        self.satellite_bandwidth = np.random.uniform(10.0, 15.0, num_satellites)  # Mbps

        # 当前资源使用情况
        self.vehicle_load = np.zeros(num_vehicles)
        self.mec_load = np.zeros(num_mec)
        self.satellite_load = np.zeros(num_satellites)

        # 任务队列
        self.tasks = []
        self.current_task_id = 0
        self.max_tasks = 10

        # 状态空间维度和动作空间维度
        self.state_dim = (num_vehicles + num_mec + num_satellites) * 2 + self.max_tasks * 3  # 节点状态+任务状态
        self.action_dim = self.max_tasks * self.num_nodes  # 任务分配到各节点的决策

    def reset(self):

        self.vehicle_load = np.zeros(self.num_vehicles)
        self.mec_load = np.zeros(self.num_mec)
        self.satellite_load = np.zeros(self.num_satellites)


        self.tasks = []
        self.generate_tasks()


        state = self._get_state()

        return state

    def generate_tasks(self):

        self.tasks = []
        for _ in range(self.max_tasks):
            task = {
                'id': self.current_task_id,
                'data_size': np.random.uniform(0.5, 2.0),  # MB
                'compute_cycles': np.random.uniform(1e8, 1e9),  # CPU cycles
                'deadline': np.random.uniform(50, 200),  # ms
                'allocated': False,
                'node_type': -1,  # -1: 未分配, 0: vehicle, 1: MEC, 2: satellite
                'node_id': -1
            }
            self.tasks.append(task)
            self.current_task_id += 1

    def _get_state(self):

        node_states = []

        # 车辆节点状态
        for i in range(self.num_vehicles):
            node_states.append(self.vehicle_compute_capacity[i])
            node_states.append(self.vehicle_load[i])

        # MEC节点状态
        for i in range(self.num_mec):
            node_states.append(self.mec_compute_capacity[i])
            node_states.append(self.mec_load[i])

        # 卫星节点状态
        for i in range(self.num_satellites):
            node_states.append(self.satellite_compute_capacity[i])
            node_states.append(self.satellite_load[i])

        # 任务状态: 数据大小、计算需求和截止时间
        task_states = []
        for task in self.tasks:
            task_states.append(task['data_size'])
            task_states.append(task['compute_cycles'] / 1e9)  # 归一化
            task_states.append(task['deadline'] / 200.0)  # 归一化

        # 组合节点状态和任务状态
        state = np.array(node_states + task_states, dtype=np.float32)

        return state

    def step(self, action):

        task_allocations = self._convert_action_to_allocation(action)

        # 应用任务分配决策
        self._apply_allocations(task_allocations)

        # 计算奖励
        reward = self._compute_reward()

        # 获取新状态
        new_state = self._get_state()

        # 检查是否所有任务都已分配
        done = all(task['allocated'] for task in self.tasks)

        return new_state, reward, done, {}

    def _convert_action_to_allocation(self, action):

        action_matrix = action.reshape(self.max_tasks, self.num_nodes)


        task_allocations = []
        for i, task_action in enumerate(action_matrix):
            if not self.tasks[i]['allocated']:
                selected_node = np.argmax(task_action)

                # 确定节点类型和ID
                if selected_node < self.num_vehicles:
                    node_type = 0  # vehicle
                    node_id = selected_node
                elif selected_node < self.num_vehicles + self.num_mec:
                    node_type = 1  # MEC
                    node_id = selected_node - self.num_vehicles
                else:
                    node_type = 2  # satellite
                    node_id = selected_node - (self.num_vehicles + self.num_mec)

                task_allocations.append((i, node_type, node_id))

        return task_allocations

    def _apply_allocations(self, task_allocations):

        for task_idx, node_type, node_id in task_allocations:
            task = self.tasks[task_idx]
            task['allocated'] = True
            task['node_type'] = node_type
            task['node_id'] = node_id

            # 更新节点负载
            compute_load = task['compute_cycles'] / 1e9  # 归一化
            if node_type == 0:  # vehicle
                self.vehicle_load[node_id] += compute_load / self.vehicle_compute_capacity[node_id]
            elif node_type == 1:  # MEC
                self.mec_load[node_id] += compute_load / self.mec_compute_capacity[node_id]
            else:  # satellite
                self.satellite_load[node_id] += compute_load / self.satellite_compute_capacity[node_id]

    def _compute_reward(self):

        total_delay = 0.0
        completed_tasks = 0

        for task in self.tasks:
            if task['allocated']:
                if task['node_type'] == 0:  # vehicle
                    node_capacity = self.vehicle_compute_capacity[task['node_id']]
                    trans_delay = 0  # 本地处理无传输延迟
                elif task['node_type'] == 1:  # MEC
                    node_capacity = self.mec_compute_capacity[task['node_id']]
                    trans_delay = task['data_size'] / self.mec_bandwidth[task['node_id']]
                else:  # satellite
                    node_capacity = self.satellite_compute_capacity[task['node_id']]
                    trans_delay = task['data_size'] / self.satellite_bandwidth[task['node_id']]

                # 计算处理延迟
                proc_delay = task['compute_cycles'] / (node_capacity * 1e9)
                total_delay += trans_delay + proc_delay
                completed_tasks += 1

        avg_delay = total_delay / completed_tasks if completed_tasks > 0 else float('inf')

        # 计算资源利用率
        vehicle_util = np.mean(self.vehicle_load)
        mec_util = np.mean(self.mec_load)
        satellite_util = np.mean(self.satellite_load)
        avg_util = np.mean([vehicle_util, mec_util, satellite_util])


        all_loads = np.concatenate([
            self.vehicle_load / np.max(self.vehicle_compute_capacity),
            self.mec_load / np.max(self.mec_compute_capacity),
            self.satellite_load / np.max(self.satellite_compute_capacity)
        ])
        load_balance = np.std(all_loads)

        # 计算任务完成率
        completion_rate = completed_tasks / len(self.tasks)

        # 综合奖励
        delay_reward = -avg_delay / 100.0  # 归一化
        util_reward = avg_util
        balance_reward = -load_balance
        completion_reward = completion_rate

        # 权重系数
        w1, w2, w3, w4 = 0.4, 0.3, 0.15, 0.15

        reward = w1 * delay_reward + w2 * util_reward + w3 * balance_reward + w4 * completion_reward

        return reward


def train_ddpg():

    env = SatVehicleEnvironment(num_vehicles=3, num_mec=1, num_satellites=2)


    agent = DDPG(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        action_high=np.ones(env.action_dim),
        action_low=-np.ones(env.action_dim)
    )


    buffer_capacity = 10000
    replay_buffer = ReplayBuffer(
        capacity=buffer_capacity,
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )

    # 训练参数
    num_episodes = 100
    max_steps_per_episode = 20
    batch_size = 64

    # 记录每个episode的总奖励
    episode_rewards = []
    avg_delays = []
    completion_rates = []

    # 训练循环
    for episode in range(num_episodes):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        total_delay = 0
        completed_tasks = 0

        for step in range(max_steps_per_episode):
            # 选择动作
            action = agent.policy(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储经验
            replay_buffer.add(state, action, reward, next_state, done)

            # 从缓冲区中学习
            if len(replay_buffer) > batch_size:
                agent.learn(replay_buffer, batch_size)

            # 更新状态
            state = next_state
            episode_reward += reward

            # 计算性能指标
            for task in env.tasks:
                if task['allocated']:
                    if task['node_type'] == 0:  # vehicle
                        node_capacity = env.vehicle_compute_capacity[task['node_id']]
                        trans_delay = 0
                    elif task['node_type'] == 1:  # MEC
                        node_capacity = env.mec_compute_capacity[task['node_id']]
                        trans_delay = task['data_size'] / env.mec_bandwidth[task['node_id']]
                    else:  # satellite
                        node_capacity = env.satellite_compute_capacity[task['node_id']]
                        trans_delay = task['data_size'] / env.satellite_bandwidth[task['node_id']]

                    proc_delay = task['compute_cycles'] / (node_capacity * 1e9)
                    total_delay += trans_delay + proc_delay
                    completed_tasks += 1

            if done:
                break

        # 记录性能指标
        episode_rewards.append(episode_reward)
        avg_delay = total_delay / completed_tasks if completed_tasks > 0 else float('inf')
        avg_delays.append(avg_delay)
        completion_rate = completed_tasks / len(env.tasks)
        completion_rates.append(completion_rate)

        # 打印训练进度
        print(
            f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}, Avg Delay: {avg_delay:.2f}ms, Completion Rate: {completion_rate:.2f}")

    return episode_rewards, avg_delays, completion_rates, agent


def plot_results(episode_rewards, avg_delays, completion_rates):

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # 绘制奖励曲线
    axes[0].plot(episode_rewards)
    axes[0].set_title('训练奖励曲线')
    axes[0].set_xlabel('Number of episodes')
    axes[0].set_ylabel('奖励值')
    axes[0].grid(True)

    # 绘制平均延迟曲线
    axes[1].plot(avg_delays)
    axes[1].set_title('平均任务处理延迟')
    axes[1].set_xlabel('Number of episodes')
    axes[1].set_ylabel('延迟 (ms)')
    axes[1].grid(True)

    # 绘制任务完成率曲线
    axes[2].plot(completion_rates)
    axes[2].set_title('任务完成率')
    axes[2].set_xlabel('Number of episodes')
    axes[2].set_ylabel('完成率')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True)


    plt.tight_layout()
    plt.savefig('ddpg_training_results.png', dpi=300)
    plt.show()


if __name__ == "__main__":

    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)


    print("开始训练")
    episode_rewards, avg_delays, completion_rates, agent = train_ddpg()


    plot_results(episode_rewards, avg_delays, completion_rates)

    print("训练完成")
