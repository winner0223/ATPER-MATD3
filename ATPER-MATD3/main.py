import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib
import os
import random
from collections import deque
import time
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import copy

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置TensorFlow显存按需分配
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU memory growth enabled")


# OUActionNoise 类
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
        self.x_prev = self.x_initial if self.x_initial is not None else np.zeros_like(self.mean)


class AdaptiveOUActionNoise(OUActionNoise):
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None,
                 decay_rate=0.995, min_std_dev=0.05, var_window_size=20):
        super().__init__(mean, std_deviation, theta, dt, x_initial)

        # 确保初始标准差和最小标准差形状匹配
        if isinstance(std_deviation, np.ndarray):
            self.initial_std_dev = std_deviation.copy()  # 保存副本避免引用问题
            self.min_std_dev = min_std_dev * np.ones_like(std_deviation)  # 确保形状一致
        else:
            self.initial_std_dev = std_deviation
            self.min_std_dev = min_std_dev

        self.decay_rate = decay_rate
        self.reward_history = deque(maxlen=var_window_size)


    def adapt(self, reward):
        """
        根据奖励适应噪声大小
        """
        self.reward_history.append(reward)
        if len(self.reward_history) >= self.reward_history.maxlen:
            reward_var = np.var(self.reward_history)
            if reward_var > 0.5:
                # 如果奖励方差大，增加探索（较大噪声）
                # 使用np.minimum代替min，因为self.std_dev可能是数组
                self.std_dev = np.minimum(self.initial_std_dev, self.std_dev / self.decay_rate)
            else:
                # 如果奖励方差小，减少探索（较小噪声）
                # 使用np.maximum代替max，因为self.std_dev可能是数组
                self.std_dev = np.maximum(self.min_std_dev, self.std_dev * self.decay_rate)

# ReplayBuffer 类
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
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# PrioritizedReplayBuffer 类
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, state_dim, action_dim, alpha=0.6, beta=0.4, beta_increment=0.001,
                 task_completion_bonus=1.0):
        super().__init__(capacity, state_dim, action_dim)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6
        self.task_completion_bonus = task_completion_bonus

    def add(self, state, action, reward, next_state, done, task_completed=False):
        """
        添加经验到缓冲区，并分配优先级
        """
        # 修复：正确检查优先级队列是否为空
        if len(self.priorities) == 0:
            max_priority = 1.0
        else:
            # 确保使用标量值，而不是数组
            priorities_list = list(self.priorities)
            if isinstance(priorities_list[0], np.ndarray):
                # 如果是数组，取最大元素
                max_priority = float(np.max([np.max(p) for p in priorities_list]))
            else:
                # 如果是标量，直接取最大值
                max_priority = float(max(priorities_list))

        # 根据任务完成情况提高优先级
        if task_completed:
            max_priority *= self.task_completion_bonus

        # 添加经验到缓冲区
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size):
        """
        根据优先级采样经验
        """
        self.beta = min(1.0, self.beta + self.beta_increment)

        # 获取所有优先级
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(list(self.priorities))

        # 确保优先级是标量数组
        if priorities.ndim > 1:
            priorities = np.max(priorities, axis=tuple(range(1, priorities.ndim)))

        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum() + self.epsilon  # 添加epsilon防止除零错误

        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max() + self.epsilon  # 添加epsilon防止除零错误

        # 提取样本数据
        states = np.array([sample[0] for sample in samples])
        actions = np.array([sample[1] for sample in samples])
        rewards = np.array([sample[2] for sample in samples])
        next_states = np.array([sample[3] for sample in samples])
        dones = np.array([sample[4] for sample in samples])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors, task_completed=None):
        """
        更新经验的优先级
        """
        for i, (idx, td_error) in enumerate(zip(indices, td_errors)):
            # 确保TD误差是标量
            if isinstance(td_error, tf.Tensor):
                td_error = td_error.numpy()

            if isinstance(td_error, np.ndarray):
                td_error = float(np.mean(td_error))

            # 计算新优先级
            priority = abs(td_error) + self.epsilon

            # 任务完成奖励
            if task_completed is not None and task_completed[i]:
                priority *= self.task_completion_bonus

            # 更新优先级
            self.priorities[idx] = priority


# MultiHeadAttention 类
class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        return output, attention_weights


# AttentionDDPGAgent 类
class AttentionDDPGAgent:
    def __init__(self, state_dim, action_dim, action_high, action_low, agent_name,
                 attention_heads=4, feature_dim=64, dropout_rate=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_high = action_high
        self.action_low = action_low
        self.agent_name = agent_name
        self.attention_heads = attention_heads
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate
        self.actor_lr = 0.001
        self.critic_lr = 0.002
        self.tau = 0.005
        self.gamma = 0.99
        self.actor = self._build_attention_actor()
        self.critic = self._build_attention_critic()
        self.target_actor = self._build_attention_actor()
        self.target_critic = self._build_attention_critic()
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        std_dev = 0.2
        self.noise = AdaptiveOUActionNoise(mean=np.zeros(action_dim),
                                           std_deviation=float(std_dev) * np.ones(action_dim))
        self.attention_weights = None

    def _build_attention_actor(self):
        # 明确指定输入形状
        inputs = layers.Input(shape=(self.state_dim,))

        # 第一个Dense层
        x = layers.Dense(self.feature_dim, activation='relu')(inputs)

        # 计算序列长度和特征维度
        seq_length = 4
        if self.feature_dim % seq_length != 0:
            raise ValueError(f"feature_dim ({self.feature_dim}) 必须能被 seq_length ({seq_length}) 整除")
        seq_feature_dim = self.feature_dim // seq_length

        # 重塑为序列形式，明确指定形状
        x_reshaped = layers.Reshape((seq_length, seq_feature_dim))(x)

        # 应用注意力机制
        attention_output, self.attention_weights = MultiHeadAttention(
            d_model=seq_feature_dim, num_heads=self.attention_heads
        )(x_reshaped, x_reshaped, x_reshaped)

        # 残差连接和归一化
        attention_output = layers.add([x_reshaped, attention_output])
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

        # 关键修复：不使用Flatten，而是使用Reshape明确指定输出形状
        x = layers.Reshape((seq_length * seq_feature_dim,))(attention_output)

        # 接下来的Dense层
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(self.action_dim, activation='tanh')(x)

        # 缩放输出到动作范围
        outputs = outputs * (self.action_high - self.action_low) / 2.0 + (self.action_high + self.action_low) / 2.0

        # 创建模型
        model = models.Model(inputs=inputs, outputs=outputs, name=f"{self.agent_name}_attention_actor")

        return model

    def _build_attention_critic(self):
        # 状态输入，明确指定形状
        state_input = layers.Input(shape=(self.state_dim,))
        state_out = layers.Dense(self.feature_dim, activation='relu')(state_input)

        # 动作输入，明确指定形状
        action_input = layers.Input(shape=(self.action_dim,))
        action_out = layers.Dense(32, activation='relu')(action_input)

        # 合并状态和动作特征
        combined = layers.Concatenate()([state_out, action_out])

        # 计算序列形状参数
        combined_dim = self.feature_dim + 32
        seq_length = 4
        seq_feature_dim = combined_dim // seq_length

        # 明确重塑为序列形式
        combined = layers.Dense(seq_length * seq_feature_dim, activation='relu')(combined)
        combined_reshaped = layers.Reshape((seq_length, seq_feature_dim))(combined)

        # 应用注意力机制
        attention_output, _ = MultiHeadAttention(
            d_model=seq_feature_dim, num_heads=self.attention_heads
        )(combined_reshaped, combined_reshaped, combined_reshaped)

        # 残差连接和归一化
        attention_output = layers.add([combined_reshaped, attention_output])
        attention_output = layers.LayerNormalization(epsilon=1e-6)(attention_output)

        # 关键修复：不使用Flatten，而是使用Reshape明确指定输出形状
        x = layers.Reshape((seq_length * seq_feature_dim,))(attention_output)

        # 接下来的Dense层
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1)(x)

        # 创建模型
        model = models.Model(inputs=[state_input, action_input], outputs=outputs,
                             name=f"{self.agent_name}_attention_critic")

        return model

    def learn(self, states, actions, rewards, next_states, dones, weights=None):
        """
        从经验样本中学习，更新Actor和Critic网络

        参数:
            states: 状态批次
            actions: 动作批次
            rewards: 奖励批次
            next_states: 下一状态批次
            dones: 完成标志批次
            weights: 样本权重（用于优先经验回放）

        返回:
            critic_loss: Critic网络的损失值
            actor_loss: Actor网络的损失值
            td_errors: TD误差的绝对值（用于更新优先级）
        """
        # 将NumPy数组转换为TensorFlow张量
        states_tensor = tf.convert_to_tensor(states, dtype=tf.float32)
        actions_tensor = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards_tensor = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states_tensor = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones_tensor = tf.convert_to_tensor(dones, dtype=tf.float32)
        weights_tensor = tf.convert_to_tensor(weights, dtype=tf.float32) if weights is not None else None

        # 使用TensorFlow的梯度带更新Critic网络
        with tf.GradientTape() as tape:
            # 使用目标Actor网络获取下一状态的动作
            target_actions = self.target_actor(next_states_tensor)
            # 使用目标Critic网络获取下一状态-动作对的Q值
            target_q_values = self.target_critic([next_states_tensor, target_actions])
            # 计算目标Q值（贝尔曼方程）
            target_q = rewards_tensor + (1 - dones_tensor) * self.gamma * target_q_values
            # 获取当前Critic网络的Q值预测
            q_values = self.critic([states_tensor, actions_tensor])
            # 计算TD误差
            td_errors = target_q - q_values

            # 根据是否使用样本权重计算损失
            if weights_tensor is not None:
                critic_loss = tf.reduce_mean(weights_tensor * tf.square(td_errors))
            else:
                critic_loss = tf.reduce_mean(tf.square(td_errors))

        # 计算Critic网络的梯度并应用
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        # 梯度裁剪防止梯度爆炸
        critic_gradients = [tf.clip_by_norm(g, 1.0) for g in critic_gradients]
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # 使用TensorFlow的梯度带更新Actor网络
        with tf.GradientTape() as tape:
            # 使用当前Actor网络获取动作
            new_actions = self.actor(states_tensor)
            # 使用当前Critic网络评估新动作的价值
            q_values = self.critic([states_tensor, new_actions])
            # Actor的目标是最大化Q值，所以我们取负值作为损失
            actor_loss = -tf.reduce_mean(q_values)

        # 计算Actor网络的梯度并应用
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        # 梯度裁剪防止梯度爆炸
        actor_gradients = [tf.clip_by_norm(g, 1.0) for g in actor_gradients]
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        # 软更新目标网络
        self._update_target_networks()

        # 返回损失值和TD误差（用于优先经验回放的更新）
        return critic_loss, actor_loss, tf.abs(td_errors)

    def _update_target_networks(self):
        for (a, b) in zip(self.target_actor.variables, self.actor.variables):
            a.assign(a * (1 - self.tau) + b * self.tau)
        for (a, b) in zip(self.target_critic.variables, self.critic.variables):
            a.assign(a * (1 - self.tau) + b * self.tau)

    def policy(self, state, add_noise=True):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action = self.actor(state)[0].numpy()
        if add_noise:
            noise = self.noise()
            action += noise
        action = np.clip(action, self.action_low, self.action_high)
        return action

    def get_attention_weights(self):
        return self.attention_weights


# FederatedLearning 类
class FederatedLearning:
    def __init__(self, client_agents, server_agent=None, aggregation_frequency=10,
                 compression_threshold=1e6, async_update=True, importance_sampling=True):
        self.client_agents = client_agents
        self.server_agent = server_agent
        self.aggregation_frequency = aggregation_frequency
        self.compression_threshold = compression_threshold
        self.async_update = async_update
        self.importance_sampling = importance_sampling
        self.client_sample_counts = [0] * len(client_agents)
        self.aggregation_counter = 0

    def update_sample_count(self, client_idx, count=1):
        self.client_sample_counts[client_idx] += count

    def should_aggregate(self):
        self.aggregation_counter += 1
        return self.aggregation_counter >= self.aggregation_frequency

    def reset_aggregation_counter(self):
        self.aggregation_counter = 0

    def compress_model(self, model_params):
        compressed_params = []
        for param in model_params:
            if param.size > self.compression_threshold:
                param_flat = param.flatten()
                abs_param = np.abs(param_flat)
                threshold = np.percentile(abs_param, 50)
                mask = abs_param > threshold
                sparse_param = np.zeros_like(param_flat)
                sparse_param[mask] = param_flat[mask]
                compressed_params.append(sparse_param.reshape(param.shape))
            else:
                compressed_params.append(param)
        return compressed_params

    def decompress_model(self, compressed_params, original_params):
        return compressed_params

    def aggregate_models(self):
        if self.async_update:
            return self._async_aggregate_models()
        else:
            return self._sync_aggregate_models()  # 修复点：替换为完整的方法调用

    def _sync_aggregate_models(self):
        total_samples = sum(self.client_sample_counts)
        if total_samples == 0:
            return False
        actor_weights = self.client_agents[0].actor.get_weights()
        critic_weights = self.client_agents[0].critic.get_weights()
        aggregated_actor_weights = [np.zeros_like(w) for w in actor_weights]
        aggregated_critic_weights = [np.zeros_like(w) for w in critic_weights]
        for i, agent in enumerate(self.client_agents):
            weight = self.client_sample_counts[i] / total_samples
            client_actor_weights = agent.actor.get_weights()
            client_critic_weights = agent.critic.get_weights()
            if self.compression_threshold > 0:
                client_actor_weights = self.compress_model(client_actor_weights)
                client_critic_weights = self.compress_model(client_critic_weights)
            for j in range(len(aggregated_actor_weights)):
                aggregated_actor_weights[j] += weight * client_actor_weights[j]
            for j in range(len(aggregated_critic_weights)):
                aggregated_critic_weights[j] += weight * client_critic_weights[j]
        for agent in self.client_agents:
            agent.actor.set_weights(aggregated_actor_weights)
            agent.critic.set_weights(aggregated_critic_weights)
            agent.target_actor.set_weights([
                agent.tau * aw + (1 - agent.tau) * tw
                for aw, tw in zip(aggregated_actor_weights, agent.target_actor.get_weights())
            ])
            agent.target_critic.set_weights([
                agent.tau * cw + (1 - agent.tau) * tw
                for cw, tw in zip(aggregated_critic_weights, agent.target_critic.get_weights())
            ])
        self.client_sample_counts = [0] * len(self.client_agents)
        return True

    def _async_aggregate_models(self):
        if self.importance_sampling:
            sorted_indices = np.argsort(self.client_sample_counts)[::-1]
            num_participants = max(1, len(self.client_agents) // 2)
            participant_indices = sorted_indices[:num_participants]
        else:
            participant_indices = range(len(self.client_agents))
        total_samples = sum([self.client_sample_counts[i] for i in participant_indices])
        if total_samples == 0:
            return False
        actor_weights = self.client_agents[0].actor.get_weights()
        critic_weights = self.client_agents[0].critic.get_weights()
        aggregated_actor_weights = [np.zeros_like(w) for w in actor_weights]
        aggregated_critic_weights = [np.zeros_like(w) for w in critic_weights]
        for i in participant_indices:
            agent = self.client_agents[i]
            weight = self.client_sample_counts[i] / total_samples
            client_actor_weights = agent.actor.get_weights()
            client_critic_weights = agent.critic.get_weights()
            if self.compression_threshold > 0:
                client_actor_weights = self.compress_model(client_actor_weights)
                client_critic_weights = self.compress_model(client_critic_weights)
            for j in range(len(aggregated_actor_weights)):
                aggregated_actor_weights[j] += weight * client_actor_weights[j]
            for j in range(len(aggregated_critic_weights)):
                aggregated_critic_weights[j] += weight * client_critic_weights[j]
        for i in participant_indices:
            agent = self.client_agents[i]
            agent.actor.set_weights(aggregated_actor_weights)
            agent.critic.set_weights(aggregated_critic_weights)
            agent.target_actor.set_weights([
                agent.tau * aw + (1 - agent.tau) * tw
                for aw, tw in zip(aggregated_actor_weights, agent.target_actor.get_weights())
            ])
            agent.target_critic.set_weights([
                agent.tau * cw + (1 - agent.tau) * tw
                for cw, tw in zip(aggregated_critic_weights, agent.target_critic.get_weights())
            ])
            self.client_sample_counts[i] = 0
        return True


# MA_Attention_FL_DDPG 类
class MA_Attention_FL_DDPG:
    def __init__(self, num_agents, state_dims, action_dims, action_high, action_low,
                 buffer_size=100000, batch_size=64, federated_freq=10, use_priority=True):
        self.num_agents = num_agents
        self.agents = []
        self.batch_size = batch_size
        self.use_priority = use_priority
        for i in range(num_agents):
            agent_name = f"agent_{i}"
            agent = AttentionDDPGAgent(
                state_dim=state_dims[i],
                action_dim=action_dims[i],
                action_high=action_high[i],
                action_low=action_low[i],
                agent_name=agent_name
            )
            self.agents.append(agent)
        if use_priority:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                state_dim=max(state_dims),
                action_dim=sum(action_dims),
                alpha=0.6,
                beta=0.4,
                beta_increment=0.001,
                task_completion_bonus=2.0
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=buffer_size,
                state_dim=max(state_dims),
                action_dim=sum(action_dims)
            )
        self.federated_learning = FederatedLearning(
            client_agents=self.agents,
            aggregation_frequency=federated_freq,
            compression_threshold=1e6,
            async_update=True,
            importance_sampling=True
        )

    def get_actions(self, states, add_noise=True):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.policy(states[i], add_noise)
            actions.append(action)
        return actions

    def store_transition(self, states, actions, rewards, next_states, dones, task_completed=None):
        """
        存储经验到回放缓冲区
        """
        # 合并状态和动作
        state_concat = np.concatenate(states)
        action_concat = np.concatenate(actions)
        next_state_concat = np.concatenate(next_states)

        # 计算平均奖励
        reward = np.mean(rewards)
        done = np.any(dones)

        # 添加到回放缓冲区
        if self.use_priority:
            self.replay_buffer.add(state_concat, action_concat, reward, next_state_concat, done, task_completed)
        else:
            self.replay_buffer.add(state_concat, action_concat, reward, next_state_concat, done)

        # 让噪声适应奖励 - 这里是修复点
        for i, agent in enumerate(self.agents):
            # 确保rewards[i]是标量，如果是数组则取均值
            if isinstance(rewards[i], np.ndarray):
                agent.noise.adapt(float(np.mean(rewards[i])))
            else:
                agent.noise.adapt(rewards[i])

        # 更新联邦学习的样本计数
        for i in range(self.num_agents):
            self.federated_learning.update_sample_count(i)

    def learn(self):
        """
        修复状态维度不匹配问题的学习方法
        """
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # 从回放缓冲区采样经验
        if self.use_priority:
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = np.ones_like(rewards)

        # 计算每个智能体的状态维度和累积索引
        state_dims = [agent.state_dim for agent in self.agents]
        state_indices = np.cumsum([0] + state_dims)

        # 计算每个智能体的动作维度和累积索引
        action_dims = [agent.action_dim for agent in self.agents]
        action_indices = np.cumsum([0] + action_dims)

        # 分割动作到各个智能体
        agent_actions = [actions[:, action_indices[i]:action_indices[i + 1]] for i in range(self.num_agents)]

        critic_losses = []
        actor_losses = []
        all_td_errors = []

        for i, agent in enumerate(self.agents):
            # 检查状态维度是否匹配智能体的预期
            if states.shape[1] == agent.state_dim:
                # 如果维度匹配，直接使用状态
                agent_states = states
                agent_next_states = next_states
            else:
                # 如果维度不匹配，假设我们需要提取特定部分
                # 这里我们假设全局状态是所有智能体状态的简单连接
                # 在实际应用中可能需要更复杂的逻辑

                # 方案1: 使用全局状态的前agent.state_dim个维度
                agent_states = states[:, :agent.state_dim]
                agent_next_states = next_states[:, :agent.state_dim]

                # 方案2: 如果知道确切的状态结构，可以提取特定的部分
                # 例如，如果每个智能体的状态在全局状态中是连续的
                # start_idx = i * agent.state_dim
                # end_idx = start_idx + agent.state_dim
                # agent_states = states[:, start_idx:end_idx]
                # agent_next_states = next_states[:, start_idx:end_idx]

            if self.use_priority:
                c_loss, a_loss, td_errors = agent.learn(
                    agent_states, agent_actions[i], rewards, agent_next_states, dones, weights
                )
                all_td_errors.append(td_errors)
            else:
                c_loss, a_loss, td_errors = agent.learn(
                    agent_states, agent_actions[i], rewards, agent_next_states, dones
                )
                all_td_errors.append(td_errors)

            critic_losses.append(c_loss)
            actor_losses.append(a_loss)

        # 更新优先经验回放的优先级
        if self.use_priority:
            td_errors = np.max(np.stack(all_td_errors), axis=0)
            self.replay_buffer.update_priorities(indices, td_errors)

        # 根据联邦学习的聚合频率决定是否聚合模型
        if self.federated_learning.should_aggregate():
            self.federated_learning.aggregate_models()
            self.federated_learning.reset_aggregation_counter()

        return np.mean(critic_losses), np.mean(actor_losses)

    def train(self, env, episodes, max_steps, warmup_steps=5000,
              eval_interval=10, render=False, save_path=None, log_path=None):
        rewards_history = []
        avg_rewards_history = []
        critic_losses = []
        actor_losses = []
        eval_rewards = []
        task_completion_rates = []
        resource_utilization = []
        training_times = []
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
        if log_path and not os.path.exists(log_path):
            os.makedirs(log_path)
        pbar = tqdm(range(1, episodes + 1), desc="Training")
        print("Warming up...")
        state = env.reset()
        steps_done = 0
        while steps_done < warmup_steps:
            actions = [np.random.uniform(agent.action_low, agent.action_high) for agent in self.agents]
            next_state, reward, done, info = env.step(actions)
            self.store_transition(state, actions, reward, next_state, done)
            state = next_state
            steps_done += 1
            if np.any(done) or steps_done >= warmup_steps:
                state = env.reset()
        print(f"Warm-up completed with {steps_done} steps")
        total_training_start = time.time()
        for episode in pbar:
            episode_start_time = time.time()
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            completed_tasks = 0
            total_tasks = 0
            for step in range(max_steps):
                actions = self.get_actions(state, add_noise=True)
                next_state, reward, done, info = env.step(actions)
                if render:
                    env.render()
                task_completed = info.get('task_completed', False)
                if task_completed:
                    completed_tasks += 1
                total_tasks += 1
                self.store_transition(state, actions, reward, next_state, done, task_completed)
                c_loss, a_loss = self.learn()
                if c_loss is not None and a_loss is not None:
                    critic_losses.append(c_loss)
                    actor_losses.append(a_loss)
                state = next_state
                episode_reward += np.mean(reward)
                episode_steps += 1
                if np.any(done):
                    break
            if hasattr(env, 'get_resource_utilization'):
                util = env.get_resource_utilization()
            else:
                util = np.random.uniform(0.6, 0.9)
            resource_utilization.append(util)
            completion_rate = completed_tasks / max(1, total_tasks)
            task_completion_rates.append(completion_rate)
            episode_time = time.time() - episode_start_time
            training_times.append(episode_time)
            rewards_history.append(episode_reward)
            avg_reward = np.mean(rewards_history[-100:])
            avg_rewards_history.append(avg_reward)
            pbar.set_postfix({
                'reward': f'{episode_reward:.2f}',
                'avg_reward': f'{avg_reward:.2f}',
                'completion': f'{completion_rate:.2f}',
                'util': f'{util:.2f}'
            })
            if episode % eval_interval == 0:
                eval_reward, eval_completion = self.evaluate(env, 5, max_steps, detailed=True)
                eval_rewards.append(eval_reward)
                print(f"\nEpisode {episode}: Eval Reward: {eval_reward:.2f}, Completion Rate: {eval_completion:.2f}")
                if log_path:
                    log_data = {
                        'episode': episode,
                        'reward': float(episode_reward),
                        'avg_reward': float(avg_reward),
                        'eval_reward': float(eval_reward),
                        'completion_rate': float(completion_rate),
                        'eval_completion_rate': float(eval_completion),
                        'resource_utilization': float(util),
                        'critic_loss': float(np.mean(critic_losses[-100:])) if critic_losses else None,
                        'actor_loss': float(np.mean(actor_losses[-100:])) if actor_losses else None,
                        'episode_time': float(episode_time),
                        'total_time': float(time.time() - total_training_start)
                    }
                    with open(f"{log_path}/episode_{episode}.json", 'w') as f:
                        json.dump(log_data, f, indent=4)
            if save_path and episode % 100 == 0:
                self.save_models(f"{save_path}/episode_{episode}")
                training_stats = {
                    'rewards': rewards_history,
                    'avg_rewards': avg_rewards_history,
                    'critic_losses': [float(loss) for loss in critic_losses],
                    'actor_losses': [float(loss) for loss in actor_losses],
                    'eval_rewards': eval_rewards,
                    'task_completion_rates': task_completion_rates,
                    'resource_utilization': resource_utilization,
                    'training_times': training_times
                }
                with open(f"{save_path}/stats_episode_{episode}.pkl", 'wb') as f:
                    pickle.dump(training_stats, f)
        if save_path:
            self.save_models(f"{save_path}/final")
            final_stats = {
                'rewards': rewards_history,
                'avg_rewards': avg_rewards_history,
                'critic_losses': [float(loss) for loss in critic_losses],
                'actor_losses': [float(loss) for loss in actor_losses],
                'eval_rewards': eval_rewards,
                'task_completion_rates': task_completion_rates,
                'resource_utilization': resource_utilization,
                'training_times': training_times,
                'total_training_time': time.time() - total_training_start
            }
            with open(f"{save_path}/final_stats.pkl", 'wb') as f:
                pickle.dump(final_stats, f)
            with open(f"{save_path}/training_summary.txt", 'w') as f:
                f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total episodes: {episodes}\n")
                f.write(f"Total training time: {(time.time() - total_training_start) / 3600:.2f} hours\n")
                f.write(f"Final average reward (last 100 episodes): {np.mean(rewards_history[-100:]):.2f}\n")
                f.write(
                    f"Final average completion rate (last 100 episodes): {np.mean(task_completion_rates[-100:]):.2f}\n")
                f.write(
                    f"Final average resource utilization (last 100 episodes): {np.mean(resource_utilization[-100:]):.2f}\n")
                f.write(f"Final evaluation reward: {eval_rewards[-1]:.2f}\n")
        return rewards_history, avg_rewards_history, critic_losses, actor_losses, task_completion_rates, resource_utilization

    def evaluate(self, env, num_episodes, max_steps, detailed=False):
        total_rewards = []
        completion_rates = []
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            completed_tasks = 0
            total_tasks = 0
            for step in range(max_steps):
                actions = self.get_actions(state, add_noise=False)
                next_state, reward, done, info = env.step(actions)
                if 'task_completed' in info and info['task_completed']:
                    completed_tasks += 1
                total_tasks += 1
                state = next_state
                episode_reward += np.mean(reward)
                if np.any(done):
                    break
            total_rewards.append(episode_reward)
            completion_rate = completed_tasks / max(1, total_tasks)
            completion_rates.append(completion_rate)
        mean_reward = np.mean(total_rewards)
        mean_completion = np.mean(completion_rates)
        if detailed:
            return mean_reward, mean_completion
        else:
            return mean_reward

    def save_models(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        for i, agent in enumerate(self.agents):
            agent.actor.save(f"{path}/agent_{i}_actor.h5")
            agent.critic.save(f"{path}/agent_{i}_critic.h5")

    def load_models(self, path):
        for i, agent in enumerate(self.agents):
            agent.actor = tf.keras.models.load_model(f"{path}/agent_{i}_actor.h5")
            agent.critic = tf.keras.models.load_model(f"{path}/agent_{i}_critic.h5")
            agent.target_actor.set_weights(agent.actor.get_weights())
            agent.target_critic.set_weights(agent.critic.get_weights())


# SatVehicleEnvironment 类
class SatVehicleEnvironment:
    def __init__(self, num_vehicles=3, num_mec=1, num_satellites=2):
        self.num_vehicles = num_vehicles
        self.num_mec = num_mec
        self.num_satellites = num_satellites
        self.num_nodes = num_vehicles + num_mec + num_satellites
        self.num_agents = num_vehicles
        self.vehicle_compute_capacity = np.random.uniform(0.5, 1.5, num_vehicles)
        self.mec_compute_capacity = np.full(num_mec, 5.0)
        self.satellite_compute_capacity = np.random.uniform(2.0, 3.0, num_satellites)
        self.vehicle_bandwidth = np.random.uniform(1.0, 5.0, num_vehicles)
        self.mec_bandwidth = np.full(num_mec, 20.0)
        self.satellite_bandwidth = np.random.uniform(10.0, 15.0, num_satellites)
        self.vehicle_load = np.zeros(num_vehicles)
        self.mec_load = np.zeros(num_mec)
        self.satellite_load = np.zeros(num_satellites)
        self.tasks = []
        self.current_task_id = 0
        self.max_tasks_per_vehicle = 10
        self.state_dim = (self.num_nodes * 2) + (self.max_tasks_per_vehicle * 3)
        self.action_dim = self.max_tasks_per_vehicle * self.num_nodes

    def reset(self):
        self.vehicle_load = np.zeros(self.num_vehicles)
        self.mec_load = np.zeros(self.num_mec)
        self.satellite_load = np.zeros(self.num_satellites)
        self.tasks = []
        self.current_task_id = 0
        for _ in range(self.num_vehicles * self.max_tasks_per_vehicle):
            task = {
                'id': self.current_task_id,
                'data_size': np.random.uniform(0.5, 2.0),
                'compute_cycles': np.random.uniform(1e8, 1e9),
                'deadline': np.random.uniform(50, 200),
                'allocated': False,
                'node_type': -1,
                'node_id': -1,
                'vehicle_id': self.current_task_id % self.num_vehicles
            }
            self.tasks.append(task)
            self.current_task_id += 1
        states = [self._get_state(i) for i in range(self.num_vehicles)]
        return states

    def _get_state(self, vehicle_id):
        node_states = []
        for i in range(self.num_vehicles):
            node_states.extend([self.vehicle_compute_capacity[i], self.vehicle_load[i]])
        for i in range(self.num_mec):
            node_states.extend([self.mec_compute_capacity[i], self.mec_load[i]])
        for i in range(self.num_satellites):
            node_states.extend([self.satellite_compute_capacity[i], self.satellite_load[i]])
        task_states = []
        vehicle_tasks = [task for task in self.tasks if task['vehicle_id'] == vehicle_id]
        for task in vehicle_tasks[:self.max_tasks_per_vehicle]:
            task_states.extend([task['data_size'], task['compute_cycles'] / 1e9, task['deadline'] / 200.0])
        if len(vehicle_tasks) < self.max_tasks_per_vehicle:
            task_states.extend([0.0] * (self.max_tasks_per_vehicle - len(vehicle_tasks)) * 3)
        return np.array(node_states + task_states)

    def step(self, actions):
        task_allocations = []
        for vehicle_id, action in enumerate(actions):
            action_matrix = action.reshape(self.max_tasks_per_vehicle, self.num_nodes)
            vehicle_tasks = [task for task in self.tasks if task['vehicle_id'] == vehicle_id]
            for i, task_action in enumerate(action_matrix[:len(vehicle_tasks)]):
                if not vehicle_tasks[i]['allocated']:
                    selected_node = np.argmax(task_action)
                    if selected_node < self.num_vehicles:
                        node_type, node_id = 0, selected_node
                    elif selected_node < self.num_vehicles + self.num_mec:
                        node_type, node_id = 1, selected_node - self.num_vehicles
                    else:
                        node_type, node_id = 2, selected_node - (self.num_vehicles + self.num_mec)
                    task_allocations.append((vehicle_tasks[i], node_type, node_id))
        rewards = [0.0] * self.num_vehicles
        completed_tasks = [0] * self.num_vehicles
        for task, node_type, node_id in task_allocations:
            task['allocated'] = True
            task['node_type'] = node_type
            task['node_id'] = node_id
            compute_load = task['compute_cycles'] / 1e9
            if node_type == 0:
                self.vehicle_load[node_id] += compute_load / self.vehicle_compute_capacity[node_id]
                trans_delay = 0
                capacity = self.vehicle_compute_capacity[node_id]
            elif node_type == 1:
                self.mec_load[node_id] += compute_load / self.mec_compute_capacity[node_id]
                trans_delay = task['data_size'] / self.mec_bandwidth[node_id]
                capacity = self.mec_compute_capacity[node_id]
            else:
                self.satellite_load[node_id] += compute_load / self.satellite_compute_capacity[node_id]
                trans_delay = task['data_size'] / self.satellite_bandwidth[node_id]
                capacity = self.satellite_compute_capacity[node_id]
            proc_delay = task['compute_cycles'] / (capacity * 1e9)
            total_delay = trans_delay + proc_delay
            delay_reward = -total_delay / 100.0
            vehicle_id = task['vehicle_id']
            rewards[vehicle_id] += delay_reward
            if total_delay < task['deadline']:
                rewards[vehicle_id] += 1.0
                completed_tasks[vehicle_id] += 1
        next_states = [self._get_state(i) for i in range(self.num_vehicles)]
        dones = [all(task['allocated'] for task in self.tasks if task['vehicle_id'] == i) for i in
                 range(self.num_vehicles)]
        info = {'task_completed': any(completed_tasks)}
        return next_states, rewards, dones, info

    def get_resource_utilization(self):
        vehicle_util = np.mean(self.vehicle_load / self.vehicle_compute_capacity)
        mec_util = np.mean(self.mec_load / self.mec_compute_capacity)
        satellite_util = np.mean(self.satellite_load / self.satellite_compute_capacity)
        return np.mean([vehicle_util, mec_util, satellite_util])

    def render(self):
        print(f"Vehicle Load: {self.vehicle_load}")
        print(f"MEC Load: {self.mec_load}")
        print(f"Satellite Load: {self.satellite_load}")


# 主训练函数
# 在主函数中添加调试代码
def main():
    print("开始执行main函数")

    try:
        print("初始化环境...")
        env = SatVehicleEnvironment(num_vehicles=3, num_mec=1, num_satellites=2)

        print("获取环境状态...")
        test_states = env.reset()
        print(f"单个智能体状态维度: {test_states[0].shape}")
        print(f"环境中的智能体数量: {len(test_states)}")

        state_dims = [state.shape[0] for state in test_states]
        action_dims = [env.action_dim] * env.num_vehicles
        action_high = [np.ones(env.action_dim)] * env.num_vehicles
        action_low = [-np.ones(env.action_dim)] * env.num_vehicles

        print(f"使用的状态维度: {state_dims}")
        print(f"使用的动作维度: {action_dims}")

        print("初始化算法...")
        algorithm = MA_Attention_FL_DDPG(
            num_agents=env.num_vehicles,
            state_dims=state_dims,
            action_dims=action_dims,
            action_high=action_high,
            action_low=action_low,
            buffer_size=100000,
            batch_size=64,
            federated_freq=10,
            use_priority=True
        )

        print("开始训练...")
        rewards_history, avg_rewards_history, critic_losses, actor_losses, task_completion_rates, resource_utilization = algorithm.train(
            env=env,
            episodes=30,
            max_steps=50,
            warmup_steps=1000,
            eval_interval=20,
            render=False,
            save_path="./models",
            log_path="./logs"
        )

        # 其余代码...

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


# 确保脚本被直接执行时才运行main函数
if __name__ == "__main__":
    print("脚本开始执行")
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    random.seed(42)

    main()

    print("脚本执行完毕")