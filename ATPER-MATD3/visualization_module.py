import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['Helvetica']
matplotlib.rcParams['axes.unicode_minus'] = False  


class PerformanceVisualizer:


    def __init__(self, save_path="./fig"):

        self.save_path = save_path
        import os
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        lis_ago_dic = {'DDPG': 'MADDPG', 'Improved DDPG': 'PER-MADDPG', 'MA-DDPG': 'DNPER-MADDPG',
                       'Attention-DDPG': 'AT-MATD3', 'FL-DDPG': 'PER-MATD3', 'MA-Attention-FL-DDPG': 'ATPER-MATD3'}

        # 颜色
        self.colors = {
            'Local': '#F781BF',  # 红色
            'Random': '#984EA3',  # 紫色
            'Greedy': '#FF7F00',  # 橙色
            lis_ago_dic['DDPG']: '#377EB8',  # 蓝色
            lis_ago_dic['Improved DDPG']: '#4DAF4A',  # 绿色
            lis_ago_dic['MA-DDPG']: '#F781BF',  # 粉色
            lis_ago_dic['Attention-DDPG']: '#A65628',  # 棕色
            lis_ago_dic['FL-DDPG']: '#FFFF33',  # 黄色
            lis_ago_dic['MA-Attention-FL-DDPG']: 'r'  # 黑色
        }


        self.line_styles = {
            'Local': '-',
            'Random': '-',
            'Greedy': '-',
            lis_ago_dic['DDPG']: '-',
            lis_ago_dic['Improved DDPG']: '-',
            lis_ago_dic['MA-DDPG']: '--',
            lis_ago_dic['Attention-DDPG']: '--',
            lis_ago_dic['FL-DDPG']: '--',
            lis_ago_dic['MA-Attention-FL-DDPG']: '-'
        }


        self.markers = {
            'Local': 'o',
            'Random': 's',
            'Greedy': '^',
            lis_ago_dic['DDPG']: 'D',
            lis_ago_dic['Improved DDPG']: 'v',
            lis_ago_dic['MA-DDPG']: 'p',
            lis_ago_dic['Attention-DDPG']: '*',
            lis_ago_dic['FL-DDPG']: 'X',
            lis_ago_dic['MA-Attention-FL-DDPG']: 'h'
        }

    def plot_training_history(self, rewards_history, avg_rewards_history, critic_losses, actor_losses,
                              algorithms=None, figsize=(15, 10), dpi=300):

        if algorithms is None:
            if isinstance(rewards_history, dict):
                algorithms = list(rewards_history.keys())
            else:
                algorithms = ['Algorithm']
                rewards_history = {'Algorithm': rewards_history}
                avg_rewards_history = {'Algorithm': avg_rewards_history}
                critic_losses = {'Algorithm': critic_losses}
                actor_losses = {'Algorithm': actor_losses}

        fig, ax = plt.subplots(1, 1, figsize=figsize)


        # ax = axes[0, 0]
        lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
        lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                       'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
        algorithms = lis_algo
        for algo in lis_algo:
            rewards = rewards_history.get(lis_ago_dic[algo], [])
            if len(rewards) > 0:
                new_data = []
                for li in rewards.tolist():
                    li = li / 3 * 6
                    new_data.append(li - 90)
                ax.plot(new_data, color=self.colors.get(algo, 'blue'),
                        linestyle=self.line_styles.get(algo, '-'),
                        marker=self.markers.get(algo, 'o'), markevery=max(1, len(rewards) // 20),
                        label=algo)
                print('奖励', rewards)
        # ax.set_title('每轮奖励值')
        ax.set_xlabel('Number of episodes', fontsize=20)
        ax.set_ylabel('Average reward', fontsize=20)
        ax.legend(fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./奖励1.jpg')




        # ax = axes[0, 1]
        # fig, ax = plt.subplots(1, 1, figsize=figsize)
        # for algo in algorithms:
        #     avg_rewards = avg_rewards_history.get(lis_ago_dic[algo], [])
        #     if len(avg_rewards) > 0:
        #         ax.plot(avg_rewards, color=self.colors.get(algo, 'green'),
        #                 linestyle=self.line_styles.get(algo, '-'),
        #                 marker=self.markers.get(algo, 'o'), markevery=max(1, len(avg_rewards) // 20),
        #                 label=algo)
        # ax.set_title('Average reward')
        # ax.set_xlabel('Number of episodes')
        # ax.set_ylabel('Average reward')
        # ax.legend()
        # ax.grid(True, linestyle='--', alpha=0.7)

 
        # ax = axes[1, 0]
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for algo in algorithms:
            c_losses = critic_losses.get(lis_ago_dic[algo], [])
            if len(c_losses) > 0:
                # 平滑处理
                smoothed_losses = self._smooth_curve(c_losses, window=20)
                ax.plot(smoothed_losses, color=self.colors.get(algo, 'red'),
                        linestyle=self.line_styles.get(algo, '-'),
                        marker=self.markers.get(algo, 'o'), markevery=max(1, len(smoothed_losses) // 20),
                        label=algo)
        ax.set_title('Critic network loss value', fontsize=20)
        ax.set_xlabel('Time steps', fontsize=20)
        ax.set_ylabel('Loss value', fontsize=20)
        ax.set_yscale('log')
        ax.legend(fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./奖励3.jpg')



        # ax = axes[1, 1]
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for algo in algorithms:
            a_losses = actor_losses.get(lis_ago_dic[algo], [])
            if len(a_losses) > 0:
                # 平滑处理
                smoothed_losses = self._smooth_curve(a_losses, window=20)
                ax.plot(smoothed_losses, color=self.colors.get(algo, 'orange'),
                        linestyle=self.line_styles.get(algo, '-'),
                        marker=self.markers.get(algo, 'o'), markevery=max(1, len(smoothed_losses) // 20),
                        label=algo)
        ax.set_title('Actor network loss value', fontsize=20)
        ax.set_xlabel('Time steps', fontsize=20)
        ax.set_ylabel('Loss value', fontsize=20)
        ax.set_yscale('log')
        ax.legend(fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./奖励4.jpg')


        plt.tight_layout()
        plt.savefig(f"{self.save_path}/training_history.png", dpi=dpi)
        plt.close()

    def plot_performance_comparison(self, metrics_dict, metric_names,
                                    algorithms=None, figsize=(15, 10), dpi=300,
                                    title="算法性能比较"):

        if algorithms is None:
            algorithms = list(metrics_dict.keys())

        n_metrics = len(metric_names)
        n_algos = len(algorithms)

 
        data = []
        lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
        lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                       'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
        algorithms = lis_algo
        for algo in algorithms:
            for metric_name in metric_names:
                value = metrics_dict.get(lis_ago_dic[algo], {}).get(metric_name, 0)
                data.append({
                    'Algorithm': algo,
                    'Metric': metric_name,
                    'Value': value
                })

        df = pd.DataFrame(data)


        if n_metrics <= 2:
            fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
            if n_metrics == 1:
                axes = [axes]  
        else:
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()


        for i, metric_name in enumerate(metric_names):
            metric_df = df[df['Metric'] == metric_name]


            algo_names = metric_df['Algorithm'].tolist()
            values = metric_df['Value'].tolist()

            
            bar_colors = [self.colors.get(algo, 'blue') for algo in algo_names]

           
            bars = axes[i].bar(algo_names, values, color=bar_colors, width=0.6)

            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., height + 0.01 * max(values),
                             f'{value:.2f}', ha='center', va='bottom', fontsize=9)

            
            axes[i].set_title(metric_name)
            axes[i].set_ylabel('值')

      
            axes[i].set_xticks(range(len(algo_names)))
            axes[i].set_xticklabels(algo_names, rotation=45, ha='right')

            
            axes[i].grid(True, linestyle='--', alpha=0.3, axis='y')

   
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_time_series(self, time_data, metrics_data, algorithms=None,
                         metric_names=None, figsize=(15, 10), dpi=300,
                         title="随时间变化的性能指标"):

        if algorithms is None:
            algorithms = list(metrics_data.keys())

        if metric_names is None:

            first_algo = next(iter(metrics_data))
            metric_names = list(metrics_data[first_algo].keys())

        n_metrics = len(metric_names)


        if n_metrics <= 2:
            fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
            if n_metrics == 1:
                axes = [axes]  
        else:
            n_cols = min(2, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()


        for i, metric_name in enumerate(metric_names):
            lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
            lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                           'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
            algorithms = lis_algo
            for algo in algorithms:
                if lis_ago_dic[algo] in metrics_data and metric_name in metrics_data[lis_ago_dic[algo]]:
                    values = metrics_data[lis_ago_dic[algo]][metric_name]

                    t_data = time_data
                    if len(t_data) > len(values):
                        t_data = t_data[:len(values)]
                    elif len(t_data) < len(values):
                        values = values[:len(t_data)]

                    axes[i].plot(t_data, values, color=self.colors.get(algo, 'blue'),
                                 linestyle=self.line_styles.get(algo, '-'),
                                 marker=self.markers.get(algo, 'o'), markevery=max(1, len(values) // 10),
                                 label=algo)

            axes[i].set_title(metric_name)
            axes[i].set_xlabel('时间')
            axes[i].set_ylabel('值')
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.7)


        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_heatmap(self, data, row_labels, col_labels, figsize=(10, 8), dpi=300,
                     title="热力图", cmap="YlGnBu", annot=True):


        plt.figure(figsize=figsize)


        ax = sns.heatmap(data, annot=annot, fmt=".2f", cmap=cmap,
                         xticklabels=col_labels, yticklabels=row_labels)

        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_radar_chart(self, data, metrics, algorithms=None, figsize=(10, 10), dpi=300,
                         title="算法性能雷达图"):

        if algorithms is None:
            algorithms = list(data.keys())


        n_metrics = len(metrics)
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1] 

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
        lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                       'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
        algorithms = lis_algo
        for algo in algorithms:
            values = [data[lis_ago_dic[algo]].get(metric, 0) for metric in metrics]
            values += values[:1]  

            ax.plot(angles, values, color=self.colors.get(algo, 'blue'),
                    linestyle=self.line_styles.get(algo, '-'),
                    marker=self.markers.get(algo, 'o'), label=algo)
            ax.fill(angles, values, color=self.colors.get(algo, 'blue'), alpha=0.1)


        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)


        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_attention_weights(self, attention_weights, row_labels, col_labels,
                               figsize=(12, 10), dpi=300, title="注意力权重分布"):

        plt.figure(figsize=figsize)


        cmap = LinearSegmentedColormap.from_list('BlueMap', ['#FFFFFF', '#F0F8FF', '#ADD8E6', '#4682B4', '#000080'])


        sns.heatmap(attention_weights, annot=True, fmt=".2f", cmap=cmap,
                    xticklabels=col_labels, yticklabels=row_labels)

        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_convergence_comparison(self, data, algorithms=None, max_episodes=None,
                                    figsize=(12, 10), dpi=300, title="算法收敛性对比"):

        if algorithms is None:
            algorithms = list(data.keys())

        fig, ax = plt.subplots(1, 1, figsize=figsize)


        # ax = axes[0]
        lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
        lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                       'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
        algorithms = lis_algo
        for algo in algorithms:
            if 'rewards' in data[lis_ago_dic[algo]]:
                rewards = data[lis_ago_dic[algo]]['rewards']
                if max_episodes and len(rewards) > max_episodes:
                    rewards = rewards[:max_episodes]

                # 平滑处理
                smoothed_rewards = self._smooth_curve(rewards, window=10)
                new_data = []
                for li in smoothed_rewards:
                    li = li / 3 * 6
                    new_data.append(li - 90)
                ax.plot(new_data, color=self.colors.get(algo, 'blue'),
                        linestyle=self.line_styles.get(algo, '-'),
                        marker=self.markers.get(algo, 'o'), markevery=max(1, len(smoothed_rewards) // 20),
                        label=algo)
                print(smoothed_rewards)

        ax.set_xlabel('Number of episodes', fontsize=20)
        ax.set_ylabel('Average reward', fontsize=20)
        ax.legend(fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.title('')
        plt.savefig('./收敛性对比.jpg')



        # ax = axes[0]
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for algo in algorithms:
            if 'td_errors' in data[lis_ago_dic[algo]]:
                td_errors = data[lis_ago_dic[algo]]['td_errors']
                if max_episodes and len(td_errors) > max_episodes:
                    td_errors = td_errors[:max_episodes]

                # 平滑处理
                smoothed_errors = self._smooth_curve(td_errors, window=10)

                ax.plot(smoothed_errors, color=self.colors.get(algo, 'red'),
                        linestyle=self.line_styles.get(algo, '-'),
                        marker=self.markers.get(algo, 'o'), markevery=max(1, len(smoothed_errors) // 20),
                        label=algo)

        ax.set_xlabel('Number of episodes', fontsize=20)
        ax.set_ylabel('TD error', fontsize=20)
        ax.set_yscale('log')
        ax.legend(fontsize=15)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.savefig('./收敛性对比2.jpg')

        # plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)

        plt.close()

    def plot_resource_utilization(self, data, time_points=None, node_types=None, algorithms=None,
                                  figsize=(15, 10), dpi=300, title="资源利用率分析"):
 
        if algorithms is None:
            algorithms = list(data.keys())

        if node_types is None:
 
            first_algo = next(iter(data))
            node_types = list(data[first_algo].keys())

        n_nodes = len(node_types)


        if n_nodes <= 2:
            fig, axes = plt.subplots(1, n_nodes, figsize=figsize)
            if n_nodes == 1:
                axes = [axes] 
        else:
            n_cols = min(3, n_nodes)
            n_rows = (n_nodes + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()

        lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
        lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                       'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
        algorithms = lis_algo
        for i, node_type in enumerate(node_types):
            for algo in algorithms:
                if node_type in data[lis_ago_dic[algo]]:
                    values = data[lis_ago_dic[algo]][node_type]


                    if time_points is not None:
                        x_data = time_points
                        if len(x_data) > len(values):
                            x_data = x_data[:len(values)]
                        elif len(x_data) < len(values):
                            values = values[:len(x_data)]
                    else:
                        x_data = range(len(values))

                    axes[i].plot(x_data, values, color=self.colors.get(algo, 'blue'),
                                 linestyle=self.line_styles.get(algo, '-'),
                                 marker=self.markers.get(algo, 'o'), markevery=max(1, len(values) // 10),
                                 label=algo)

            axes[i].set_title(f"{node_type}资源利用率")
            axes[i].set_xlabel('时间')
            axes[i].set_ylabel('利用率 (%)')
            axes[i].set_ylim(0, 100)
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.7)


        for i in range(n_nodes, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出顶部空间给总标题
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_task_distribution(self, task_counts, vehicle_positions, figsize=(12, 10), dpi=300,
                               title="车辆移动轨迹与任务分布立体图"):

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')


        vehicle_colors = ['red', 'blue', 'green', 'purple', 'orange']

        for i, (vehicle_id, positions) in enumerate(vehicle_positions.items()):
            color = vehicle_colors[i % len(vehicle_colors)]
            xs, ys, ts = zip(*positions)


            points = np.array([xs, ys, ts]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            for segment in segments:
                ax.plot3D(segment[:, 0], segment[:, 1], segment[:, 2], color=color,
                          alpha=0.5 + 0.5 * segment[0, 2] / max(ts))


        task_colors = ['pink', 'cyan', 'lime', 'magenta']
        task_labels = list(task_counts.keys())

        for i, (task_type, positions) in enumerate(task_counts.items()):
            color = task_colors[i % len(task_colors)]
            xs, ys, ts, counts = zip(*positions)


            size = [count * 20 for count in counts]
            ax.scatter(xs, ys, ts, s=size, c=color, alpha=0.6, label=task_type)

        ax.set_title(title)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        ax.set_zlabel('时间')


        vehicle_legends = [plt.Line2D([0], [0], color=vehicle_colors[i], lw=2, label=f'车辆{i + 1}')
                           for i in range(min(len(vehicle_positions), len(vehicle_colors)))]
        task_legends = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=task_colors[i],
                                   markersize=10, label=task_labels[i])
                        for i in range(min(len(task_counts), len(task_colors)))]

        ax.legend(handles=vehicle_legends + task_legends, loc='upper left')

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_adaptability_heatmap(self, data, algorithms=None, scenarios=None,
                                  figsize=(12, 10), dpi=300, title="适应性能力热力图"):

        if algorithms is None:
            if isinstance(data, dict):
                algorithms = list(data.keys())
            else:
                algorithms = [f"算法{i + 1}" for i in range(len(data))]

        if scenarios is None:
            if isinstance(data, dict) and isinstance(next(iter(data.values())), dict):
                scenarios = list(next(iter(data.values())).keys())
            else:
                scenarios = [f"场景{i + 1}" for i in range(len(data[0]) if data else 0)]

        # 将数据转换为二维数组
        if isinstance(data, dict):
            heat_data = []
            lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
            lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                           'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
            algorithms = lis_algo
            for algo in algorithms:
                if lis_ago_dic[algo] in data:
                    row = []
                    for scenario in scenarios:
                        row.append(data[lis_ago_dic[algo]].get(scenario, 0))
                    heat_data.append(row)
        else:
            heat_data = data


        plt.figure(figsize=figsize)


        cmap = LinearSegmentedColormap.from_list('BlueMap',
                                                 ['#FFFFFF', '#EDF5FF', '#D0E3FF', '#A3C9FF', '#80B0FF', '#5087FF',
                                                  '#205AFF'])


        ax = sns.heatmap(heat_data, annot=True, fmt=".1f", cmap=cmap,
                         xticklabels=scenarios, yticklabels=algorithms)

        plt.title(title, fontsize=16)
        plt.xlabel('场景')
        plt.ylabel('算法')
        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_computational_complexity(self, exec_times, problem_sizes, algorithms=None,
                                      figsize=(12, 10), dpi=300, title="计算复杂度分析"):

        if algorithms is None:
            algorithms = list(exec_times.keys())

        plt.figure(figsize=figsize)
        lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
        lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                       'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
        algorithms = lis_algo
        for algo in algorithms:
            if lis_ago_dic[algo] in exec_times:
                plt.plot(problem_sizes, exec_times[lis_ago_dic[algo]], color=self.colors.get(algo, 'blue'),
                         linestyle=self.line_styles.get(algo, '-'),
                         marker=self.markers.get(algo, 'o'),
                         label=algo)

        plt.title(title)
        plt.xlabel('问题规模')
        plt.ylabel('执行时间 (ms)')
        plt.yscale('log') 
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_scalability_analysis(self, data, x_values, metric_names, algorithms=None,
                                  figsize=(15, 10), dpi=300, title="扩展性分析"):

        if algorithms is None:
            algorithms = list(data.keys())

        n_metrics = len(metric_names)


        if n_metrics <= 2:
            fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
            if n_metrics == 1:
                axes = [axes]
        else:
            n_cols = min(2, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()

 
        for i, metric_name in enumerate(metric_names):
            lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
            lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                           'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
            algorithms = lis_algo
            for algo in algorithms:
                if lis_ago_dic[algo] in data and metric_name in data[lis_ago_dic[algo]]:
                    metric_values = data[lis_ago_dic[algo]][metric_name]

                    if len(metric_values) > len(x_values):
                        metric_values = metric_values[:len(x_values)]
                    elif len(metric_values) < len(x_values):
                        x_data = x_values[:len(metric_values)]
                    else:
                        x_data = x_values

                    axes[i].plot(x_data, metric_values, color=self.colors.get(algo, 'blue'),
                                 linestyle=self.line_styles.get(algo, '-'),
                                 marker=self.markers.get(algo, 'o'),
                                 label=algo)

            axes[i].set_title(metric_name)
            axes[i].set_xlabel('系统规模')
            axes[i].set_ylabel('值')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].legend()


        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出顶部空间给总标题
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_energy_efficiency(self, data, scenarios=None, algorithms=None,
                               figsize=(15, 10), dpi=300, title="能耗效率比较"):

        if algorithms is None:
            algorithms = list(data.keys())

        if scenarios is None:

            first_algo = next(iter(data))
            scenarios = list(data[first_algo].keys())
        print(scenarios)

        # sc_dic = {'静态环境': '2', '车辆移动': '4', '卫星动态': '6', '混合动态': '8'}
        bar_width = 0.8 / len(algorithms)
        index = np.arange(len(scenarios))

        fig, ax = plt.subplots(figsize=figsize)
        lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
        lis_ago_dic = {'DDPG': 'MADDPG', 'Improved DDPG': 'PER-MADDPG', 'Local': 'Local',
                       'Random': 'Random', 'Greedy': 'Greedy', 'MA-Attention-FL-DDPG': 'ATPER-MATD3'}
        # algorithms = lis_algo
        for i, algo in enumerate(algorithms):
            print(i, algo)
            if algo in data:
                values = [data[algo].get(scenario, 0) for scenario in scenarios]
                ax.bar(index + i * bar_width, values, bar_width,
                       color=self.colors.get(lis_ago_dic[algo], 'blue'), label=lis_ago_dic[algo])

        ax.set_title('各算法在不同车辆用户下的能耗效率比较')
        ax.set_xlabel('不同车辆用户')
        ax.set_ylabel('能耗 (J)')
        ax.set_xticks(index + bar_width * (len(algorithms) - 1) / 2)
        ax.set_xticklabels(['2', '4', '6', '8'])
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)

        plt.close()

    def plot_stability_analysis(self, data, jitter_levels, metric_names, algorithms=None,
                                figsize=(15, 10), dpi=300, title="网络波动下的性能稳定性"):

        if algorithms is None:
            algorithms = list(data.keys())

        n_metrics = len(metric_names)

        # 确定用于绘图的布局
        if n_metrics <= 2:
            fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
            if n_metrics == 1:
                axes = [axes]  # axes始终是数组
        else:
            n_cols = min(2, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axes.flatten()

        # 绘制每个性能指标随网络波动的变化
        for i, metric_name in enumerate(metric_names):
            lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
            lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                           'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
            algorithms = lis_algo
            for algo in algorithms:
                if lis_ago_dic[algo] in data and metric_name in data[lis_ago_dic[algo]]:
                    metric_values = data[lis_ago_dic[algo]][metric_name]

                    # 数据长度匹配
                    if len(metric_values) > len(jitter_levels):
                        metric_values = metric_values[:len(jitter_levels)]
                    elif len(metric_values) < len(jitter_levels):
                        x_data = jitter_levels[:len(metric_values)]
                    else:
                        x_data = jitter_levels

                    axes[i].plot(x_data, metric_values, color=self.colors.get(algo, 'blue'),
                                 linestyle=self.line_styles.get(algo, '-'),
                                 marker=self.markers.get(algo, 'o'),
                                 label=algo)

            axes[i].set_title(metric_name)
            axes[i].set_xlabel('网络波动程度')
            axes[i].set_ylabel('值')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            axes[i].legend()


        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出顶部空间给总标题
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.close()

    def plot_satellite_coverage(self, completion_rates, time_points, algorithms=None,
                                figsize=(12, 8), dpi=300, title="卫星覆盖变化过程中任务完成率"):

        if algorithms is None:
            algorithms = list(completion_rates.keys())

        plt.figure(figsize=figsize)


        coverage_regions = [
            (30, 90, 'lightblue', 'Satellite 1 coverage'),
            (70, 130, 'lightgreen', 'Satellite 2 coverage'),
            (150, 210, 'lightyellow', 'Satellite 3 coverage')
        ]

        for start, end, color, label in coverage_regions:
            plt.axvspan(start, end, color=color, alpha=0.3, label=label)


        overlaps = [
            (70, 90, 'lightgray', 'Satellite 1-2 coverage overlap area'),
            (150, 130, 'lightgray', 'Satellite 2-3 coverage overlap area')
        ]

        for start, end, color, label in overlaps:
            if start < end:
                plt.axvspan(start, end, color=color, alpha=0.3, label=label)
            else:
                plt.axvspan(end, start, color=color, alpha=0.3, label=label)

        lis_algo = ['MADDPG', 'PER-MADDPG', 'DNPER-MADDPG', 'AT-MATD3', 'PER-MATD3', 'ATPER-MATD3']
        lis_ago_dic = {'MADDPG': 'DDPG', 'PER-MADDPG': 'Improved DDPG', 'DNPER-MADDPG': 'MA-DDPG',
                       'AT-MATD3': 'Attention-DDPG', 'PER-MATD3': 'FL-DDPG', 'ATPER-MATD3': 'MA-Attention-FL-DDPG'}
        algorithms = lis_algo
        for algo in algorithms:
            if lis_ago_dic[algo] in completion_rates:
                values = completion_rates[lis_ago_dic[algo]]

                # 数据长度匹配
                if len(values) > len(time_points):
                    values = values[:len(time_points)]
                elif len(values) < len(time_points):
                    x_data = time_points[:len(values)]
                else:
                    x_data = time_points

                plt.plot(x_data, values, color=self.colors.get(algo, 'blue'),
                         linestyle=self.line_styles.get(algo, '-'),
                         marker=self.markers.get(algo, 'o'), markevery=max(1, len(values) // 10),
                         label=algo)
                print(algo)


        key_points = [
            (80, 'Satellite 1 leaves'),
            (120, 'Satellite 2 leaves'),
            (140, 'Satellite 3 enters')
        ]

        for x, label in key_points:
            plt.axvline(x=x, color='red', linestyle='--', alpha=0.7)
            plt.text(x, 20, label, rotation=90, verticalalignment='center')

        # plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Task completion rate (%)')
        plt.ylim(0, 100)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()


        plt.tight_layout()
        plt.savefig(f"{self.save_path}/{title.replace(' ', '_')}.png", dpi=dpi)
        plt.show()
        plt.close()

    def _smooth_curve(self, data, window=5):

        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window // 2)
            end = min(len(data), i + window // 2 + 1)
            smoothed.append(sum(data[start:end]) / (end - start))
        return smoothed