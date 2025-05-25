import random
import math
import time

import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# === 新增字体配置 ===
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# =====================
class TS(object):
    def __init__(self, num_city, data, optimal_length=None,
                 taboo_size_params=None, neighbor_size=1000, candidate_num=200):
        self.taboo_size = 10 if taboo_size_params is None else taboo_size_params
        self.iteration = 200
        self.num_city = num_city
        self.location = data
        self.taboo = []

        self.dynamic_taboo = True
        self.min_taboo = 8
        self.max_taboo = 20

        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.path = self.hybrid_init(dis_mat=self.dis_mat, num_total=candidate_num, num_city=num_city)
        self.best_path = self.path
        self.cur_path = self.path
        self.best_length = self.compute_pathlen(self.path, self.dis_mat)
        self.initial_length = self.best_length  # 记录初始解距离

        self.iter_x = [0]
        self.iter_y = [self.best_length]

        # 重启机制参数
        self.restart_threshold = 200
        self.no_improve_count = 0
        self.last_best_length = self.best_length

        # 新增参数
        self.optimal_length = optimal_length  # 理论最优解
        self.neighbor_size = neighbor_size  # 邻域大小
        self.candidate_num = candidate_num  # 候选解数量
        self.iter_time = []  # 每轮时间
        self.total_time = 0  # 总时间
        self.start_time = time.time()  # 记录开始时间

        # 记录搜索过程中的路径
        self.search_paths = [self.path.copy()]
        self.search_lengths = [self.best_length]

        # 4.3.3 混合初始化：贪心+随机

    def hybrid_init(self, dis_mat, num_total, num_city):
        # 贪心初始化生成num_total个路径，选择前50%最优的
        greedy_paths = self.greedy_init(dis_mat, num_total, num_city)
        pathlens = self.compute_paths(greedy_paths)
        sortindex = np.argsort(pathlens)
        selected_greedy = [greedy_paths[i] for i in sortindex[:num_total//2]]

        # 随机初始化生成50%的路径
        random_paths = [self.random_init(num_city) for _ in range(num_total//2)]
        all_paths = selected_greedy + random_paths

        # 选择所有路径中最优的
        pathlens = self.compute_paths(all_paths)
        sortindex = np.argsort(pathlens)
        index = sortindex[0]
        return all_paths[index]
    def greedy_init(self, dis_mat, num_total, num_city):
        start_index = 0
        result = []
        for i in range(num_total):
            rest = [x for x in range(num_city)]
            if start_index >= num_city:
                start_index = np.random.randint(0, num_city)
                # 复制已有路径避免越界
                result.append(result[start_index % len(result)].copy())
                continue
            current = start_index
            rest.remove(current)
            result_one = [current]
            while rest:
                tmp_min = math.inf
                tmp_choose = -1
                for x in rest:
                    if dis_mat[current][x] < tmp_min:
                        tmp_min = dis_mat[current][x]
                        tmp_choose = x
                current = tmp_choose
                result_one.append(tmp_choose)
                rest.remove(tmp_choose)
            result.append(result_one)
            start_index += 1
        return result  # 返回所有生成的路径


    # 初始化一条随机路径
    def random_init(self, num_city):
        tmp = [x for x in range(num_city)]
        random.shuffle(tmp)
        return tmp

    # 计算不同城市之间的距离
    def compute_dis_mat(self, num_city, location):
        dis_mat = np.zeros((num_city, num_city))
        for i in range(num_city):
            for j in range(num_city):
                if i == j:
                    dis_mat[i][j] = np.inf
                    continue
                a = location[i]
                b = location[j]
                tmp = np.sqrt(sum([(x[0] - x[1]) ** 2 for x in zip(a, b)]))
                dis_mat[i][j] = tmp
        return dis_mat

    # 计算路径长度
    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    # 计算一个群体的长度
    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    # 4.2.1 多种扰动操作（Swap/2-opt/3-opt）
    def generate_neighbors(self, x, operator="2-opt"):
        """候选解生成策略
        数学推导：
        - Swap操作：选择两个节点交换，邻域大小为C(n,2) = O(n^2)
        - 2-opt操作：反转子路径，邻域大小为O(n^2)
        - 3-opt操作：分三段重组路径，邻域大小为O(n^3)
        复杂度与邻域大小正相关，扰动强度：3-opt > 2-opt > Swap
        """
        new_paths = []
        moves = []
        if operator == "swap":
            # 增加Swap操作的邻域大小
            for _ in range(self.neighbor_size):  # 使用参):
                i, j = random.sample(range(len(x)), 2)
                new_path = x.copy()
                new_path[i], new_path[j] = new_path[j], new_path[i]
                new_paths.append(new_path)
                moves.append((i, j))
        elif operator == "2-opt":
            # 增加2-opt操作的邻域大小
            while len(new_paths) < self.neighbor_size:  # 使用参数
                i = np.random.randint(len(x) - 1)
                j = np.random.randint(i + 1, len(x))
                new_path = x[:i] + x[i:j][::-1] + x[j:]
                new_paths.append(new_path)
                moves.append((i, j))
        elif operator == "3-opt":
            # 增加3-opt操作的邻域大小
            while len(new_paths) < self.neighbor_size:
                i = np.random.randint(len(x) - 2)
                j = np.random.randint(i + 1, len(x) - 1)
                k = np.random.randint(j + 1, len(x))
                # 增加重组方式
                mode = np.random.randint(4)  # 增加到4种重组方式
                if mode == 0:
                    new_path = x[:i] + x[i:j][::-1] + x[j:k] + x[k:][::-1]
                elif mode == 1:
                    new_path = x[:i] + x[j:k] + x[i:j] + x[k:]
                elif mode == 2:
                    new_path = x[:i] + x[k:] + x[i:j] + x[j:k]
                else:
                    new_path = x[:i] + x[j:k][::-1] + x[i:j][::-1] + x[k:]
                new_paths.append(new_path)
                moves.append((i, j, k))
        return new_paths, moves

    # 4.3.1 动态禁忌长度策略
    def update_taboo_size(self, cnt):
        """根据迭代次数动态调整禁忌长度
        策略：前期探索时禁忌表较小，后期逐渐增大以加强局部搜索
        """
        if self.dynamic_taboo:
            self.taboo_size = int(self.min_taboo + (self.max_taboo - self.min_taboo) * (cnt / self.iteration))
        else:
            self.taboo_size = 5

    # 禁忌搜索
    def ts(self):
        start_time = time.time()
        for cnt in range(self.iteration):
            iter_start = time.time()
            self.update_taboo_size(cnt)
            
            # 动态调整搜索策略
            if cnt < self.iteration * 0.3:  # 前期更多使用3-opt
                if np.random.rand() < 0.3:
                    new_paths, moves = self.generate_neighbors(self.cur_path, operator="3-opt")
                else:
                    new_paths, moves = self.generate_neighbors(self.cur_path, operator="2-opt")
            else:  # 后期更多使用2-opt
                if np.random.rand() < 0.1:
                    new_paths, moves = self.generate_neighbors(self.cur_path, operator="3-opt")
                else:
                    new_paths, moves = self.generate_neighbors(self.cur_path, operator="2-opt")

            new_lengths = self.compute_paths(new_paths)
            sort_index = np.argsort(new_lengths)
            
            # 检查是否找到更好的解
            found_better = False
            for idx in sort_index:
                if new_lengths[idx] < self.best_length:
                    self.best_length = new_lengths[idx]
                    self.best_path = new_paths[idx]
                    found_better = True
                    break
            
            # 更新未改善计数
            if found_better:
                self.no_improve_count = 0
                self.last_best_length = self.best_length
            else:
                self.no_improve_count += 1
            
            # 重启机制
            if self.no_improve_count >= self.restart_threshold:
                self.cur_path = self.hybrid_init(self.dis_mat, 200, self.num_city)
                self.no_improve_count = 0
                self.taboo = []  # 清空禁忌表
                continue
            
            # 更新当前路径
            for idx in sort_index:
                if new_paths[idx] not in self.taboo:
                    self.cur_path = new_paths[idx]
                    self.taboo.append(new_paths[idx])
                    break
            
            # 管理禁忌表
            if len(self.taboo) > self.taboo_size:
                self.taboo = self.taboo[1:]
            
            # 记录搜索过程
            self.search_paths.append(self.cur_path.copy())
            self.search_lengths.append(self.best_length)
            
            self.iter_x.append(cnt)
            self.iter_y.append(self.best_length)
            self.iter_time.append(time.time() - iter_start)
            
            print(cnt, self.best_length)
            
        self.total_time = time.time() - start_time
        
        # 输出评估指标
        print(f"初始解距离: {self.initial_length:.2f}")
        print(f"最终解距离: {self.best_length:.2f}")
        if self.optimal_length:
            error = (self.best_length - self.optimal_length) / self.optimal_length * 100
            print(f"相对误差: {error:.2f}%")
        print(f"总时间: {self.total_time:.2f}秒")
        print(f"平均每轮时间: {np.mean(self.iter_time):.4f}秒")

    def run(self):
        self.ts()
        return self.location[self.best_path], self.best_length

    def get_performance_metrics(self):
        """返回性能指标"""
        return {
            'best_length': self.best_length,
            'execution_time': self.total_time,
            'improvements': len(self.iter_y),
            'best_iteration': np.argmin(self.iter_y),
            'initial_length': self.initial_length,
            'improvement_ratio': (self.initial_length - self.best_length) / self.initial_length,
            'convergence_data': {
                'iterations': self.iter_x,
                'lengths': self.iter_y,
                'times': self.iter_time
            },
            'search_paths': self.search_paths,
            'search_lengths': self.search_lengths
        }

# 读取数据
def read_tsp(path):
    lines = open(path, 'r').readlines()
    assert 'NODE_COORD_SECTION\n' in lines
    index = lines.index('NODE_COORD_SECTION\n')
    data = lines[index + 1:-1]
    tmp = []
    for line in data:
        line = line.strip().split(' ')
        if line[0] == 'EOF':
            continue
        tmpline = []
        for x in line:
            if x == '':
                continue
            else:
                tmpline.append(float(x))
        if tmpline == []:
            continue
        tmp.append(tmpline)
    data = tmp
    return data


data = read_tsp('data/berlin52.tsp')
data = np.array(data)
data = data[:, 1:]


# ========== 参数敏感性分析 ==========
def parameter_sensitivity_analysis():
    # 参数设置
    taboo_sizes = [5, 10,15]
    neighbor_sizes = [100, 500, 1000]
    candidate_nums = [100, 200, 400]
    results = []

    # 运行不同参数组合的实验
    for ts_size in taboo_sizes:
        for ns in neighbor_sizes:
            for cn in candidate_nums:
                print(f"Running: taboo_size={ts_size}, neighbor_size={ns}, candidate_num={cn}")
                model = TS(
                    num_city=data.shape[0],
                    data=data.copy(),
                    optimal_length=7542,  # 柏林52的最优解
                    taboo_size_params=ts_size,
                    neighbor_size=ns,
                    candidate_num=cn
                )
                model.run()
                results.append({
                    '禁忌长度': ts_size,
                    '邻域大小': ns,
                    '候选解数量': cn,
                    '最终长度': model.best_length,
                    '时间': model.total_time,
                    '收敛曲线': model.iter_y
                })

    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. 禁忌长度影响
    for ts_size in taboo_sizes:
        # 对每个禁忌长度，计算平均收敛曲线
        curves = [res['收敛曲线'] for res in results if res['禁忌长度'] == ts_size]
        mean_curve = np.mean([np.array(curve) for curve in curves], axis=0)
        std_curve = np.std([np.array(curve) for curve in curves], axis=0)
        iterations = range(len(mean_curve))
        ax1.plot(iterations, mean_curve, label=f'禁忌长度={ts_size}')
        ax1.fill_between(iterations, 
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('路径长度')
    ax1.set_title('禁忌长度对收敛速度的影响')
    ax1.legend()
    ax1.grid(True)

    # 2. 邻域大小影响
    for ns in neighbor_sizes:
        # 对每个邻域大小，计算平均收敛曲线
        curves = [res['收敛曲线'] for res in results if res['邻域大小'] == ns]
        mean_curve = np.mean([np.array(curve) for curve in curves], axis=0)
        std_curve = np.std([np.array(curve) for curve in curves], axis=0)
        iterations = range(len(mean_curve))
        ax2.plot(iterations, mean_curve, label=f'邻域大小={ns}')
        ax2.fill_between(iterations, 
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('路径长度')
    ax2.set_title('邻域大小对收敛速度的影响')
    ax2.legend()
    ax2.grid(True)

    # 3. 候选解数量影响
    for cn in candidate_nums:
        # 对每个候选解数量，计算平均收敛曲线
        curves = [res['收敛曲线'] for res in results if res['候选解数量'] == cn]
        mean_curve = np.mean([np.array(curve) for curve in curves], axis=0)
        std_curve = np.std([np.array(curve) for curve in curves], axis=0)
        iterations = range(len(mean_curve))
        ax3.plot(iterations, mean_curve, label=f'候选解数量={cn}')
        ax3.fill_between(iterations, 
                        mean_curve - std_curve,
                        mean_curve + std_curve,
                        alpha=0.2)
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('路径长度')
    ax3.set_title('候选解数量对收敛速度的影响')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png')
    plt.show()

    # 打印参数影响分析结果
    print("\n参数敏感性分析结果:")
    print("=" * 80)
    
    # 分析禁忌长度影响
    print("\n禁忌长度影响:")
    for ts_size in taboo_sizes:
        lengths = [res['最终长度'] for res in results if res['禁忌长度'] == ts_size]
        times = [res['时间'] for res in results if res['禁忌长度'] == ts_size]
        print(f"禁忌长度={ts_size}:")
        print(f"  平均路径长度: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
        print(f"  平均执行时间: {np.mean(times):.2f} ± {np.std(times):.2f}秒")
    
    # 分析邻域大小影响
    print("\n邻域大小影响:")
    for ns in neighbor_sizes:
        lengths = [res['最终长度'] for res in results if res['邻域大小'] == ns]
        times = [res['时间'] for res in results if res['邻域大小'] == ns]
        print(f"邻域大小={ns}:")
        print(f"  平均路径长度: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
        print(f"  平均执行时间: {np.mean(times):.2f} ± {np.std(times):.2f}秒")
    
    # 分析候选解数量影响
    print("\n候选解数量影响:")
    for cn in candidate_nums:
        lengths = [res['最终长度'] for res in results if res['候选解数量'] == cn]
        times = [res['时间'] for res in results if res['候选解数量'] == cn]
        print(f"候选解数量={cn}:")
        print(f"  平均路径长度: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
        print(f"  平均执行时间: {np.mean(times):.2f} ± {np.std(times):.2f}秒")

if __name__ == "__main__":
    # 主程序
    model = TS(num_city=data.shape[0], data=data.copy(), optimal_length=7542)
    Best_path, Best_length = model.run()

    # 绘制路径和收敛曲线
    plt.suptitle('TS in berlin52.tsp')
    plt.subplot(2, 2, 1)
    plt.title('raw data')
    show_data = np.vstack([data, data[0]])
    plt.plot(data[:, 0], data[:, 1])

    Best_path = np.vstack([Best_path, Best_path[0]])
    fig, axs = plt.subplots(2, 1, sharex=False, sharey=False)
    axs[0].scatter(Best_path[:, 0], Best_path[:,1])
    Best_path = np.vstack([Best_path, Best_path[0]])
    axs[0].plot(Best_path[:, 0], Best_path[:, 1])
    axs[0].set_title('规划结果')
    iterations = model.iter_x
    best_record = model.iter_y
    axs[1].plot(iterations, best_record)
    axs[1].set_title('收敛曲线')
    plt.show()

    # 运行参数敏感性分析
    parameter_sensitivity_analysis()

