import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt

# === 新增字体配置 ===
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# =====================

# === 读取数据 ===
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

# === 禁忌搜索算法 ===
class TS:
    def __init__(self, num_city, data, optimal_length=None,
                 taboo_size_params=None, neighbor_size=1000, candidate_num=200,
                 dynamic_taboo=False, use_hybrid_neighbor=False, use_heuristic_init=False, enable_restart=False):
        self.taboo_size = 10 if taboo_size_params is None else taboo_size_params
        self.iteration = 200
        self.num_city = num_city
        self.location = data
        self.taboo = []

        self.dynamic_taboo = dynamic_taboo
        self.min_taboo = 8
        self.max_taboo = 20

        self.dis_mat = self.compute_dis_mat(num_city, data)
        self.path = self.hybrid_init(dis_mat=self.dis_mat, num_total=candidate_num, num_city=num_city) if use_heuristic_init else self.random_init(num_city)
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

        # 消融实验参数
        self.use_hybrid_neighbor = use_hybrid_neighbor
        self.enable_restart = enable_restart

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
        return result

    def random_init(self, num_city):
        tmp = [x for x in range(num_city)]
        random.shuffle(tmp)
        return tmp

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

    def compute_pathlen(self, path, dis_mat):
        a = path[0]
        b = path[-1]
        result = dis_mat[a][b]
        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            result += dis_mat[a][b]
        return result

    def compute_paths(self, paths):
        result = []
        for one in paths:
            length = self.compute_pathlen(one, self.dis_mat)
            result.append(length)
        return result

    def generate_neighbors(self, x, operator="2-opt"):
        new_paths = []
        moves = []
        if operator == "swap":
            for _ in range(self.neighbor_size):
                i, j = random.sample(range(len(x)), 2)
                new_path = x.copy()
                new_path[i], new_path[j] = new_path[j], new_path[i]
                new_paths.append(new_path)
                moves.append((i, j))
        elif operator == "2-opt":
            while len(new_paths) < self.neighbor_size:
                i = np.random.randint(len(x) - 1)
                j = np.random.randint(i + 1, len(x))
                new_path = x[:i] + x[i:j][::-1] + x[j:]
                new_paths.append(new_path)
                moves.append((i, j))
        elif operator == "3-opt":
            while len(new_paths) < self.neighbor_size:
                i = np.random.randint(len(x) - 2)
                j = np.random.randint(i + 1, len(x) - 1)
                k = np.random.randint(j + 1, len(x))
                mode = np.random.randint(4)
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

    def update_taboo_size(self, cnt):
        if self.dynamic_taboo:
            self.taboo_size = int(self.min_taboo + (self.max_taboo - self.min_taboo) * (cnt / self.iteration))
        else:
            self.taboo_size = 5

    def ts(self):
        start_time = time.time()
        for cnt in range(self.iteration):
            iter_start = time.time()
            self.update_taboo_size(cnt)

            if self.use_hybrid_neighbor:
                if cnt < self.iteration * 0.3:
                    if np.random.rand() < 0.3:
                        new_paths, moves = self.generate_neighbors(self.cur_path, operator="3-opt")
                    else:
                        new_paths, moves = self.generate_neighbors(self.cur_path, operator="2-opt")
                else:
                    if np.random.rand() < 0.1:
                        new_paths, moves = self.generate_neighbors(self.cur_path, operator="3-opt")
                    else:
                        new_paths, moves = self.generate_neighbors(self.cur_path, operator="2-opt")
            else:
                new_paths, moves = self.generate_neighbors(self.cur_path, operator="2-opt")

            new_lengths = self.compute_paths(new_paths)
            sort_index = np.argsort(new_lengths)

            found_better = False
            for idx in sort_index:
                if new_lengths[idx] < self.best_length:
                    self.best_length = new_lengths[idx]
                    self.best_path = new_paths[idx]
                    found_better = True
                    break

            if found_better:
                self.no_improve_count = 0
                self.last_best_length = self.best_length
            else:
                self.no_improve_count += 1

            if self.enable_restart and self.no_improve_count >= self.restart_threshold:
                self.cur_path = self.hybrid_init(self.dis_mat, 200, self.num_city)
                self.no_improve_count = 0
                self.taboo = []
                continue

            for idx in sort_index:
                if new_paths[idx] not in self.taboo:
                    self.cur_path = new_paths[idx]
                    self.taboo.append(new_paths[idx])
                    break

            if len(self.taboo) > self.taboo_size:
                self.taboo = self.taboo[1:]

            self.search_paths.append(self.cur_path.copy())
            self.search_lengths.append(self.best_length)

            self.iter_x.append(cnt)
            self.iter_y.append(self.best_length)
            self.iter_time.append(time.time() - iter_start)

        self.total_time = time.time() - start_time

    def run(self):
        self.ts()
        return self.location[self.best_path], self.best_length

    def get_performance_metrics(self):
        return {
            'best_length': self.best_length,
            'execution_time': self.total_time,
            'initial_length': self.initial_length,
            'convergence_data': {
                'iterations': self.iter_x,
                'lengths': self.iter_y
            }
        }

# === 消融实验 ===
def ablation_study():
    experiments = [
        {
            "name": "Baseline",
            "params": {
                "dynamic_taboo": False,
                "use_hybrid_neighbor": False,
                "use_heuristic_init": False,
                "enable_restart": False
            }
        },
        {
            "name": "+动态禁忌",
            "params": {
                "dynamic_taboo": True,
                "use_hybrid_neighbor": False,
                "use_heuristic_init": False,
                "enable_restart": False
            }
        },
        {
            "name": "+混合邻域",
            "params": {
                "dynamic_taboo": False,
                "use_hybrid_neighbor": True,
                "use_heuristic_init": False,
                "enable_restart": False
            }
        },
        {
            "name": "+启发式初始化",
            "params": {
                "dynamic_taboo": False,
                "use_hybrid_neighbor": False,
                "use_heuristic_init": True,
                "enable_restart": False
            }
        },
        {
            "name": "+重启机制",
            "params": {
                "dynamic_taboo": False,
                "use_hybrid_neighbor": False,
                "use_heuristic_init": False,
                "enable_restart": True
            }
        },
        {
            "name": "完整模型",
            "params": {
                "dynamic_taboo": True,
                "use_hybrid_neighbor": True,
                "use_heuristic_init": True,
                "enable_restart": True
            }
        }
    ]

    results = []

    for exp in experiments:
        print(f"\n=== 正在运行实验组：{exp['name']} ===")
        lengths = []
        initial_lengths = []
        times = []
        curves = []

        for _ in range(10):
            model = TS(
                num_city=data.shape[0],
                data=data.copy(),
                optimal_length=7542,
                neighbor_size=1000,
                candidate_num=200,
                **exp["params"]
            )
            model.run()

            metrics = model.get_performance_metrics()
            lengths.append(metrics['best_length'])
            initial_lengths.append(metrics['initial_length'])
            times.append(metrics['execution_time'])
            curves.append(metrics['convergence_data']['lengths'])

        results.append({
            "name": exp['name'],
            "initial_length": np.mean(initial_lengths),
            "final_length": np.mean(lengths),
            "time": np.mean(times),
            "std": np.std(lengths),
            "curves": curves
        })

    print("\n=== 消融实验结果 ===")
    print("{:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "实验组", "初始解距离", "最终解距离", "相对误差(%)", "标准差"))
    for res in results:
        error = (res['final_length'] - 7542) / 7542 * 100
        print("{:<15} {:<15.1f} {:<15.1f} {:<15.2f} {:<15.1f}".format(
            res['name'], res['initial_length'], res['final_length'], error, res['std']))

    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(res['curves'][0], label=res['name'], alpha=0.8)
    plt.xlabel('迭代次数')
    plt.ylabel('路径长度')
    plt.title('各实验组收敛曲线对比')
    plt.legend()
    plt.grid(True)
    plt.savefig('ablation_convergence.png')
    plt.show()

# === 主程序 ===
if __name__ == "__main__":
    data = read_tsp('data/berlin52.tsp')
    data = np.array(data)
    data = data[:, 1:]

    ablation_study()