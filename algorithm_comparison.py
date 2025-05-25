import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from TS import TS
from GA import GA
from SA import SA
from ACO import ACO
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d

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

def normalize_convergence_curve(x, y, target_length=200):
    """使用插值方法将收敛曲线标准化到指定长度"""
    # 创建插值函数
    f = interp1d(np.linspace(0, 1, len(x)), y, kind='linear')
    # 生成新的x轴点
    new_x = np.linspace(0, 1, target_length)
    # 计算对应的y值
    new_y = f(new_x)
    return new_x, new_y

def run_benchmark(data_path, num_runs=10):
    # 读取数据
    data = read_tsp(data_path)
    data = np.array(data)
    data = data[:, 1:]
    num_city = data.shape[0]
    
    # 存储结果
    results = {
        'TS': {'best_lengths': [], 'times': [], 'paths': [], 'convergence': []},
        'GA': {'best_lengths': [], 'times': [], 'paths': [], 'convergence': []},
        'SA': {'best_lengths': [], 'times': [], 'paths': [], 'convergence': []},
        'ACO': {'best_lengths': [], 'times': [], 'paths': [], 'convergence': []}
    }
    
    # 运行多次实验
    for i in tqdm(range(num_runs), desc="Running benchmark"):
        # TS
        start_time = time.time()
        ts_model = TS(num_city=num_city, data=data.copy(), optimal_length=7542)
        ts_path, ts_length = ts_model.run()
        ts_time = time.time() - start_time
        results['TS']['best_lengths'].append(ts_length)
        results['TS']['times'].append(ts_time)
        results['TS']['paths'].append(ts_path)
        results['TS']['convergence'].append(ts_model.iter_y)
        
        # GA
        start_time = time.time()
        ga_model = GA(num_city=num_city, num_total=25, iteration=1000, data=data.copy())
        ga_path, ga_length = ga_model.run()
        ga_time = time.time() - start_time
        results['GA']['best_lengths'].append(ga_length)
        results['GA']['times'].append(ga_time)
        results['GA']['paths'].append(ga_path)
        results['GA']['convergence'].append(ga_model.best_record)
        
        # SA
        start_time = time.time()
        sa_model = SA(num_city=num_city, data=data.copy())
        sa_path, sa_length = sa_model.run()
        sa_time = time.time() - start_time
        results['SA']['best_lengths'].append(sa_length)
        results['SA']['times'].append(sa_time)
        results['SA']['paths'].append(sa_path)
        results['SA']['convergence'].append(sa_model.iter_y)
        
        # ACO
        start_time = time.time()
        aco_model = ACO(num_city=num_city, data=data.copy())
        aco_path, aco_length = aco_model.run()
        aco_time = time.time() - start_time
        results['ACO']['best_lengths'].append(aco_length)
        results['ACO']['times'].append(aco_time)
        results['ACO']['paths'].append(aco_path)
        results['ACO']['convergence'].append(aco_model.iter_y)
    
    return results

def analyze_results(results):
    analysis = {}
    for algo in results:
        lengths = np.array(results[algo]['best_lengths'])
        times = np.array(results[algo]['times'])
        
        analysis[algo] = {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'best_length': np.min(lengths),
            'worst_length': np.max(lengths)
        }
    
    return analysis

def plot_convergence_curves(data_path):
    # 读取数据
    data = read_tsp(data_path)
    data = np.array(data)
    data = data[:, 1:]
    num_city = data.shape[0]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 运行TS算法
    print("Running TS...")
    ts_model = TS(num_city=num_city, data=data.copy(), optimal_length=7542)
    ts_path, ts_length = ts_model.run()
    ts_x, ts_y = normalize_convergence_curve(ts_model.iter_x, ts_model.iter_y)
    plt.plot(ts_x, ts_y, 'b-', label='TS', linewidth=2)
    
    # 运行GA算法
    print("Running GA...")
    ga_model = GA(num_city=num_city, num_total=25, iteration=200, data=data.copy())
    ga_path, ga_length = ga_model.run()
    ga_x, ga_y = normalize_convergence_curve(range(len(ga_model.best_record)), ga_model.best_record)
    plt.plot(ga_x, ga_y, 'r-', label='GA', linewidth=2)
    
    # 运行SA算法
    print("Running SA...")
    sa_model = SA(num_city=num_city, data=data.copy())
    sa_path, sa_length = sa_model.run()
    sa_x, sa_y = normalize_convergence_curve(sa_model.iter_x, sa_model.iter_y)
    plt.plot(sa_x, sa_y, 'g-', label='SA', linewidth=2)
    
    # 运行ACO算法
    print("Running ACO...")
    start_time = time.time()
    aco_model = ACO(num_city=num_city, data=data.copy())
    aco_path, aco_length = aco_model.run()
    aco_time = time.time() - start_time
    aco_x, aco_y = normalize_convergence_curve(aco_model.iter_x, aco_model.iter_y)
    plt.plot(aco_x, aco_y, 'm-', label='ACO', linewidth=2)
    
    # 设置图形属性
    plt.xlabel('归一化进度迭代 (0-1)', fontsize=12)
    plt.ylabel('路径长度', fontsize=12)
    plt.title('收敛曲线归一化对比', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(7000, 9000)  # 新增：截断y轴显示范围
    
    # 添加最优解参考线
    if 'berlin52' in data_path:
        plt.axhline(y=7542, color='k', linestyle='--', alpha=0.5, label='Optimal Solution')
    
    # 保存图形
    plt.savefig('normalized_convergence_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印性能指标
    print("\nPerformance Metrics:")
    print("=" * 80)
    print("\nTS Algorithm:")
    print(f"Final Path Length: {ts_length:.2f}")
    print(f"Execution Time: {ts_model.total_time:.2f} seconds")
    print(f"Total Iterations: {len(ts_model.iter_x)}")
    print(f"Improvement Ratio: {(ts_model.initial_length - ts_length) / ts_model.initial_length * 100:.2f}%")
    
    print("\nGA Algorithm:")
    print(f"Final Path Length: {ga_length:.2f}")
    print(f"Execution Time: {ga_model.total_time:.2f} seconds")
    print(f"Total Iterations: {len(ga_model.best_record)}")
    print(f"Improvement Ratio: {(ga_model.initial_length - ga_length) / ga_model.initial_length * 100:.2f}%")
    
    print("\nSA Algorithm:")
    print(f"Final Path Length: {sa_length:.2f}")
    print(f"Execution Time: {sa_model.total_time:.2f} seconds")
    print(f"Total Iterations: {len(sa_model.iter_x)}")
    print(f"Improvement Ratio: {(sa_model.initial_length - sa_length) / sa_model.initial_length * 100:.2f}%")
    
    print("\nACO Algorithm:")
    print(f"Final Path Length: {aco_length:.2f}")
    print(f"Execution Time: {aco_time:.2f} seconds")
    print(f"Total Iterations: {len(aco_model.iter_x)}")
    # initial_length 兼容性处理
    if hasattr(aco_model, 'initial_length'):
        initial_length = aco_model.initial_length
    else:
        initial_length = aco_model.iter_y[0] if hasattr(aco_model, 'iter_y') and len(aco_model.iter_y) > 0 else 0
    if initial_length:
        print(f"Improvement Ratio: {(initial_length - aco_length) / initial_length * 100:.2f}%")
    else:
        print("Improvement Ratio: N/A")

    # 新增：ACO路径可视化，风格与示例一致
    aco_best_path = np.vstack([aco_path, aco_path[0]])
    plt.figure(figsize=(10, 6))
    plt.scatter(aco_best_path[:, 0], aco_best_path[:, 1], c='blue', label='Cities')
    plt.plot(aco_best_path[:, 0], aco_best_path[:, 1], 'r-', label='Path')
    plt.title('ACO')
    plt.legend()
    plt.grid(True)
    plt.savefig('ACO_best_path.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_path_animation(results, data, algo_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def update(frame):
        ax.clear()
        path = results[algo_name]['paths'][frame]
        path = np.vstack([path, path[0]])  # 闭合路径
        
        ax.scatter(data[:, 0], data[:, 1], c='blue', label='Cities')
        ax.plot(path[:, 0], path[:, 1], 'r-', label='Path')
        ax.set_title(f'{algo_name}')
        ax.legend()
        ax.grid(True)
    
    anim = FuncAnimation(fig, update, frames=len(results[algo_name]['paths']),
                        interval=1000, repeat=False)
    anim.save(f'{algo_name}_path_animation.gif', writer='pillow')
    plt.close()

def main():
    # 运行收敛曲线对比
    plot_convergence_curves('data/berlin52.tsp')
    
    # 运行基准测试
    results = run_benchmark('data/berlin52.tsp', num_runs=10)
    
    # 分析结果
    analysis = analyze_results(results)
    
    # 打印分析结果
    print("\nAlgorithm Comparison Results:")
    print("=" * 80)
    for algo in analysis:
        print(f"\n{algo}:")
        print(f"Mean Path Length: {analysis[algo]['mean_length']:.2f} ± {analysis[algo]['std_length']:.2f}")
        print(f"Mean Execution Time: {analysis[algo]['mean_time']:.2f} ± {analysis[algo]['std_time']:.2f} seconds")
        print(f"Best Path Length: {analysis[algo]['best_length']:.2f}")
        print(f"Worst Path Length: {analysis[algo]['worst_length']:.2f}")
    
    # 创建路径动画
    data = read_tsp('data/berlin52.tsp')
    data = np.array(data)
    data = data[:, 1:]
    for algo in results:
        create_path_animation(results, data, algo)

if __name__ == "__main__":
    main() 