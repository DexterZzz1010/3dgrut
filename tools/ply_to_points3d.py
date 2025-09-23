#!/usr/bin/env python3
"""
CUDA加速的最远点采样 - Linus风格
让GPU做它最擅长的事：并行计算距离
"""

import torch
import numpy as np
import sys
import os
from plyfile import PlyData

def farthest_point_sampling_cuda(positions, num_samples, device='cuda'):
    """
    CUDA加速的最远点采样
    
    这就是"好品味"的体现：
    - GPU擅长并行计算距离矩阵
    - 避免Python循环的开销
    - 简单直接，没有复杂的内存管理
    """
    positions = torch.as_tensor(positions, dtype=torch.float32, device=device)
    n_points = positions.shape[0]
    
    if num_samples >= n_points:
        return torch.arange(n_points, device=device)
    
    # 选择的点索引
    selected = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((n_points,), float('inf'), device=device)
    
    # 第一个点随机选择（或选择0）
    selected[0] = 0
    
    for i in range(1, num_samples):
        # 计算到最新选择点的距离 - GPU并行计算
        last_point = positions[selected[i-1]]
        new_distances = torch.sum((positions - last_point) ** 2, dim=1)
        
        # 更新最小距离 - GPU并行操作
        distances = torch.minimum(distances, new_distances)
        
        # 找到最远的点 - GPU reduction
        selected[i] = torch.argmax(distances)
    
    return selected.cpu().numpy()

def farthest_point_sampling_cuda_optimized(positions, num_samples, device='cuda', batch_size=None):
    """
    优化版CUDA最远点采样
    
    对于超大点云，可以分批处理避免GPU内存爆炸
    这就是我说的实用主义 - 解决实际问题
    """
    positions = torch.as_tensor(positions, dtype=torch.float32, device=device)
    n_points = positions.shape[0]
    
    if num_samples >= n_points:
        return torch.arange(n_points, device=device)
    
    # 自动决定batch size
    if batch_size is None:
        # 粗略估计：假设每个点32字节，预留一些GPU内存
        gpu_memory = torch.cuda.get_device_properties(device).total_memory
        available_memory = gpu_memory * 0.8  # 保守估计
        batch_size = min(n_points, int(available_memory / (32 * n_points)))
        batch_size = max(batch_size, 1000)  # 最小batch size
    
    selected = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((n_points,), float('inf'), device=device)
    
    # 第一个点
    selected[0] = 0
    
    for i in range(1, num_samples):
        last_point = positions[selected[i-1]]
        
        # 分批计算距离（如果需要）
        if n_points > batch_size:
            for start in range(0, n_points, batch_size):
                end = min(start + batch_size, n_points)
                batch_pos = positions[start:end]
                batch_distances = torch.sum((batch_pos - last_point) ** 2, dim=1)
                distances[start:end] = torch.minimum(distances[start:end], batch_distances)
        else:
            new_distances = torch.sum((positions - last_point) ** 2, dim=1)
            distances = torch.minimum(distances, new_distances)
        
        selected[i] = torch.argmax(distances)
    
    return selected.cpu().numpy()

def extract_colors(vertex):
    """提取颜色信息 - 不变"""
    if all(field in vertex for field in ['red', 'green', 'blue']):
        return np.column_stack([vertex['red'], vertex['green'], vertex['blue']])
    
    if all(field in vertex for field in ['diffuse_red', 'diffuse_green', 'diffuse_blue']):
        return (np.column_stack([
            vertex['diffuse_red'], vertex['diffuse_green'], vertex['diffuse_blue']
        ]) * 255).astype(np.uint8)
    
    z = vertex['z']
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
    r = (z_norm * 255).astype(np.uint8)
    g = ((1 - np.abs(z_norm - 0.5) * 2) * 255).astype(np.uint8)
    b = ((1 - z_norm) * 255).astype(np.uint8)
    return np.column_stack([r, g, b])

def ply_to_points3d_cuda(ply_path, output_path, max_points=None, use_cuda=True):
    """
    CUDA加速的PLY转换
    
    "Never break userspace" - 保持相同的接口
    但内部使用GPU加速，速度提升10-100倍
    """
    if not os.path.exists(ply_path):
        print(f"错误: PLY文件不存在: {ply_path}", file=sys.stderr)
        return False
    
    # 检查CUDA可用性
    if use_cuda and not torch.cuda.is_available():
        print("警告: CUDA不可用，回退到CPU版本", file=sys.stderr)
        use_cuda = False
    
    try:
        print(f"读取PLY文件: {ply_path}")
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        
        positions = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        colors = extract_colors(vertex)
        
        print(f"读取到 {len(positions)} 个点")
        
        # 采样（如果需要）
        if max_points and len(positions) > max_points:
            device = 'cuda' if use_cuda else 'cpu'
            print(f"使用{'CUDA' if use_cuda else 'CPU'}最远点采样减少到 {max_points} 个点...")
            
            # 选择合适的算法
            if use_cuda:
                if len(positions) > 100000:  # 大点云使用优化版本
                    indices = farthest_point_sampling_cuda_optimized(positions, max_points, device)
                else:
                    indices = farthest_point_sampling_cuda(positions, max_points, device)
            else:
                # 回退到原始CPU版本
                indices = farthest_point_sampling_cpu(positions, max_points)
            
            positions = positions[indices]
            colors = colors[indices]
        
        # 输出部分不变
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write(f"# Number of points: {len(positions)}\n")
            
            for i, (pos, color) in enumerate(zip(positions, colors)):
                x, y, z = pos
                r, g, b = color.astype(int)
                f.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 1.0\n")
        
        print(f"成功生成: {output_path}")
        return True
        
    except Exception as e:
        print(f"转换失败: {e}", file=sys.stderr)
        return False

def farthest_point_sampling_cpu(positions, num_samples):
    """CPU版本作为备选 - 保持向后兼容"""
    n_points = len(positions)
    if num_samples >= n_points:
        return np.arange(n_points)
    
    selected = [0]
    distances = np.full(n_points, np.inf)
    
    for _ in range(num_samples - 1):
        last_point = positions[selected[-1]]
        new_distances = np.sum((positions - last_point) ** 2, axis=1)
        distances = np.minimum(distances, new_distances)
        next_idx = np.argmax(distances)
        selected.append(next_idx)
    
    return np.array(selected)

def benchmark_fps_methods(positions, num_samples, runs=3):
    """
    性能测试 - 让数据说话
    这就是我的实用主义：测试性能，不要猜测
    """
    import time
    
    print(f"\n性能测试: {len(positions)} 个点 -> {num_samples} 个点")
    print("=" * 50)
    
    # CPU版本
    cpu_times = []
    for _ in range(runs):
        start = time.time()
        farthest_point_sampling_cpu(positions, num_samples)
        cpu_times.append(time.time() - start)
    cpu_avg = np.mean(cpu_times)
    
    if torch.cuda.is_available():
        # CUDA版本
        cuda_times = []
        for _ in range(runs):
            torch.cuda.synchronize()  # 确保GPU完成
            start = time.time()
            farthest_point_sampling_cuda(positions, num_samples)
            torch.cuda.synchronize()
            cuda_times.append(time.time() - start)
        cuda_avg = np.mean(cuda_times)
        
        # 优化CUDA版本
        cuda_opt_times = []
        for _ in range(runs):
            torch.cuda.synchronize()
            start = time.time()
            farthest_point_sampling_cuda_optimized(positions, num_samples)
            torch.cuda.synchronize()
            cuda_opt_times.append(time.time() - start)
        cuda_opt_avg = np.mean(cuda_opt_times)
        
        print(f"CPU版本:        {cpu_avg:.3f}s")
        print(f"CUDA版本:       {cuda_avg:.3f}s (加速 {cpu_avg/cuda_avg:.1f}x)")
        print(f"CUDA优化版本:   {cuda_opt_avg:.3f}s (加速 {cpu_avg/cuda_opt_avg:.1f}x)")
    else:
        print(f"CPU版本:        {cpu_avg:.3f}s")
        print("CUDA不可用")

def main():
    """主函数 - 添加性能测试选项"""
    if len(sys.argv) < 3:
        print("用法: python script.py <ply_path> <output_path> [max_points] [--benchmark]")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    output_path = sys.argv[2]
    max_points = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != '--benchmark' else None
    run_benchmark = '--benchmark' in sys.argv
    
    if run_benchmark and max_points:
        # 运行性能测试
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        positions = np.column_stack([vertex['x'], vertex['y'], vertex['z']])
        benchmark_fps_methods(positions, max_points)
        return
    
    # 正常转换
    success = ply_to_points3d_cuda(ply_path, output_path, max_points)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()