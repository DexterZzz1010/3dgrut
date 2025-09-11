#!/usr/bin/env python3
"""
将PLY文件转换为COLMAP格式的points3D.txt文件
"""

import numpy as np
import sys
import os
from plyfile import PlyData, PlyElement

def read_ply_file(ply_path):
    """读取PLY文件"""
    print(f"读取PLY文件: {ply_path}")
    
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    
    # 提取坐标
    x = np.array(vertex['x'])
    y = np.array(vertex['y']) 
    z = np.array(vertex['z'])
    
    positions = np.column_stack([x, y, z])
    
    # 提取颜色（如果存在）
    colors = None
    if 'red' in vertex and 'green' in vertex and 'blue' in vertex:
        r = np.array(vertex['red'])
        g = np.array(vertex['green'])
        b = np.array(vertex['blue'])
        colors = np.column_stack([r, g, b])
    elif 'diffuse_red' in vertex and 'diffuse_green' in vertex and 'diffuse_blue' in vertex:
        r = np.array(vertex['diffuse_red']) * 255
        g = np.array(vertex['diffuse_green']) * 255
        b = np.array(vertex['diffuse_blue']) * 255
        colors = np.column_stack([r, g, b]).astype(np.uint8)
    else:
        # 默认颜色：基于Z坐标的渐变
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        r = (128 + 127 * z_norm).astype(np.uint8)
        g = (64 + 64 * z_norm).astype(np.uint8)
        b = (32 + 32 * z_norm).astype(np.uint8)
        colors = np.column_stack([r, g, b])
    
    print(f"读取到 {len(positions)} 个点")
    print(f"坐标范围: X[{x.min():.2f}, {x.max():.2f}], Y[{y.min():.2f}, {y.max():.2f}], Z[{z.min():.2f}, {z.max():.2f}]")
    
    return positions, colors

def write_points3d_txt(positions, colors, output_path, max_points=None):
    """写入points3D.txt文件"""
    
    if max_points and len(positions) > max_points:
        # 随机采样
        indices = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        print(f"随机采样到 {max_points} 个点")
    
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(positions)}\n")
        
        for i, (pos, color) in enumerate(zip(positions, colors)):
            x, y, z = pos
            r, g, b = color.astype(int)
            # 使用默认误差值，不包含track信息
            f.write(f"{i+1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 1.0\n")
    
    print(f"生成points3D.txt: {output_path}")

def convert_ply_to_points3d(ply_path, output_path, max_points=None):
    """转换PLY到points3D.txt"""
    
    if not os.path.exists(ply_path):
        print(f"错误: PLY文件不存在: {ply_path}")
        return False
    
    try:
        # 读取PLY文件
        positions, colors = read_ply_file(ply_path)
        
        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 写入points3D.txt
        write_points3d_txt(positions, colors, output_path, max_points)
        
        return True
        
    except Exception as e:
        print(f"转换失败: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("用法: python ply_to_points3d.py <input.ply> <output_points3D.txt> [max_points]")
        print("示例: python ply_to_points3d.py reconstruction.ply data/fisheye/colmap/points3D.txt 10000")
        sys.exit(1)
    
    ply_path = sys.argv[1]
    output_path = sys.argv[2]
    max_points = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    if convert_ply_to_points3d(ply_path, output_path, max_points):
        print("转换成功!")
    else:
        print("转换失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()