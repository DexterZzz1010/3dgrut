```
python tools/ply_to_points3d.py data/fisheye/point_cloud.ply data/fisheye/colmap/points3D.txt 10000000

```


```
python train.py --config-name apps/scannetpp_3dgrt.yaml path=data/fisheye out_dir=runs experiment_name=fisheye_3dgrt dataset.downsample_factor=2

```


sbatch --partition=ztestpreemp scripts/3dgrut_train.sh --config-name apps/scannetpp_3dgrt.yaml --binary

sbatch --partition=ztestpreemp scripts/3dgrut_train.sh --config-name apps/fisheye_3dgut.yaml --binary


把mask加载到文件夹
```
cd data/3dgrut_fisheye

# 定义所有相机文件夹
cameras=("FC" "FISHB" "FISHF" "FISHL" "FISHR")

for camera in "${cameras[@]}"; do
    echo "=== 处理 $camera 相机的mask文件 ==="
    
    # 确保目标目录存在
    mkdir -p "image_undistorted_fisheye/$camera"
    
    # 处理.png格式的mask文件
    for file in colmap/masks_rectified/$camera/*.png; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            
            # 处理双扩展名情况：original_images_000175.jpg.png
            if [[ "$filename" == *.jpg.png ]]; then
                base_name="${filename%.jpg.png}"
            else
                # 普通.png文件：original_images_000175.png
                base_name="${filename%.png}"
            fi
            
            target_file="image_undistorted_fisheye/$camera/${base_name}_mask.png"
            echo "复制: $filename -> ${base_name}_mask.png"
            cp "$file" "$target_file"
        fi
    done
    
    # 统计结果
    mask_count=$(ls image_undistorted_fisheye/$camera/*_mask.png 2>/dev/null | wc -l)
    echo "✓ $camera: 复制了 $mask_count 个mask文件"
    echo
done

```