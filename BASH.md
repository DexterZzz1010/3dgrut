```
python tools/ply_to_points3d.py data/fisheye/point_cloud.ply data/fisheye/colmap/points3D.txt 10000000

```


```
python train.py --config-name apps/scannetpp_3dgrt.yaml path=data/fisheye out_dir=runs experiment_name=fisheye_3dgrt dataset.downsample_factor=2

```