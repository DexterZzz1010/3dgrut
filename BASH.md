```
python tools/ply_to_points3d.py data/fisheye/point_cloud.ply data/fisheye/colmap/points3D.txt 10000000

```


```
python train.py --config-name apps/fisheye_3dgrt.yaml path=data/fisheye out_dir=runs experiment_name=fisheye_3dgrt dataset.downsample_factor=2

```


sbatch --partition=ztestpreemp scripts/3dgrut_train.sh --config-name apps/scannetpp_3dgrt.yaml --binary

sbatch --partition=ztestpreemp scripts/3dgrut_train.sh --config-name apps/fisheye_3dgut.yaml --binary


在drivelund上启动docker并使用vscode ssh
```
docker run -d   --name 3dgrut-dev   --gpus all   --ipc=host   --shm-size=16g   --restart unless-stopped   -v /home/yifei/workspace/3dgrut:/workspace   -v /home/yifei/workspace/3dgrut/data:/data   -w /workspace   3dgrut:latest   tail -f /dev/null
```

```
Dev Containers: Attach to Running Container…
```