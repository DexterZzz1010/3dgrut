#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH --time=unlimited
#SBATCH --output=/staging/fisheye/mthesis/run_logs/3DGRUT/%A_%a.out
# #SBATCH --partition=ztestpreemp

echo "$PARTITION_NAME"

data_dir="/staging/fisheye/mthesis/data"
# SINGULARITY_IMAGE="/staging/fisheye/mthesis/docker/3dgrut.sif"
SINGULARITY_IMAGE="/workspaces/s0002322/src/3dgrut/3dgrut.sif"
ORIGINAL_CODE_DIR="/workspaces/s0002322/src/3dgrut"
config_name="apps/scannetpp_3dgrt.yaml"

# Parse command-line arguments
binary=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config-name) config_name="$2"; shift ;;
        --binary) binary="--binary"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done


git_dir=$ORIGINAL_CODE_DIR
cd "$git_dir" || exit
git_commit=$(git rev-parse --short HEAD)
cd - || exit


TEMP_CODE_DIR=$(mktemp -d /staging/fisheye/mthesis/runs/"${git_commit}_3dgrut_${SLURM_JOB_ID}_XXXXXX")
cp -r "$ORIGINAL_CODE_DIR"/* "$TEMP_CODE_DIR"


experiment_name="fisheye_3dgrt"
model_name="${git_commit}_${experiment_name}_$(date +"%Y-%m-%d_%H-%M-%S")"
out_dir="runs/${model_name}"


singularity exec --nv \
    --bind "$TEMP_CODE_DIR":/3dgrut \
    --bind "$data_dir":/3dgrut/data \
    --bind /staging/fisheye/mthesis/runs:/3dgrut/runs \
    --pwd /3dgrut \
    "$SINGULARITY_IMAGE" \
    bash -c "
    export PYTHONUSERBASE=/tmp/python_user
    export PATH=$PYTHONUSERBASE/bin:$PATH
    export PYTHONPATH=$PYTHONUSERBASE/lib/python3.11/site-packages:$PYTHONPATH
    mkdir -p $PYTHONUSERBASE
    pip install --user nvidia-cuda-nvrtc-cu11
    python -c 'import torch; print(torch.cuda.is_available())'
    nvidia-smi
    export CC=/usr/bin/gcc-11
    export CXX=/usr/bin/g++-11
    export CUDAHOSTCXX=/usr/bin/g++-11
    export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-11'
    python train.py \
        --config-name '$config_name' \
        path=data/3dgrut_fisheye \
        out_dir='$out_dir' \
        experiment_name='$experiment_name' \
        dataset.downsample_factor=2
    "

# singularity exec --nv \
#     --bind "$TEMP_CODE_DIR":/3dgrut \
#     --pwd /3dgrut \
#     "$SINGULARITY_IMAGE" \
#     bash -c "
#     source /opt/conda/etc/profile.d/conda.sh
#     conda activate 3dgrut
    
#     python -c '
# import sys
# sys.path.insert(0, \"/3dgrut\")

# print(\"=== 测试基础导入 ===\")
# import torch
# import threedgrt_tracer
# print(\"导入成功\")

# print(\"=== 测试配置创建 ===\")
# from omegaconf import DictConfig
# test_conf = DictConfig({
#     \"render\": {
#         \"particle_radiance_sph_degree\": 2,
#         \"particle_kernel_degree\": 3,
#         \"particle_kernel_min_response\": 0.01,
#         \"particle_kernel_min_alpha\": 0.01,
#         \"particle_kernel_max_alpha\": 0.99,
#         \"enable_normals\": False,
#         \"primitive_type\": \"gaussian\"
#     }
# })
# print(\"配置创建成功\")

# print(\"=== 测试 Tracer 初始化（可能导致段错误）===\")
# try:
#     tracer = threedgrt_tracer.Tracer(test_conf)
#     print(\"Tracer 创建成功！\")
# except Exception as e:
#     print(f\"Tracer 创建失败: {e}\")
#     import traceback
#     traceback.print_exc()
# '
#     "
#
#EOF