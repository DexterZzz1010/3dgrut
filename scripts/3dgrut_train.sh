#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH --time=unlimited
#SBATCH --output=/staging/fisheye/mthesis/run_logs/3DGRUT/%A_%a.out
# #SBATCH --partition=ztestpreemp

echo "$PARTITION_NAME"

data_dir="/staging/fisheye/mthesis/data"
SINGULARITY_IMAGE="/staging/fisheye/mthesis/docker/3dgrut-ngc.sif"
# SINGULARITY_IMAGE="/workspaces/s0002322/src/3dgrut/3dgrut-ngc.sif"
ORIGINAL_CODE_DIR="/workspaces/s0002322/src/3dgrut"
# config_name="apps/scannetpp_3dgut.yaml"
config_name="apps/fisheye_3dgut.yaml"

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


experiment_name="fisheye_3dgut"
model_name="${git_commit}_${experiment_name}_$(date +"%Y-%m-%d_%H-%M-%S")"
out_dir="runs/${model_name}"


singularity exec --nv \
    --bind "$TEMP_CODE_DIR":/3dgrut \
    --bind "$data_dir":/3dgrut/data \
    --bind /staging/fisheye/mthesis/runs:/3dgrut/runs \
    --pwd /3dgrut \
    "$SINGULARITY_IMAGE" \
    bash -c "
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


#
#EOF