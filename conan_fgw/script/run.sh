# Check if the .env file exists
if [ ! -f .env ]
then
  # If the .env file exists, export the environment variables defined in it
  export $(cat .env | xargs)
fi

# Set the working directory to the current directory
export WORKDIR=$(pwd)
# Add the working directory to the PYTHONPATH
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
# Get the current date and time in the format YYYY-MM-DD-T
DATE=$(date +"%Y-%m-%d-%T")

# Define variables for the model, task, dataset, number of conformers, and number of runs
model=schnet   
task=property_regression
ds=esol ## lipo || esol || freesolv || bace
n_cfm_conan_fgw_pre=5
n_cfm_conan_fgw=5
runs=5

# Set the visible CUDA devices to the first GPU for conan_fgw_pre training stage
export CUDA_VISIBLE_DEVICES=0
# Run the conan_fgw_pre training stage
python conan_fgw/src/train_val.py \
        --config_path=${WORKDIR}/conan_fgw/config/$model/$task/$ds/$ds\_$n_cfm_conan_fgw_pre.yaml \
        --cuda_device=0 \
        --data_root=${WORKDIR} \
        --number_of_runs=$runs \
        --checkpoints_dir=${WORKDIR}/models \
        --logs_dir=${WORKDIR}/outputs \
        --run_name=$model\_$ds\_$n_cfm_conan_fgw_pre \
        --stage=conan_fgw_pre \
        --model_name=${model} \
        --run_id=$DATE \
        # --verbose

# Set the visible CUDA devices to GPUs 0, 1, 2, and 3 for using Distributed Data Parallel
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Run the FGW (Fused Gromov-Wasserstein) training stage
python conan_fgw/src/train_val.py \
        --config_path=${WORKDIR}/conan_fgw/config/$model/$task/$ds/$ds\_$n_cfm_conan_fgw\_bc.yaml \
        --cuda_device=0 \
        --data_root=${WORKDIR} \
        --number_of_runs=$runs \
        --checkpoints_dir=${WORKDIR}/models \
        --logs_dir=${WORKDIR}/outputs \
        --run_name=$model\_$ds\_$n_cfm_conan_fgw \
        --stage=conan_fgw \
        --model_name=${model} \
        --run_id=$DATE \
        --conan_fgw_pre_ckpt_dir=${WORKDIR}/models/$model\_$ds\_$n_cfm_conan_fgw_pre/$DATE \
        # --verbose