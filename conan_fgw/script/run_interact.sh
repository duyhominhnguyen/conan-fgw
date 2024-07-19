#!/bin/bash
# Function to prompt the user for a selection from a list
select_option() {
    local prompt_message=$1
    shift
    local options=("$@")
    local default_value=${options[-1]}  # Last option is the default value

    PS3="$prompt_message"
    select choice in "${options[@]}"; do
        if [[ $REPLY -ge 1 && $REPLY -le ${#options[@]} ]]; then
            echo "$choice"
            break
        else
            echo 'Invalid selection. Please try again!' >&2
        fi
    done
}

# Define options for each variable
model_options=("visnet" "schnet")
dataset_options=("sars_cov" "sars_cov_2_gen" "bace" "esol" "freesolv" "lipo")
conformers_options=("3" "5" "10")
n_runs_options=("1" "5")


# Prompt the user for each variable
model=$(select_option 'Please select model: ' "${model_options[@]}")
ds=$(select_option 'Please select dataset: ' "${dataset_options[@]}")
n_cfm=$(select_option 'Please select number of conformers: ' "${conformers_options[@]}")
n_runs=$(select_option 'Please select number of runs: ' "${n_runs_options[@]}")

# Determine the task based on the dataset selected
if [[ " ${ds} " =~ " bace " || " ${ds} " =~ " esol " || " ${ds} " =~ " freesolv " || " ${ds} " =~ " lipo " ]]; then
    task="property_regression"
elif [[ " ${ds} " =~ " sars_cov " || " ${ds} " =~ " sars_cov_2_gen " ]]; then
    task="classification"
else
    echo "Dataset not recognized. Exiting." >&2
    exit 1
fi

# Display the collected inputs
echo
echo "--Collected inputs--"
echo "model: $model"
echo "task: $task"
echo "dataset: $ds"
echo "n_cfm: $n_cfm"
echo "n_runs: $n_runs"
echo "--------------------"

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

# Set the visible CUDA devices to the first GPU for conan_fgw_pre training stage
export CUDA_VISIBLE_DEVICES=0
# Run the conan_fgw_pre training stage
python conan_fgw/src/train_val.py \
        --config_path=${WORKDIR}/conan_fgw/config/$model/$task/$ds/$ds\_$n_cfm.yaml \
        --cuda_device=0 \
        --data_root=${WORKDIR} \
        --number_of_runs=$n_runs \
        --checkpoints_dir=${WORKDIR}/models \
        --logs_dir=${WORKDIR}/outputs \
        --run_name=$model\_$ds\_$n_cfm \
        --stage=conan_fgw_pre \
        --model_name=${model} \
        --run_id=$DATE

# Set the visible CUDA devices to GPUs 0, 1, 2, and 3 for using Distributed Data Parallel
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Run the FGW (Fused Gromov-Wasserstein) training stage
python conan_fgw/src/train_val.py \
        --config_path=${WORKDIR}/conan_fgw/config/$model/$task/$ds/$ds\_$n_cfm\_bc.yaml \
        --cuda_device=0 \
        --data_root=${WORKDIR} \
        --number_of_runs=$n_runs \
        --checkpoints_dir=${WORKDIR}/models \
        --logs_dir=${WORKDIR}/outputs \
        --run_name=$model\_$ds\_$n_cfm \
        --stage=conan_fgw \
        --model_name=${model} \
        --run_id=$DATE \
        --conan_fgw_pre_ckpt_dir=${WORKDIR}/models/$model\_$ds\_$n_cfm/$DATE