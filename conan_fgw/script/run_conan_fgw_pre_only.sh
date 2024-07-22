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
model=visnet                      
task=property_regression
ds=esol
n_cfm=5
runs=1

# Set the visible CUDA devices to the first GPU for conan_fgw_pre training stage
export CUDA_VISIBLE_DEVICES=3
# Run the conan_fgw_pre training stage
python conan_fgw/src/train_val.py \
        --config_path=${WORKDIR}/conan_fgw/config/$model/$task/$ds/$ds\_$n_cfm.yaml \
        --cuda_device=0 \
        --data_root=${WORKDIR} \
        --number_of_runs=$runs \
        --checkpoints_dir=${WORKDIR}/models \
        --logs_dir=${WORKDIR}/outputs \
        --run_name=$model\_$ds\_$n_cfm \
        --stage=conan_fgw_pre \
        --model_name=${model} \
        --run_id=$DATE