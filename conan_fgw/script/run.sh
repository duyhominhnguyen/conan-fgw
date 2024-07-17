if [ ! -f .env ]
then
  export $(cat .env | xargs)
fi
export WORKDIR=$(pwd)
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

DATE=$(date +"%Y-%m-%d-%T")

model=schnet
task=property_regression
ds=lipo
n_cfm=3
runs=3

python conan_fgw/src/train_val.py \
        --config_path=${WORKDIR}/conan_fgw/config/$model/$task/$ds/$ds\_$n_cfm.yaml \
        --cuda_device=0 \
        --data_root=${WORKDIR} \
        --number_of_runs=$runs \
        --checkpoints_dir=${WORKDIR}/models \
        --logs_dir=${WORKDIR}/outputs \
        --run_name=$model\_$ds\_$n_cfm \
        --stage=initial \
        --model_name=${model} \
        --run_id=$DATE

export CUDA_VISIBLE_DEVICES=0,1,2,3

python conan_fgw/src/train_val.py \
        --config_path=${WORKDIR}/conan_fgw/config/$model/$task/$ds/$ds\_$n_cfm\_bc.yaml \
        --cuda_device=0 \
        --data_root=${WORKDIR} \
        --number_of_runs=$runs \
        --checkpoints_dir=${WORKDIR}/models \
        --logs_dir=${WORKDIR}/outputs \
        --run_name=$model\_$ds\_$n_cfm \
        --stage=fgw \
        --model_name=${model} \
        --run_id=$DATE \
        --initial_ckpt_dir=${WORKDIR}/models/$model\_$ds\_$n_cfm/$DATE