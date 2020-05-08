#!/bin/sh
#SBATCH -N 1 # 1 node
#SBATCH --gres=gpu:4 # 4 GPUs
#SBATCH --job-name=seed_47_alg_A2C_n_ch_2_stages_234
#SBATCH --output=/gpfs/projects/hcli64/shaping/large_actObs_space//logs/seed_015_alg_A2C_n_ch_2_stages_234.out
#SBATCH -D /home/hcli64/hcli64745/
#SBATCH --time=48:00:00
#SBATCH -n 16 # number of tasks
#SBATCH -c 10 # 10 cpus-per-task
module purge
module load gcc/6.4.0
module load cuda/9.1
module load cudnn/7.1.3
module load openmpi/3.0.0
module load atlas/3.10.3
module load scalapack/2.0.2
module load fftw/3.3.7
module load szip/2.1.1
module load ffmpeg
module load opencv/3.4.1
module load python/3.6.5_ML

# COMPUTE NUBMER OF TASKS PER GPU
REQUESTED_GPUS=4 # ask for 4 gpus (each node has 4 gpus)
if [[ $SLURM_TASKS_PER_NODE -lt $REQUESTED_GPUS ]]; then
  NTASKS_PER_GPU=1
else
  NTASKS_PER_GPU=$((SLURM_TASKS_PER_NODE/$REQUESTED_GPUS))
fi

# LAUNCH RUNS
DEVICES=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n')
SEED=0
for DEVICE in $DEVICES; do
  export CUDA_VISIBLE_DEVICES=$DEVICE
  I=0
  while [[ $I -lt $NTASKS_PER_GPU ]]; do
    /home/hcli64/hcli64745/shaping_run.py --seed $SEED --alg A2C --n_ch 20 --stages 234 &
    declare "PID${DEVICE}${I}=$!"
    I=$(($I+1))
    SEED=$(($SEED+1))
  done
done

# CHECK THAT EVERYTHING IS FINE (?)
RET=0
for DEVICE in $DEVICES; do
  I=0
  while [[ $I -lt $NTASKS_PER_GPU ]]; do
    PID="PID${DEVICE}${I}"
    wait ${!PID}
    RET=$(($RET+$?))
    I=$(($I+1))
  done
done
exit $RET #if 0 all ok, else something failed
