#!/bin/bash
#!
#! Example SLURM job script for Wilkes2 (Broadwell, ConnectX-4, P100)
#! Last updated: Mon 13 Nov 12:06:57 GMT 2017
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J gpujob
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A OZTIRELI-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 3 cpus per GPU.
#SBATCH --gres=gpu:4
#! How much wallclock time will be required?
#SBATCH --time=5:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=ALL
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime. 

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location 
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment

#! Insert additional module load commands after this line if needed:

source /usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh
conda activate nerf_pl_new

#! Full path to application executable: 
#! application="/rds/user/tw554/hpc-work/workspace/hypernerf/foo.sh"
#! application="/rds/user/tw554/hpc-work/workspace/hypernerf/$TARGET_SCRIPT"
#! application = $app

#! Run options for the application:
options=""

#! Work directory (i.e. where the job will run):
workdir="/rds/user/tw554/hpc-work/workspace/nerf_pl"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 12:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

# if [ "$SLURM_JOB_NODELIST" ]; then
#         #! Create a machine file:
#         export NODEFILE=`generate_pbs_nodefile`
#         cat $NODEFILE | uniq > machine.file.$JOBID
#         echo -e "\nNodes allocated:\n================"
#         echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
# fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

# BLENDER_DIR="data/nerf_synthetic/lego"
# EXP_NAME="debug"

# python train.py \
#    --dataset_name blender \
#    --root_dir $BLENDER_DIR \
#    --N_importance 64 --img_wh 400 400 --noise_std 0 \
#    --num_epochs 20 --batch_size 1024 \
#    --optimizer adam --lr 5e-4 --lr_scheduler cosine \
#    --exp_name $EXP_NAME \
#    --data_perturb occ \
#    --encode_t --beta_min 0.1 --num_gpus 4

DATA_NAME="kubric_single_car_rand_v2"
DIR="data/hypernerf/$DATA_NAME"
EXP_NAME="t"

# python train.py \
#    --dataset_name llff \
#    --root_dir $DIR \
#    --N_importance 64 --img_wh 256 256 \
#    --lr_scheduler cosine \
#    --exp_name $DATA_NAME/$EXP_NAME \
#    --encode_t --N_vocab 300 --num_gpus 4

python eval.py  \
   --root_dir $DIR  \
   --dataset_name llff --split val --img_wh 256 256   \
   --N_importance 64   \
   --encode_t --N_vocab 300 \
   --ckpt_path ckpts/$DATA_NAME/$EXP_NAME/last.ckpt   \
   --scene_name $DATA_NAME/$EXP_NAME


#! sbatch ./run_slurm_train_eval_kb_sfm.sh 