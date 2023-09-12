# Training
## CC ML practices

**Step 1: Remove all graphical display**
Edit your program such that it doesn't use a graphical display. All graphical results will have to be written on disk, and visualized on your personal computer, when the job is finished. For example, if you show plots using matplotlib, you need to write the plots to image files instead of showing them on screen.

**Step 2: Archiving a data set**
Shared storage on our clusters is not designed to handle lots of small files (they are optimized for very large files). Make sure that the data set which you need for your training is an archive format like tar, which you can then transfer to your job's compute node when the job starts. If you do not respect these rules, you risk causing enormous numbers of I/O operations on the shared filesystem, leading to performance issues on the cluster for all of its users. If you want to learn more about how to handle collections of large number of files, we recommend that you spend some time reading this page.

Assuming that the files which you need are in the directory mydataset:
    tar cf mydataset.tar mydataset/*
*The above command does not compress the data.*

**Step 3: Preparing your virtual environment**
Create a virtual environment in your home space.
For details on installation and usage of machine learning frameworks, refer to our documentation:

**Step 4: Interactive job (salloc)**
We recommend that you try running your job in an interactive job before submitting it using a script (discussed in the following section). You can diagnose problems more quickly using an interactive job. An example of the command for submitting such a job is:

_salloc --account=def-someuser --gres=gpu:1 --cpus-per-task=3 --mem=32000M --time=1:00:00_
Once the job has started:
    - Activate your virtual environment.
    - Try to run your program.
    - Install any missing modules if necessary. Since the compute nodes don't have internet access, you will have to install them from a login node. Please refer to our documentation on virtual environments.
    - Note the steps that you took to make your program work.
    - Now is a good time to verify that your job reads and writes as much as possible on the compute node's local storage ($SLURM_TMPDIR) and as little as possible on the shared filesystems (home, scratch and project).

**Step 5: Scripted job (sbatch)**
You must submit your jobs using a script in conjunction with the sbatch command, so that they can be entirely automated as a batch process. Interactive jobs are just for preparing and debugging your jobs, so that you can execute them fully and/or at scale using sbatch.

Important elements of a sbatch script
    - Account that will be "billed" for the resources used
    - Resources required:
        - Number of CPUs, suggestion: 6
        - Number of GPUs, suggestion: 1 (Use one (1) single GPU, unless you are certain that your program can use several. By default, TensorFlow and PyTorch use just one GPU.)
        - Amount of memory, suggestion: 32000M
        - Duration (Maximum Béluga: 7 days, Graham and Cedar: 28 days)
    Bash commands:
        - Preparing your environment (modules, virtualenv)
        - Transferring data to the compute node
        - Starting the executable


EXAMPLE BATCH SCRIPT:
```
#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-03:00     # DD-HH:MM:SS

module load python/3.6 cuda cudnn

SOURCEDIR=~/ml-test

# Prepare virtualenv
source ~/my_env/bin/activate
# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.

# Prepare data
mkdir $SLURM_TMPDIR/data
tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

# Start training
python $SOURCEDIR/train.py $SLURM_TMPDIR/data
```

## Fold 0:

Completed 56 full epochs in 18 hours with 1 epoch taking 19m56s

|'Epoch 57:  58%█████▊     208/356 [09:46<06:57,  2.82s/it, loss=slurmstepd: error: *** JOB 7725767 ON gra1181 CANCELLED AT 2023-07-08T08:46:25 DUE TO TIME LIMIT ***
slurmstepd: error: *** STEP 7725767.0 ON gra1181 CANCELLED AT 2023-07-08T08:46:25 DUE TO TIME LIMIT ***

Epoch 0: 100%|██████████| 356/356 [19:56<00:00,  3.36s/it, loss=1.16] |
|----------------------------------|

SBATCH setup for fold 0:
```
!/bin/bash
- #SBATCH --account def-training-wa
- #SBATCH --gpus-per-node=t4:2
- #SBATCH --cpus-per-task=16
- #SBATCH --mem=32G
- #SBATCH --time=18:00:00
```
```
SLURM efficiency is:
Job ID: 7725767
Cluster: graham
User/Group: guest187/guests
State: TIMEOUT (exit code 0)
Nodes: 1
Cores per node: 16
CPU Utilized: 19:36:11
CPU Efficiency: 6.80% of 12-00:06:08 core-walltime
Job Wall-clock time: 18:00:23
Memory Utilized: 3.18 GB
Memory Efficiency: 9.93% of 32.00 GB
```
Attempted increasing hardware requests
 - Accepts 3 GPUS
 - need to include srun --ntasks-per-node
 - 128G too much or too little memory requested?
    - tried with variations of cpu per task ranging from 8 to 16
- Error seems GPU related?
    --> Are we limited to two GPUs??

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 512.00 MiB (GPU 0; 14.58 GiB total capacity; 8.50 GiB already allocated; 349.31 MiB free; 8.75 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
Epoch 0:   0%|          | 0/356 [00:13<?, ?it/s]srun: error: gra1177: task 0: Exited with exit code 1
srun: launch/slurm: _step_signal: Terminating StepId=8867242.2
slurmstepd: error: *** STEP 8867242.2 ON gra1177 CANCELLED AT 2023-07-09T19:18:33 ***
srun: error: gra1181: task 2: Exited with exit code 1
bypassing sigterm
bypassing sigterm
srun: error: gra1177: task 1: Killed
srun: error: gra1181: task 3: Killed
srun: Force Terminated StepId=8867242.2
```

## Fold 1
### New SBATCH params == saving only 2m40s

```
#SBATCH --nodes=2
#SBATCH --gpus-per-node=t4:3
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=12G
n tasks == 2

Cluster: graham
User/Group: guest187/guests
State: CANCELLED (exit code 0)
Nodes: 2
Cores per node: 6
CPU Utilized: 01:29:47
CPU Efficiency: 23.76% of 06:17:48 core-walltime
Job Wall-clock time: 00:31:29
Memory Utilized: 5.68 GB (estimated maximum)
Memory Efficiency: 3.94% of 144.00 GB (12.00 GB/core)
```

Epoch 0: 100%|██████████| 260/260 [17:18<00:00,  3.99s/it, loss=2.06]

## Set granularity in OptiNet
This function is meant to help with ensuring multi-GPU usage is split properly
It does not work
I have tried using available cuda_python instead of libartcuda but it says package not found (installed and triple checked avail wheels in hackathon python set up)
```
##########
# --gpus-per-node=t4:3 --cpus-per-task=12 --mem-per-cpu=8G --time=0:45:00 --nodes=2 --account def-training-wa

# WORKS
# --gpus-per-node=t4:3 --cpus-per-task=6 --mem-per-cpu=12G --time=00:15:00 --account def-training-wa
## This uses only one node even when --nodes 2 is part of script call
## Gets stuck at model callback summary

#SBATCH --gpus-per-node=t4:3
#SBATCH --cpus-per-task=22
#SBATCH --mem-per-cpu=6G
#SBATCH --time=12:00:00 

###########
# FAILED ATTEMPTS:
# --nodes=2 --gpus-per-node=t4:3 --cpus-per-task=6 --mem-per-cpu=16G --time=00:15:00 --account def-training-wa
### This receives GPU memory error

# tried removing srun command: PossibleUserWarning: The `srun` command is available on your system but is not used. HINT: If your intention is to run Lightning on SLURM, prepend your python command with `srun` like so: srun python3 /home/guest187/BrainHackProject/nnUNet/main.py --da ...  rank_zero_warn(Using 16bit native Automatic Mixed Precision (AMP)Trainer already configured with model summary callbacks: [<class 'pytorch_lightning.callbacks.model_summary.ModelSummary'>]. Skipping setting a default `ModelSummary` callback.

```

Working:
#!/bin/bash
#SBATCH --account def-training-wa
#SBATCH --gpus-per-node=t4:3
#SBATCH --cpus-per-task=22
#SBATCH --mem-per-cpu=6G
#SBATCH --time=12:00:00
#SBATCH --mail-user=amodar7@gmail.com
#SBATCH --mail-type=ALL

srun python3 /home/guest187/BrainHackProject/nnUNet/main.py --data $data_dirTr --results $results_dirTr --ckpt_store_dir $ckpt_store --brats --depth 6 --filters 64 96 128 192 256 384 512 --scheduler --learning_rate 0.0005 --epochs 50 --fold 1 --amp --gpus 3 --task 11 --save_ckpt --nfolds 10 --nodes 2
df = pd.read_csv(f,sep=';',header=None, names=['PARTITION','AVAIL','TIMELIMIT','NODE','STATE','NODELIST'])

# NB: to run optinet we must have older versions of packages as some functions are deprecated
git+https://github.com/NVIDIA/dllogger
git+https://github.com/NVIDIA/mlperf-common.git
nibabel==3.2.1
joblib==1.0.1
pytorch-lightning==1.7.7
scikit-learn==1.0
scikit-image==0.18.3
scipy==1.8.1
rich==12.5.0