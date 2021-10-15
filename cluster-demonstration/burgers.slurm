#!/bin/bash
#SBATCH --job-name=Burgers-continues-inference        # create a short name for your job
#SBATCH --nodes=1                                     # node count
#SBATCH --ntasks=1                                    # total number of tasks across all nodes
#SBATCH --cpus-per-task=1                             # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:1                                  # number of gpu per node
#SBATCH --mem-per-cpu=4G                              # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=00:15:00                               # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2021.05

# The followings is required on Della
_conda_setup="$('/usr/licensed/anaconda3/2021.5/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/licensed/anaconda3/2021.5/etc/profile.d/conda.sh" ]; then
        . "/usr/licensed/anaconda3/2021.5/etc/profile.d/conda.sh"
    else
        export PATH="/usr/licensed/anaconda3/2021.5/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate tf24
python3 ./cluster-demonstration/Burgers_continous_inference_TF2.py
python3 ./cluster-demonstration/plots.py
