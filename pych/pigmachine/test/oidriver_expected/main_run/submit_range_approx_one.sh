#!/bin/bash

#SBATCH -J oidriver
#SBATCH -o oidriver.%j.out
#SBATCH -e oidriver.%j.err
#SBATCH -N 1
#SBATCH -n 28
#SBATCH -p normal
#SBATCH -t 72:00:00


eval "$(conda shell.bash hook)"
conda activate py38_tim

python3 -c "import pych.pigmachine.OIDriver as OIDriver;oid = OIDriver('test_write','range_approx_one')"
