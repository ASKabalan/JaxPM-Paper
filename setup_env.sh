module load arch/a100
module load gcc/11.3.1 openmpi/4.1.5
pip install jax[cuda]
pip install --no-build-isolation -r requirements.txt