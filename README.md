# Morphing Datasets

This package wraps around my other package `morphing-agents` and provides a dataset collection utility. One application for such datasets is for model-based optimization of robot morphologies. 

## Installation

You may download the code and install the dependenmcies with the following snippet.

```bash
git clone https://github.com/brandontrabucco/morphing-datasets.git
conda env create -f morphing-datasets/environment.yml
conda activate morphing-datasets
```

## Data Collection

You may start data collection, and try out other settings, with the following command.

```bash
python morphing-datasets/make_data.py \
  --local-dir LOCAL_DIR \
  --num-legs NUM_LEGS \
  --dataset-size DATASET_SIZE \
  --num-parallel NUM_PARALLEL \
  --num-gpus NUM_GPUS \
  --n-envs N_ENVS \
  --max-episode-steps MAX_EPISODE_STEPS \
  --total-timesteps TOTAL_TIMESTEPS \
  --noise-std NOISE_STD \
  --method {uniform,centered} \
  --domain {ant,dog,dkitty}
```
