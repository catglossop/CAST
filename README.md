# CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models

## Running the CAST data generation pipeline

### Setup 

First, clone the repository, also pulling the submodules
```
git clone https://github.com/catglossop/CAST.git --recursive
```

Set up the environment
```
cd CAST/
uv sync
```

We assume that your dataset takes the form used in the [GNM code base](https://github.com/robodhruv/visualnav-transformer). It can have a root directory with subdirectories for different datasets but should have a unique trajectory name for each trajectory (across all trajectories): 

```
root_dataset_dir
|____subdatset_dir_0
|    |___unique_traj_name_0
|    |      0.jpg
|    |      ...
|    |      N.jpg
|    |      traj_data.pkl
|    |___unique_traj_name_1
|    ...
|    |___unique_traj_name_X
|____subdataset_dir_Y
...
```

### Setting up GCP

To run the data generation pipeline, you will need to set up a GCP project (so that you can do batch inference with gemini). You can follow the steps to create a project [here](https://developers.google.com/workspace/guides/create-project). 

Once you have a project created, you should link it to your Gemini account. Go to [Google AI Studio](https://aistudio.google.com/prompts/new_chat) and click on "Get API key" on the bottom left. Click create API key and link your GCP project. 

In your terminal, you will need to login with gcloud

```
gcloud auth application-default login
gcloud auth application-default set-quota-project <your project>
```

To run the data augmentation code, first setup your config file. The template config file is provided in CAST/data/configs/data_gen.yaml with descriptions of the parameters. 

The data pipeline runs the following steps. You can individually enable each of these steps with the associated commandline argument: 
1. Hindsight labeling (--hindsight-step):
    a. Describe the images in a trajectory
    b. Summarize the descriptions into instructions
2. Filtering (--filter-step): Filter the instructions based on the atomic instructions and propose new instructions
3. Counterfactual generation (--cf-step): Generate counterfactual instructions and atomic commands
4. Action generation (--action-gen-step): Generate the atomic action chunks

To run an individual step, use the following command: 
```
python cast/data/generate_cast_dataset.py <--individual-step> --config-path <your config path>
```

To run all the steps, use the following command: 
```
python cast/data/generate_cast_dataset.py --run-all --config-path <your config path>
```

After the data is generated, convert the data into tf records. Fill in the TODOs with your values

```
cd data/data_conversion
tfds build
```

### Atomic Policy
The script for building an atomic dataset with the method in the paper is provided in the repo as well. You can perform these two steps: 

*Step 1*: Build the atomic dataset (use the same config as generate_cast_dataset.py)
```
python cast/data/build_atomic_dataset.py --config-path <your config path> (--rebuild)
```
Optionally, you can rerun the atomic dataset generation with `--rebuild`. 

To get the data splits (train and val), fill in `cast/atomic_policy/data/split_atomic_dataset.sh`
```
./cast/atomic_policy/data/split_atomic_dataset.sh
```

*Step 2*: Train the policy

Create a config, follow the template in `configs/atomic_model.yaml`

```
python cast/atomic_policy/train_atomic.py --config configs/atomic_model.yaml
```

### Convert to TFDS datasets 

Now you can modify `cast/data/conversion/cast_counterfactual/cast_counterfactual_dataset_builder.py` and `cast/data/conversion/cast_filtered/cast_filtered_dataset_builder.py` (see the variables at the top of the scripts). Then run the conversion script

```
cd cast/data/conversion/cast_counterfactual
tfds build
```

```
cf cast/data/conversion/cast_filtered
tfds build 
```

Optionally, to overwrite the dataset, you can append `--overwrite`

```
tfds build --overwrite
```

### Training your own CounterfactualVLA

To get started with training your own model with the CAST dataset, download and unzip the CAST dataset. Follow the instructions in the `cast-vla` repo to start your own training run. 

To start using CounterfactualVLA, download the checkpoint and follow the instructions for inference in the `cast-vla` repo.

Once the inference server has been launched using `cast-vla`, you can launch the robot side client script

```
cd deployment
./navigate_vla.sh '-s <ngrok server address> -w <number of steps to use in generated action chunk> --prompt "your prompt"'
```



