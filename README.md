# CADTextures

Work in progress experiments for CAD texturing project.

### Dependencies

Check out `requirements.txt` for dependencies. 

### Basic Example

Run the following command for an overfitting example on the checked in data: 
```bash

```
Logs are uploaded to [weights and biases](https://wandb.ai/).

### File Structure


```
config/                             # refer to hydra on how these configs are used
    ├── dataset/                    # dataset config group
    ├── inputs/                     # input related config group, i.e. what inputs go in the network
    ├── model/                      # input group related to model params
    ├── base.yaml                   # hydra config#1
    ├── gan_conditional.yaml        # hydra config#2
    ├── gan_pure.yaml               # hydra config#3

data/                               # raw training data for datasets
    ├── <dataset_name>/
        ├──<dataset_subname>/
            ├──<sample_folder>/     # one folder per sample
            ├──<sample_folder>/     # each folder contains data for sample
            ├──<sample_folder>      # e.g. rgb, normal, texture map etc.

data_processing/                    # stuff for creating raw data in data/ folder
    ├── blender                     # blender render scripts and bpy files
    ├── sdf_gen/                    # sdf-gen for mesh to df conversion 
    ├── slurm/                      # slurm scripts for dataprocessing
    ├── create_cube_models.py       # cube dataset related dataprocessing functions
    ├── create_render_normals_df.py # driving script for blender / sdf-gen
    ├── create_splits.py            # random train / val split
    ├── future_categories.py        # 3D-Future category parsing

model/                              # neural network models
    ├── discriminator.py            # discriminator architectures
    ├── texture_gan.py              # generator architectures

slurm/                              # slurm driving scripts
test/                               # unit tests

trainer/                                # pytorch-lightning trainer scripts
    ├── train_texture_map_predictor.py  # for training texture gan
    ├── train_pure_gan.py               # for training pure gan generating uniform colors

util/                               # utility scripts
    ├── feature_loss.py             # style / content loss helper
    ├── gan_loss.py                 # gan loss helper
    ├── regression_loss.py          # l1/l2 loss helper
    ├── timer.py                    # timer utility
    ├── filesystem_logger.py        # workaround for pl-logger multi-gpu single folder output
    ├── misc.py                     # misc. utility functions
```