# CADTextures

Work in progress experiments for CAD texturing project.

### Dependencies

Check out `requirements.txt` for dependencies. Additionally, you'll need [PyMarchingCubes](https://github.com/JustusThies/PyMarchingCubes).

### Basic Example

Download the data

```bash

```

Make sure that the data exists in `data/` folder.

Run the following command for an overfitting example on the checked in data: 
```bash
# make sure current directory is in python path
export PYTHONPATH=.
# run training script
python trainer/train_end2end_retrieval_fuse.py sanity_steps=1 dataset=single_cube_textures_overfit_32 dataset.preload=True dictionary.patch_size=16 experiment=test_overfitting wandb_main=True val_check_interval=5 save_epoch=5 warmup_epochs_constrastive=10
```

For generalization use `dataset=single_cube_textures`, `val_check_interval=30`, `save_epoch=30`, `warmup_epochs_constrastive=120`.

Logs are uploaded to [weights and biases](https://wandb.ai/), so you may need to be signed in there.

### File Structure


```
.editorconfig                           # code style

config/                                 # refer to hydra on how these configs are used
    ├── dataset/                        # dataset config group
    ├── inputs/                         # input related config group, i.e. inputs to the network
    ├── model/                          # input group related to model params
    ├── base.yaml                       # hydra config#1
    ├── gan_conditional.yaml            # hydra config#2
    ├── gan_pure.yaml                   # hydra config#3

data/                                   # raw training data for datasets
    ├── <dataset_name>/
        ├──<dataset_subname>/
            ├──<sample_folder>/         # one folder per sample
            ├──<sample_folder>/         # each folder contains data for sample
            ├──<sample_folder>          # e.g. rgb, normal, texture map etc.

dataset/
    ├── noise_dataset.py                # dataset for pure gan experiments
    ├── texture_completion_dataset.py   # dataset for inpainting style experiments (small holes missing)
    ├── texture_end2end_dataset.py      # dataset for end to end retrieval fuse style experiments
    ├── texture_map_dataset.py          # dataset for texture gan / l1 regression experiments
    ├── texture_patch_dataset.py        # dataset for texture optimization experiments

data_processing/                        # stuff for creating raw data in data/ folder
    ├── blender                         # blender render scripts and bpy files
    ├── sdf_gen/                        # sdf-gen for mesh to df conversion 
    ├── slurm/                          # slurm scripts for dataprocessing
    ├── create_cube_models.py           # cube dataset related dataprocessing functions
    ├── create_render_normals_df.py     # driving script for blender / sdf-gen
    ├── create_splits.py                # random train / val split
    ├── future_categories.py            # 3D-Future category parsing

model/                                  # neural network models
    ├── attention.py                    # retrieval fuse attention module
    ├── differentiable_retrieval.py     # differentiable argmax
    ├── discriminator.py                # discriminator architectures
    ├── refinement                      # retrieval fuse refinment modules
    ├── retrieval                       # retrieval fuse retrieval modules 
    ├── texture_gan.py                  # texture gan generator architectures
slurm/                                  # slurm driving scripts
test/                                   # unit tests
runs/                                   # output logs and results from training

trainer/                                # pytorch-lightning trainer scripts
    ├── train_end2end_retrieval_fuse.py # for training end-to-end retrieval and refinement
    ├── train_patch_optimization.py     # pure optimization of texture
    ├── train_patch_refinement.py       # retrieval fuse refinement
    ├── train_patch_retrieval.py        # retrieval fuse retrieval
    ├── train_pure_gan.py               # for training pure gan generating uniform colors
    ├── train_texture_completion.py     # inpainting / hole filling
    ├── train_texture_map_predictor.py  # for training texture gan
    
util/                                   # utility scripts
    ├── contrastive_loss.py             # contrastive loss helper
    ├── feature_loss.py                 # style / content loss helper
    ├── filesystem_logger.py            # workaround for pl-logger multi-gpu single folder output
    ├── gan_loss.py                     # gan loss helper
    ├── inference_end2end.py            # for kNN inference of end2end models 
    ├── regression_loss.py              # l1/l2 loss helper
    ├── timer.py                        # timer utility
    ├── retrieval.py                    # retrieval utils from retrieval fuse
    ├── misc.py                         # misc. utility functions
```