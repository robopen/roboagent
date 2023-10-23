
# RoboAgent

This repo contains code for training RoboAgent with the RoboSet dataset and evaluating on Franka Panda arms for manipulation tasks. For videos and details about the work, please refer to the project website https://robopen.github.io/ 



![RoboAgent](./static/roboagent_showcase.gif)


## Installation

Run `bash install.sh` for creating conda environment and installing dependencies. In addition, for interfacing with hardware, follow the instructions for installting RoboHive https://sites.google.com/view/robohive 

## Dataset

Follow the instructions on the RoboSet page for downloading the dataset https://robopen.github.io/roboset/

## Multi-Task Training template

To run multi-task with task-embedding conditioned policies. Check constants.py for Task embeddings and modify utils.py to add / modify new tasks. Task names are loaded from data file names. `--rn` is just for wandb logging.

We would recommend having all training .h5 files in a single folder that is specified in `--dataset_dir` to avoid errors - otherwise you can specify appropriate sub-folders with regex. `utils.py` will automatically parse all `.h5` files in the specified folders

```
python train.py --dataset_dir <path to data directory> --ckpt_dir <where to save ckpts> --policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 --seed 0 --temporal_agg --num_epochs 2000 --lr 1e-5 --multi_task --rn multi_task_run
```



## For evaluating on hardware 

Omit the `--multi_task` flag for single task policies. Make sure other parameters are same as training
```
python evaluate.py -e rpFrankaRobotiqDataRP02-v0 --ckpt_dir <where to load ckpts> --policy_class ACT --kl_weight 10 --chunk_size 20 --hidden_dim 512 --batch_size 64 --dim_feedforward 3200 --num_repeat 10 --task_name pick_butter --multi_task
```


## Acknowledgement

We thank Tony Zhao for help with the ACT codebase https://github.com/tonyzhaozh/act and discussions regarding ACT. RoboAgent's inference and data collection (teleop) pipeline is based on RoboHive https://sites.google.com/view/robohive 

## License

The RoboAgent code and RoboSet datasets are both licensed under MIT License

## Citation

If you find the repository helpful, please consider citing our paper

```
@misc{bharadhwaj2023roboagent,
                            title={RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking},
                            author={Homanga Bharadhwaj and Jay Vakil and Mohit Sharma and Abhinav Gupta and Shubham Tulsiani and Vikash Kumar},
                            year={2023},
                            eprint={2309.01918},
                            archivePrefix={arXiv},
                            primaryClass={cs.RO}
                      }
```
