conda create -n roboagent python=3.8
conda activate roboagent
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco
pip install dm_control
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install h5py_cache
cd detr && pip install -e .