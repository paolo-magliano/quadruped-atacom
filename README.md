# SafeLocomotion

## Installation setup
1. Install Isaac Sim: https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html
2. Locate the python executable in Isaac Sim
    ```bash
    alias PY=~/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh
    ```
3. Install OmniIsaacGymEnvs and create Virtual environment: https://github.com/isaac-sim/OmniIsaacGymEnvs 
    ```bash
    git clone https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs.git
    PY -m pip install -e OmniIsaacGymEnvs
    ```
4. Install MushroomRL 2.0.0
    ```bash
    git clone https://github.com/MushroomRL/mushroom-rl.git
    PY -m pip install -e mushroom-rl
    ```
5. Install SafeLocomotion
    ```bash
    git clone https://github.com/paolo-magliano/SafeLocomotion.git
    PY -m pip install -e SafeLocomotion
    ```
6. Run example script:
    ```bash
    cd SafeLocomotion
    PY experiments/isaac_atacom.py
    ```