# SafeLocomotion

## Installation setup
1. Install MushroomRL 2.0.0
    ```bash
    git clone https://github.com/MushroomRL/mushroom-rl.git
    python -m pip install -e mushroom-rl
    ```
2. Install SafeLocomotion
    ```bash
    git clone https://github.com/paolo-magliano/SafeLocomotion.git
    python -m pip install -e SafeLocomotion
    ```
3. Run example script:
    ```bash
    cd SafeLocomotion
    python experiments/isaac_atacom.py
    ```

## Issues
### Anymal C
The implementation of the Anymal C Env is not working. don't use it.