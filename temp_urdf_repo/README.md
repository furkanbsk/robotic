# SO-ARM100 Synthetic Data Pipeline for LeRobot

This directory contains a complete pipeline for generating synthetic training data for the SO-ARM100 robot using the `robotic` simulator, and converting it for use with Hugging Face LeRobot.

## Prerequisites
- `robotic` library installed (`pip install robotic` or source build)
- `numpy`, `torch`, `datasets`, `Pillow`

## Pipeline Overview

### 1. Verification & Import
**Script:** `01_import_test.py`
- Loads the SO-ARM100 URDF using a custom `patched_urdf_io.py`.
- Visualizes the robot to ensure meshes and frames are correct.
- **Run:** `python3 01_import_test.py`

### 2. Data Generation (KOMO)
**Script:** `02_data_collection.py`
- Sets up a simulation scene with a table and a randomized target box.
- Uses **KOMO** (k-Order Markov Optimization) to plan smooth, expert-level trajectories to the target.
- Captures RGB images via `ry.Simulation`.
- **Output:** `raw_dataset_sim.pkl` (contains raw episodes).
- **Run:** `python3 02_data_collection.py`

### 3. LeRobot Conversion
**Script:** `03_convert_to_lerobot.py`
- Converts the raw pickle file into a **Hugging Face Dataset**.
- Enforces the LeRobot schema:
    - `observation.state` (6-DoF joint angles)
    - `observation.images.top` (RGB Camera)
    - `action` (Target joint angles)
- **Output:** `lerobot_dataset/` directory.
- **Run:** `python3 03_convert_to_lerobot.py`

## Files
- `patched_urdf_io.py`: A patched version of the `robotic` URDF loader to handle specific naming collisions in the SO-ARM100 URDF.
- `lerobot_dataset/`: The final generated dataset (ready for `datasets.load_from_disk`).
