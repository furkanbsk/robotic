import pickle
import numpy as np
import torch
from datasets import Dataset, Features, Sequence, Value, Image
import PIL.Image

def main():
    input_file = "raw_dataset_sim.pkl"
    output_dir = "lerobot_dataset"
    
    print(f"Loading data from {input_file}...")
    with open(input_file, "rb") as f:
        raw_data = pickle.load(f)
    print(f"Loaded {len(raw_data)} episodes.")

    # Schema Definition
    # 6 DOF assumed for SO-ARM100 (state and action)
    features = Features({
        "observation.state": Sequence(Value("float32"), length=6),
        "observation.images.top": Image(),
        "action": Sequence(Value("float32"), length=6),
        "episode_index": Value("int64"),
        "frame_index": Value("int64"),
        "timestamp": Value("float32"),
        "next.done": Value("bool"),
        # "observation.environment_state": Sequence(Value("float32"), length=3) # Optional: box pose
    })

    # Prepare data for HF Dataset
    # We flatten the list of episodes into a list of samples
    data_dict = {
        "observation.state": [],
        "observation.images.top": [],
        "action": [],
        "episode_index": [],
        "frame_index": [],
        "timestamp": [],
        "next.done": []
    }

    print("Processing episodes...")
    for episode_idx, episode in enumerate(raw_data):
        for frame_idx, step in enumerate(episode):
            # step keys: 'observation', 'image', 'action', 'target_pos'
            
            # 1. State
            q = np.array(step['observation'], dtype=np.float32)
            data_dict["observation.state"].append(q)
            
            # 2. Image
            # ry returns RGB array [h, w, 3], dataset needs PIL Image or valid array
            # Assuming channel last from ry (check if it's correct visually later)
            img_array = step['image']
            # If float [0,1], convert to [0,255] uint8
            if img_array.dtype == np.float32 or img_array.dtype == np.float64:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
            
            # Convert to PIL Image for the Image() feature
            img_pil = PIL.Image.fromarray(img_array)
            data_dict["observation.images.top"].append(img_pil)
            
            # 3. Action
            action = np.array(step['action'], dtype=np.float32)
            data_dict["action"].append(action)
            
            # 4. Metadata
            data_dict["episode_index"].append(episode_idx)
            data_dict["frame_index"].append(frame_idx)
            data_dict["timestamp"].append(frame_idx * 0.1) # Assuming 10Hz (0.1s step) as per main loop pauses isn't precise but good enough for sim
            
            # 5. Done flag
            is_done = (frame_idx == len(episode) - 1)
            data_dict["next.done"].append(is_done)

    print(f"Flattened data into {len(data_dict['episode_index'])} samples.")
    print("Creating Hugging Face Dataset...")
    
    # Create Dataset
    # from_dict with features argument ensures strict schema
    dataset = Dataset.from_dict(data_dict, features=features)
    
    # Save to disk
    print(f"Saving dataset to {output_dir}...")
    dataset.save_to_disk(output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
