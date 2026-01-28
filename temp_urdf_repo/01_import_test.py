import robotic as ry
import time
import os

def main():
    # Path to URDF
    # Relative to this script (in temp_urdf_repo)
    urdf_path = os.path.abspath("Simulation/SO100/so100.urdf")
    print(f"Loading URDF from: {urdf_path}")

    try:
        from patched_urdf_io import URDFLoader
    except ImportError:
        print("Could not import URDFLoader from patched_urdf_io")
        return

    # C = ry.Config()
    # C.addFile(urdf_path)
    
    loader = URDFLoader(urdf_path)
    C = loader.C

    print("Frame names:")
    print(C.getFrameNames())

    # Add Camera
    # Adjust position to look at the robot (assuming robot base is at 0,0,0)
    cam = C.addFrame("sim_camera", "world")
    cam.setPosition([0.5, 0.5, 0.5])
    cam.setQuaternion([0.5, -0.5, 0.5, -0.5]) 
    
    # Show viewer
    C.view()

    current_frames = C.getFrameNames()
    if 'base' in current_frames and 'gripper' in current_frames:
        print("\nSUCCESS: 'base' and 'gripper' frames found.")
    else:
        print("\nWARNING: Critical frames 'base' or 'gripper' NOT found.")

    print("\nCheck the viewer window.")
    print("If you see stick figures, the meshes failed to load.")
    print("If you see a full robot model, meshes loaded correctly.")
    print("Press Ctrl+C to exit.")

    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            break
    
    del C # Clean up

if __name__ == "__main__":
    main()
