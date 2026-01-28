import robotic as ry
import numpy as np
import time
import os
import pickle

# Use patched loader from Phase 1
try:
    from patched_urdf_io import URDFLoader
except ImportError:
    print("Error: patched_urdf_io.py not found. Please run this script from the generic_urdf directory.")
    exit(1)

class RySimRobot:
    """
    Wrapper around robotic.Config to provide a Gym-like interface.
    """
    def __init__(self, config):
        self.C = config
        self.Sim = ry.Simulation(config, ry.SimulationEngine.kinematic, 0)
        self.Sim.addSensor("sim_camera")
    
    def get_joint_state(self):
        """Returns the current joint angle vector (numpy array)."""
        return self.C.getJointState()
    
    def get_image(self):
        """
        Returns the RGB image from the 'sim_camera' frame.
        Assumes 'sim_camera' has been added to the configuration.
        """
        # view_getRgb() returns the rendered image from the specific frame
        return self.Sim.getImageAndDepth()[0]

    def step(self, action):
        """
        Sets the target joint angles (action) and updates the simulation.
        In this kinematic sim, we set the state directly.
        """
        self.C.setJointState(action)
        self.C.computeCollisions()
        # Sync simulation with config for rendering
        self.Sim.step([], 0.01, ry.ControlMode.none)
        # Ensure the viewer updates if open
        # self.C.view() # Optional here, usually handled in loop

def setup_scene(C):
    """
    Sets up the static environment (table) and dynamic elements (target box).
    """
    # 1. Add Table
    # A large static box below the robot
    # Position: [0, 0, -0.05], Size: [1, 1, 0.1]
    table = C.addFrame("table", "world")
    table.setPosition([0, 0, -0.05])
    table.setShape(ry.ST.box, [1, 1, 0.1])
    table.setColor([0.3, 0.3, 0.3]) # Dark gray
    table.setContact(1) # Enable collision

    # 2. Add Target Box
    # A small red cube that will be randomized later
    target = C.addFrame("target_box", "world")
    target.setShape(ry.ST.box, [0.04, 0.04, 0.04]) # 4cm cube
    target.setColor([1, 0, 0]) # Red
    # Initial position (will be overwritten in reset)
    target.setPosition([0.4, 0.0, 0.1]) 
    
    # 3. Position Camera
    # Ensure sim_camera acts as a sensor
    # (Assuming sim_camera was added by the import script or needs to be re-added if fresh config)
    camera_name = "sim_camera"
    if not C.getFrame(camera_name):
        cam = C.addFrame(camera_name, "world")
        cam.setPosition([0.5, 0.5, 0.5])
        cam.setQuaternion([0.5, -0.5, 0.5, -0.5]) # Look at origin
        # Important: To render, we might need to define attributes or just rely on view_getRgb
        # ry.Config.view_getRgb renders offscreen using the named frame's view

def main():
    # Load Robot
    urdf_path = os.path.abspath("Simulation/SO100/so100.urdf")
    print(f"Loading robot from: {urdf_path}")
    loader = URDFLoader(urdf_path)
    C = loader.C
    
    # Setup Scene
    setup_scene(C)
    
    # Initialize Wrapper
    robot = RySimRobot(C)
    
    # Visualization
    C.view()

    # Data Collection Configuration
    num_episodes = 50
    episode_data = []

    print(f"Starting data collection for {num_episodes} episodes...")

    for ep in range(num_episodes):
        # --- 1. Reset ---
        # Randomize target position
        # x in [0.3, 0.5], y in [-0.2, 0.2], z usually fixed or slight variation
        # Let's keep z at table height + half box size (approx 0.02 + table surface 0.0)
        rand_x = np.random.uniform(0.3, 0.5)
        rand_y = np.random.uniform(-0.2, 0.2)
        target_pos = [rand_x, rand_y, 0.05] 
        C.getFrame("target_box").setPosition(target_pos)
        
        # Reset robot to home configuration (all zeros) for start of episode
        q_home = C.getJointState() * 0.0
        C.setJointState(q_home)
        
        # --- 2. Plan with KOMO ---
        # phases=1 : Single phase movement
        # slices_per_phase=40 : Discretization steps
        # k_order=2 : Optimization considers acceleration (2nd derivative)
        komo = ry.KOMO(C, phases=1, slicesPerPhase=40, kOrder=2, enableCollisions=True)
        
        # Objectives:
        # 1. Reach the target: Eq constraint (type=OT.eq) on position difference (FS.positionDiff)
        #    between "gripper" and "target_box" at time=1.0 (end of phase)
        #    Scale=1e1 (high weight)
        komo.addObjective(times=[1.0], 
                          feature=ry.FS.positionDiff, 
                          frames=["gripper", "target_box"], 
                          type=ry.OT.eq, 
                          scale=[1e1])
        
        # 2. Smoothness: Sum of Squares (OT.sos) on joint velocity (order=1) and acceleration (order=2)
        #    qItself represents the joint angles
        komo.addObjective(times=[], # All times
                          feature=ry.FS.qItself, 
                          frames=[], 
                          type=ry.OT.sos, 
                          scale=[1e-1], 
                          order=1) # Minimize Velocity
                          
        komo.addObjective(times=[], 
                          feature=ry.FS.qItself, 
                          frames=[], 
                          type=ry.OT.sos, 
                          scale=[1e-2], 
                          order=2) # Minimize Acceleration

        # Solve
        nlp = komo.nlp()
        solver = ry.NLP_Solver()
        solver.setProblem(nlp)
        ret = solver.solve()
        # Check if valid (optional, but good for data quality)
        # inputs: ret.getReport()

        # --- 3. Execute and Record ---
        path = komo.getPath() # Returns [steps, q_dim] array
        
        episode_trajectory = []
        
        for t in range(len(path)):
            # Current state
            q_current = robot.get_joint_state()
            img = robot.get_image()
            
            # Action: Next configuration in the plan
            # (In a real RL setting, action might be delta, but here we store the target q)
            q_target = path[t]
            
            # Store step
            step_dict = {
                "observation": q_current,
                "image": img,
                "action": q_target,
                "target_pos": target_pos
            }
            episode_trajectory.append(step_dict)
            
            # Execute step
            robot.step(q_target)
            
            # Small delay for visualization
            # time.sleep(0.01) 
        
        episode_data.append(episode_trajectory)
        print(f"Episode {ep+1}/{num_episodes} complete. Path length: {len(path)}")

    # --- 4. Save ---
    output_file = "raw_dataset_sim.pkl"
    with open(output_file, "wb") as f:
        pickle.dump(episode_data, f)
    
    print(f"Data collection complete. Saved {len(episode_data)} episodes to {output_file}.")
    del C

if __name__ == "__main__":
    main()
