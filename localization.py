import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load and parse the robot pose data
pose_data = {
    "x": 0.032211639,
    "y": 0.0307900608,
    "theta": -1.86058533
}

# Map settings
map_settings = {
    "resolution": 0.03,
    "origin": [-2.789049911499023, -21.67792419433594, 0]
}

def world_to_pixel(x, y, resolution, origin):
    """Convert world coordinates to pixel coordinates"""
    pixel_x = int((x - origin[0]) / resolution)
    pixel_y = int((y - origin[1]) / resolution)
    return pixel_x, pixel_y

def visualize_robot_pose(map_path, pose_data, map_settings):
    # Load the map image
    map_img = mpimg.imread(map_path)
    
    # Convert robot position from world coordinates to pixel coordinates
    robot_x, robot_y = world_to_pixel(
        pose_data["x"], 
        pose_data["y"], 
        map_settings["resolution"], 
        map_settings["origin"]
    )
    
    # Create figure and plot map
    plt.figure(figsize=(12, 12))
    plt.imshow(map_img, cmap='gray')
    
    # Calculate arrow direction components
    arrow_length = 20  # pixels
    dx = arrow_length * np.cos(pose_data["theta"])
    dy = arrow_length * np.sin(pose_data["theta"])
    
    # Plot robot position and orientation
    plt.arrow(robot_x, map_img.shape[0] - robot_y, dx, -dy,  # Flip y-axis
              head_width=5, head_length=10, fc='r', ec='r', width=2)
    
    # Add title and labels
    plt.title('Robot Pose Visualization')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    
    # Add legend for robot pose
    plt.plot(robot_x, map_img.shape[0] - robot_y, 'ro', label='Robot Position')
    plt.legend()
    
    # Show grid
    plt.grid(True)
    plt.show()

# Visualize the robot pose
visualize_robot_pose('map.png', pose_data, map_settings)