import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import requests
import json
import time
from matplotlib.animation import FuncAnimation

# API settings
URL = "http://192.168.1.87:5480/woosh/robot/PoseSpeed"
HEADERS = {'Content-Type': 'application/json'}
PAYLOAD = json.dumps({"robotId": 30001})

# Map settings
MAP_SETTINGS = {
    "resolution": 0.03,
    "origin": [-2.789049911499023, -21.67792419433594, 0]
}

def get_robot_pose():
    """Request robot pose from API"""
    try:
        response = requests.post(URL, headers=HEADERS, data=PAYLOAD)
        data = response.json()
        if data['ok']:
            return data['body']['pose']
        return None
    except Exception as e:
        print(f"Error getting pose: {e}")
        return None

def world_to_pixel(x, y, resolution, origin):
    """Convert world coordinates to pixel coordinates"""
    pixel_x = int((x - origin[0]) / resolution)
    pixel_y = int((y - origin[1]) / resolution)
    return pixel_x, pixel_y

class RobotPoseVisualizer:
    def __init__(self, map_path):
        # Set up the figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        
        # Load and display the map
        self.map_img = mpimg.imread(map_path)
        self.ax.imshow(self.map_img, cmap='gray')
        
        # Initialize robot position plot (will be updated)
        self.robot_point = self.ax.plot([], [], 'ro', label='Robot Position')[0]
        self.robot_arrow = self.ax.quiver([], [], [], [], color='r', scale=50)
        
        # Set up plot properties
        self.ax.set_title('Real-time Robot Pose Visualization')
        self.ax.set_xlabel('X (pixels)')
        self.ax.set_ylabel('Y (pixels)')
        self.ax.grid(True)
        self.ax.legend()

        # Store the map dimensions
        self.map_height = self.map_img.shape[0]

    def update(self, frame):
        """Update function for animation"""
        pose = get_robot_pose()
        if pose is None:
            return self.robot_point, self.robot_arrow

        # Convert world coordinates to pixel coordinates
        robot_x, robot_y = world_to_pixel(
            pose['x'], 
            pose['y'], 
            MAP_SETTINGS['resolution'], 
            MAP_SETTINGS['origin']
        )

        # Update robot position point
        self.robot_point.set_data([robot_x], [self.map_height - robot_y])

        # Update robot orientation arrow
        arrow_length = 20
        dx = arrow_length * np.cos(pose['theta'])
        dy = -arrow_length * np.sin(pose['theta'])  # Negative for display coordinates
        
        self.robot_arrow.remove()
        self.robot_arrow = self.ax.quiver(robot_x, self.map_height - robot_y, 
                                        dx, dy, color='r', scale=50)

        # Add text with current coordinates
        plt.title(f'Robot Pose - X: {pose["x"]:.2f}, Y: {pose["y"]:.2f}, θ: {pose["theta"]:.2f}')
        
        return self.robot_point, self.robot_arrow

def main():
    # Create visualizer
    visualizer = RobotPoseVisualizer('map.png')
    
    # Set up animation
    ani = FuncAnimation(
        visualizer.fig, 
        visualizer.update,
        interval=100,  # Update every 100ms
        blit=True
    )
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    main()