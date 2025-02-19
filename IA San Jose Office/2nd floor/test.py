import pyrealsense2 as rs
import numpy as np
from scipy.spatial.transform import Rotation
import cv2

class RealSenseLocalizer:
    def __init__(self, occupancy_map_path, map_resolution):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.pose)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Load occupancy map
        self.occupancy_map = cv2.imread(occupancy_map_path, cv2.IMREAD_GRAYSCALE)
        self.map_resolution = map_resolution  # meters per pixel
        
        # Initialize particle filter parameters
        self.num_particles = 1000
        self.particles = None
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def initialize_particles(self, initial_pose=None):
        if initial_pose is not None:
            # Initialize around known position
            x, y, theta = initial_pose
            self.particles = np.zeros((self.num_particles, 3))
            self.particles[:, 0] = x + np.random.normal(0, 0.1, self.num_particles)
            self.particles[:, 1] = y + np.random.normal(0, 0.1, self.num_particles)
            self.particles[:, 2] = theta + np.random.normal(0, 0.1, self.num_particles)
        else:
            # Random initialization across free space
            free_space = np.where(self.occupancy_map == 255)
            random_indices = np.random.choice(len(free_space[0]), self.num_particles)
            self.particles = np.zeros((self.num_particles, 3))
            self.particles[:, 0] = free_space[1][random_indices] * self.map_resolution
            self.particles[:, 1] = free_space[0][random_indices] * self.map_resolution
            self.particles[:, 2] = np.random.uniform(0, 2*np.pi, self.num_particles)

    def update(self):
        # Get RealSense frames
        frames = self.pipeline.wait_for_frames()
        pose_frame = frames.get_pose_frame()
        depth_frame = frames.get_depth_frame()
        
        if pose_frame and depth_frame:
            # Get pose data
            pose_data = pose_frame.get_pose_data()
            translation = np.array([pose_data.translation.x, pose_data.translation.y, pose_data.translation.z])
            rotation = Rotation.from_quat([
                pose_data.rotation.x, pose_data.rotation.y,
                pose_data.rotation.z, pose_data.rotation.w
            ])
            
            # Motion update
            delta_translation = translation - self.previous_translation
            delta_rotation = rotation * self.previous_rotation.inv()
            
            # Update particles based on motion
            self.motion_update(delta_translation, delta_rotation)
            
            # Measurement update using depth data
            depth_image = np.asanyarray(depth_frame.get_data())
            self.measurement_update(depth_image)
            
            # Store current pose for next update
            self.previous_translation = translation
            self.previous_rotation = rotation
            
            # Resample particles if needed
            self.resample_if_needed()
            
            return self.get_estimated_pose()
    
    def motion_update(self, delta_translation, delta_rotation):
        # Add noise to motion model
        translation_noise = np.random.normal(0, 0.01, (self.num_particles, 2))
        rotation_noise = np.random.normal(0, 0.01, self.num_particles)
        
        # Update particle positions
        self.particles[:, :2] += delta_translation[:2] + translation_noise
        self.particles[:, 2] += delta_rotation.as_euler('xyz')[2] + rotation_noise
        
    def measurement_update(self, depth_image):
        for i in range(self.num_particles):
            x, y, theta = self.particles[i]
            
            # Convert particle position to map coordinates
            map_x = int(x / self.map_resolution)
            map_y = int(y / self.map_resolution)
            
            # Skip particles outside map
            if not (0 <= map_x < self.occupancy_map.shape[1] and 
                   0 <= map_y < self.occupancy_map.shape[0]):
                self.weights[i] = 0
                continue
            
            # Compare expected depth measurements with actual measurements
            expected_depth = self.raytrace(map_x, map_y, theta)
            actual_depth = self.get_depth_measurements(depth_image)
            
            # Update particle weight based on measurement likelihood
            self.weights[i] *= self.measurement_likelihood(expected_depth, actual_depth)
            
        # Normalize weights
        self.weights /= np.sum(self.weights)
    
    def raytrace(self, x, y, theta):
        # Implement raytracing in occupancy map to get expected measurements
        # This is a simplified version - you might want to implement more sophisticated raytracing
        rays = np.linspace(theta - np.pi/4, theta + np.pi/4, 10)
        depths = []
        
        for ray in rays:
            ray_x = x
            ray_y = y
            depth = 0
            
            while (0 <= ray_x < self.occupancy_map.shape[1] and 
                   0 <= ray_y < self.occupancy_map.shape[0] and 
                   depth < 5.0):  # Maximum depth in meters
                if self.occupancy_map[int(ray_y), int(ray_x)] < 127:  # Hit obstacle
                    break
                    
                ray_x += np.cos(ray) * self.map_resolution
                ray_y += np.sin(ray) * self.map_resolution
                depth += self.map_resolution
                
            depths.append(depth)
            
        return np.array(depths)
    
    def get_depth_measurements(self, depth_image):
        # Extract relevant depth measurements from RealSense depth image
        # This is a simplified version - you might want to implement more sophisticated depth processing
        center_y = depth_image.shape[0] // 2
        rays = np.linspace(0, depth_image.shape[1]-1, 10).astype(int)
        depths = depth_image[center_y, rays] / 1000.0  # Convert to meters
        return depths
    
    def measurement_likelihood(self, expected_depth, actual_depth):
        # Calculate likelihood of measurements using a simple Gaussian model
        sigma = 0.1  # Standard deviation of measurement noise
        diff = np.abs(expected_depth - actual_depth)
        likelihood = np.exp(-diff**2 / (2 * sigma**2))
        return np.mean(likelihood)
    
    def resample_if_needed(self):
        # Resample particles if effective sample size is too low
        effective_sample_size = 1.0 / np.sum(self.weights**2)
        
        if effective_sample_size < self.num_particles / 2:
            indices = np.random.choice(
                self.num_particles, 
                self.num_particles, 
                p=self.weights
            )
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def get_estimated_pose(self):
        # Return weighted average of particle poses
        mean_x = np.average(self.particles[:, 0], weights=self.weights)
        mean_y = np.average(self.particles[:, 1], weights=self.weights)
        mean_theta = np.average(self.particles[:, 2], weights=self.weights)
        
        return mean_x, mean_y, mean_theta
    
    def start(self):
        self.pipeline.start(self.config)
        frames = self.pipeline.wait_for_frames()
        pose_frame = frames.get_pose_frame()
        
        if pose_frame:
            # Initialize previous pose data
            pose_data = pose_frame.get_pose_data()
            self.previous_translation = np.array([
                pose_data.translation.x,
                pose_data.translation.y,
                pose_data.translation.z
            ])
            self.previous_rotation = Rotation.from_quat([
                pose_data.rotation.x,
                pose_data.rotation.y,
                pose_data.rotation.z,
                pose_data.rotation.w
            ])
    
    def stop(self):
        self.pipeline.stop()

if __name__ == "__main__":
    # Initialize localizer with your occupancy map
    localizer = RealSenseLocalizer('map.png', map_resolution=0.03)

    # Start the RealSense pipeline
    localizer.start()

    # Initialize particles (optionally with initial pose if known)
    localizer.initialize_particles()

    # Main loop
    try:
        while True:
            # Update localization
            x, y, theta = localizer.update()
            print(f"Estimated pose: x={x:.2f}m, y={y:.2f}m, theta={np.degrees(theta):.2f}°")
            
    except KeyboardInterrupt:
        localizer.stop()