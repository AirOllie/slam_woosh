import pyrealsense2 as rs
import numpy as np
import requests
import math
from typing import Optional, Tuple, Dict
import torch
from bev_mapper import BirdsEyeMapper
import json

class RealSenseMapper:
    """
    Integrates RealSense depth camera with robot pose for mapping.
    """
    def __init__(self, 
                 map_size_cm: int = 4000,
                 map_resolution: int = 5,
                 api_url: str = None,
                 api_headers: Dict = None,
                 api_payload: Dict = None):
        """
        Initialize RealSense camera and mapping system.
        
        Args:
            map_size_cm: Size of map in centimeters
            map_resolution: Resolution in cm per pixel
            api_url: URL for robot pose API
            api_headers: Headers for API request
            api_payload: Payload for API request
        """
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        # Get camera intrinsics
        pipeline_profile = self.config.resolve(self.pipeline)
        depth_profile = pipeline_profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        
        # Initialize mapper
        self.mapper = BirdsEyeMapper(
            map_size_cm=map_size_cm,
            map_resolution=map_resolution,
            camera_height=480,
            camera_width=640,
            camera_fov=math.degrees(2 * math.atan(320 / self.intrinsics.fx)),
            max_depth=5.0
        )
        
        # API configuration
        self.api_url = api_url
        self.api_headers = api_headers
        self.api_payload = api_payload
        
        # Start RealSense pipeline
        self.pipeline.start(self.config)
        
    def get_robot_pose(self) -> Optional[Dict]:
        """
        Get robot pose from API.
        
        Returns:
            Dictionary containing pose data or None if request fails
        """
        try:
            response = requests.post(
                self.api_url, 
                headers=self.api_headers, 
                data=self.api_payload
            )
            data = response.json()
            if data['ok']:
                return data['body']['pose']
            return None
        except Exception as e:
            print(f"Error getting pose: {e}")
            return None
            
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """
        Get depth frame from RealSense camera.
        
        Returns:
            Depth image as numpy array in meters, or None if capture fails
        """
        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
                
            # Convert depth to meters
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_scale = self.pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()
            depth_meters = depth_image * depth_scale
            
            return depth_meters
            
        except Exception as e:
            print(f"Error capturing depth: {e}")
            return None
            
    def process_frame(self) -> Tuple[Optional[np.ndarray], Optional[torch.Tensor]]:
        """
        Process one frame from camera and robot pose.
        
        Returns:
            Tuple of (traversible_map, current_pose) or (None, None) if processing fails
        """
        # Get depth frame
        depth_frame = self.get_depth_frame()
        if depth_frame is None:
            return None, None
            
        # Get robot pose
        pose_data = self.get_robot_pose()
        if pose_data is None:
            return None, None
            
        # Extract pose components
        try:
            x = pose_data['x']
            y = pose_data['y']
            heading = pose_data['theta']  # Assuming heading in radians
            
            # Store previous pose for collision detection
            previous_pose = self.mapper.full_pose.clone()
            
            # Update mapper
            self.mapper.update_pose(
                gps=np.array([x, y]),
                compass=heading
            )
            self.mapper.update_map(depth_frame)
            
            # Get traversible map
            traversible = self.mapper.get_traversible_map()
            
            return traversible, previous_pose
            
        except KeyError as e:
            print(f"Missing pose component: {e}")
            return None, None
            
    def close(self):
        """Clean up resources."""
        self.pipeline.stop()

def main():
    """Example usage of RealSenseMapper."""
    
    # API configuration
    API_CONFIG = {
        'url': 'http://192.168.1.87:5480/woosh/robot/PoseSpeed',
        'headers': {'Content-Type': 'application/json'},
        'payload': json.dumps({"robotId": 30001})
    }
    
    # Initialize mapper
    mapper = RealSenseMapper(
        map_size_cm=4000,
        map_resolution=5,
        api_url=API_CONFIG['url'],
        api_headers=API_CONFIG['headers'],
        api_payload=API_CONFIG['payload']
    )
    
    try:
        while True:
            # Process frame
            traversible, previous_pose = mapper.process_frame()
            
            if traversible is not None:
                # Visualize map
                map_image = mapper.mapper.visualize_map(show_agent=True)
                map_image.save('current_map.png')
                
                # Your navigation code here...
                pass
                
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        mapper.close()

if __name__ == "__main__":
    main()