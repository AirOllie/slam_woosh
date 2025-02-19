import torch
import numpy as np
import skimage.morphology
from PIL import Image
import copy

class BirdsEyeMapper:
    """
    Creates and updates a bird's eye view map from camera depth observations.
    The map uses a grid where each cell represents map_resolution×map_resolution cm².
    """
    def __init__(self, 
                 map_size_cm=4000,           # Total map size in centimeters
                 map_resolution=5,            # Resolution in cm per pixel
                 camera_height=480,           # Height of camera image in pixels
                 camera_width=640,            # Width of camera image in pixels 
                 camera_fov=79,              # Horizontal field of view in degrees
                 max_depth=5.0,              # Maximum depth to consider in meters
                 device=None):
        """
        Initialize the mapper.
        
        Args:
            map_size_cm (int): Size of the map in centimeters
            map_resolution (int): Resolution of the map in cm per pixel
            camera_height (int): Height of depth image in pixels
            camera_width (int): Width of depth image in pixels
            camera_fov (float): Horizontal field of view in degrees
            max_depth (float): Maximum depth distance to consider in meters
            device (torch.device): Device to use for computations
        """
        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Camera parameters
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.camera_fov = camera_fov
        self.max_depth = max_depth
        
        # Initialize projection matrices
        self._init_camera_matrices()
        
        # Initialize maps
        self.map_size = self.map_size_cm // self.map_resolution
        self.full_map = torch.zeros(1, 1, self.map_size, self.map_size).float().to(self.device)
        self.fbe_free_map = copy.deepcopy(self.full_map).to(self.device)  # 0 is unknown, 1 is free
        self.full_pose = torch.zeros(3).float().to(self.device)
        
        # For collision detection
        self.collision_map = np.zeros((self.map_size, self.map_size))
        self.visited = np.zeros((self.map_size, self.map_size))
        self.selem = skimage.morphology.square(1)

    def _init_camera_matrices(self):
        """Initialize matrices for projecting depth image to 3D points."""
        hfov = float(self.camera_fov) * np.pi / 180
        vfov = 2 * np.arctan(np.tan(hfov/2) * self.camera_height / self.camera_width)
        
        # Generate camera rays
        xs = np.linspace(0, self.camera_width-1, self.camera_width)
        ys = np.linspace(0, self.camera_height-1, self.camera_height)
        xv, yv = np.meshgrid(xs, ys)
        
        # Convert to normalized device coordinates
        x_ndc = 2 * (xv - self.camera_width/2) / self.camera_width
        y_ndc = 2 * (yv - self.camera_height/2) / self.camera_height
        
        # Create camera rays
        x_cam = x_ndc * np.tan(hfov/2)
        y_cam = y_ndc * np.tan(vfov/2)
        z_cam = np.ones_like(x_cam)
        
        # Stack into projection matrix
        self.camera_matrix = torch.from_numpy(
            np.stack([x_cam, y_cam, z_cam], axis=2)
        ).float().to(self.device)

    def reset(self):
        """Reset all maps and poses."""
        self.full_map.fill_(0.)
        self.fbe_free_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:2] = self.map_size_cm / 100.0 / 2.0  # put the agent in the middle of the map
        self.collision_map.fill(0)
        self.visited.fill(0)

    def update_pose(self, gps, compass):
        """
        Update agent pose based on GPS and compass readings.
        
        Args:
            gps (numpy.ndarray): [x, y] GPS coordinates in meters relative to start position
                                x is east-west, y is north-south
            compass (float): Compass reading in radians, 0 is east, increases counter-clockwise
        """
        self.full_pose[0] = self.map_size_cm / 100.0 / 2.0 + gps[0]  # Add to center of map
        self.full_pose[1] = self.map_size_cm / 100.0 / 2.0 - gps[1]  # Subtract from center (y-axis flip)
        self.full_pose[2] = compass * 57.29577951308232  # Convert to degrees

    def update_map(self, depth_image):
        """
        Update map using depth camera input.
        
        Args:
            depth_image (numpy.ndarray): Depth image of shape (H, W) or (H, W, 1)
                                       with values in meters
        """
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]
            
        # Convert to tensor
        depth = torch.from_numpy(depth_image).float().to(self.device)
        
        # Mask out invalid depths
        depth[depth == 0] = self.max_depth
        depth = torch.clamp(depth, 0, self.max_depth)
        
        # Project depths to 3D points
        points = self.camera_matrix * depth.unsqueeze(-1)
        
        # Transform points to world coordinates
        angle = self.full_pose[2].item() * np.pi / 180
        rot_matrix = torch.tensor([
            [np.cos(angle), 0, -np.sin(angle)],
            [0, 1, 0],
            [np.sin(angle), 0, np.cos(angle)]
        ]).float().to(self.device)
        
        points = torch.matmul(points, rot_matrix.T)
        
        # Add agent position
        points[:, :, 0] += self.full_pose[0]
        points[:, :, 2] += self.full_pose[1]
        
        # Convert to map coordinates
        map_x = ((points[:, :, 0] * 100) / self.map_resolution).long()
        map_y = ((points[:, :, 2] * 100) / self.map_resolution).long()
        
        # Update maps
        valid = (map_x >= 0) & (map_x < self.map_size) & \
                (map_y >= 0) & (map_y < self.map_size)
                
        self.full_map[0, 0, map_y[valid], map_x[valid]] = 1
        self.fbe_free_map[0, 0, map_y[valid], map_x[valid]] = 1

    def get_traversible_map(self):
        """
        Get traversible map showing free space and obstacles.
        
        Returns:
            numpy.ndarray: Binary map where 1 indicates traversible space
        """
        input_pose = np.zeros(7)
        input_pose[:3] = self.full_pose.cpu().numpy()
        input_pose[1] = self.map_size_cm/100 - input_pose[1]
        input_pose[2] = -input_pose[2]
        input_pose[4] = self.full_map.shape[-2]
        input_pose[6] = self.full_map.shape[-1]
        
        # Get pose prediction and planning window
        start_x, start_y, start_o, gx1, gx2, gy1, gy2 = input_pose
        gx1, gx2, gy1, gy2 = int(gx1), int(gx2), int(gy1), int(gy2)
        
        # Get current location
        r, c = start_y, start_x
        start = [int(r*100/self.map_resolution - gy1),
                int(c*100/self.map_resolution - gx1)]
        start = self._threshold_poses(start, self.full_map.shape[-2:])
        
        # Update visited map
        self.visited[gy1:gy2, gx1:gx2][start[0]-2:start[0]+3,
                                      start[1]-2:start[1]+3] = 1

        # Create traversible map
        grid = np.rint(self.full_map.cpu().numpy()[0,0,::-1])
        traversible = skimage.morphology.binary_dilation(
            grid[gy1:gy2, gx1:gx2],
            self.selem) != True
            
        # Add boundaries and process map
        traversible = 1 - traversible
        selem = skimage.morphology.disk(4)
        traversible = skimage.morphology.binary_dilation(
            traversible, selem) != True
            
        traversible[int(start[0]-gy1)-1:int(start[0]-gy1)+2,
                   int(start[1]-gx1)-1:int(start[1]-gx1)+2] = 1
        traversible = traversible * 1.
        
        traversible[self.visited[gy1:gy2, gx1:gx2][gy1:gy2, gx1:gx2] == 1] = 1
        traversible[self.collision_map[gy1:gy2, gx1:gx2][gy1:gy2, gx1:gx2] == 1] = 0
        
        return self._add_boundary(traversible)

    def update_collision(self, last_action, last_pose, collision_threshold=0.08):
        """
        Update collision map based on failed forward action.
        
        Args:
            last_action (int): Last action taken (1 for forward)
            last_pose (torch.Tensor): Previous pose tensor [x, y, angle]
                                    with x,y in meters and angle in degrees
            collision_threshold (float): Distance threshold for collision detection in meters
        """
        if last_action != 1:
            return
            
        x1, y1, t1 = last_pose.cpu().numpy()
        x2, y2, t2 = self.full_pose.cpu()
        y1 = self.map_size_cm/100 - y1
        y2 = self.map_size_cm/100 - y2
        t1 = -t1
        t2 = -t2
        
        dist = self._get_l2_distance(x1, x2, y1, y2)
        
        if dist < collision_threshold:
            width = 3
            length = 5
            buf = 4
            
            for i in range(length):
                for j in range(width):
                    wx = x1 + 0.05*((i+buf) * np.cos(np.deg2rad(t1)) + 
                                   (j-width//2) * np.sin(np.deg2rad(t1)))
                    wy = y1 + 0.05*((i+buf) * np.sin(np.deg2rad(t1)) - 
                                   (j-width//2) * np.cos(np.deg2rad(t1)))
                    r, c = wy, wx
                    r, c = int(round(r*100/self.map_resolution)), \
                           int(round(c*100/self.map_resolution))
                    [r, c] = self._threshold_poses([r, c],
                                self.collision_map.shape)
                    self.collision_map[r,c] = 1

    @staticmethod
    def _threshold_poses(poses, shape):
        """Threshold poses to within map bounds."""
        for i in range(len(poses)):
            poses[i] = min(max(poses[i], 0), shape[i]-1)
        return poses

    @staticmethod
    def _get_l2_distance(x1, x2, y1, y2):
        """Calculate L2 distance between points."""
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    @staticmethod
    def _add_boundary(mat, value=1):
        """Add boundary to matrix."""
        h, w = mat.shape
        new_mat = np.zeros((h+2, w+2)) + value
        new_mat[1:h+1, 1:w+1] = mat
        return new_mat

    def visualize_map(self, show_agent=True):
        """
        Create visualization of the map.
        
        Args:
            show_agent (bool): Whether to show agent position
            
        Returns:
            PIL.Image: Visualization of the map
        """
        # Create RGB map
        map_vis = torch.zeros((3, self.map_size, self.map_size))
        
        # Add free space (grey)
        free_space = self.fbe_free_map.cpu()[0,0] > 0.5
        map_vis[:, free_space] = torch.tensor([0.8, 0.8, 0.8]).reshape(3, 1)
        
        # Add obstacles (dark grey)
        obstacles = self.full_map.cpu()[0,0] > 0.5
        map_vis[:, obstacles] = torch.tensor([0.4, 0.4, 0.4]).reshape(3, 1)
        
        # Add agent position (red)
        if show_agent:
            agent_y = int((self.map_size_cm/100-self.full_pose[1].cpu())*100/self.map_resolution)
            agent_x = int(self.full_pose[0].cpu()*100/self.map_resolution)
            agent_size = 2
            map_vis[:, agent_y-agent_size:agent_y+agent_size, 
                      agent_x-agent_size:agent_x+agent_size] = 0
            map_vis[0, agent_y-agent_size:agent_y+agent_size,
                      agent_x-agent_size:agent_x+agent_size] = 1
                      
        # Convert to PIL image
        map_vis = (map_vis.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(map_vis)