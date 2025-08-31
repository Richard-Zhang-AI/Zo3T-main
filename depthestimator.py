import os
import sys
import torch
import numpy as np
from PIL import Image, ExifTags
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add DepthPro path
sys.path.append('./ml-depth-pro-main/src')

class DepthProEstimator:

    
    def __init__(self, checkpoint_path="./ml-depth-pro-main/src/checkpoints/depth_pro.pt", device=None):
        """
        Initialize the depth estimator.
        
        Args:
            checkpoint_path (str): Path to the DepthPro model weights.
            device (torch.device, optional): Computation device. Defaults to auto-detection (CUDA or CPU).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        
        print(f"🔍 Loading DepthPro model from: {checkpoint_path}")
        
        try:
            # Import DepthPro modules
            from depth_pro.depth_pro import create_model_and_transforms, DepthProConfig
            # Check if the model file exists
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"DepthPro model file not found: {checkpoint_path}")
            
            # Create config, specifying the correct checkpoint path
            config = DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384", 
                checkpoint_uri=checkpoint_path,
                decoder_features=256,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            )
            
            # Create model and transforms
            self.model, self.transform = create_model_and_transforms(
                config=config,
                device=self.device,
                precision=torch.half if self.device.type == "cuda" else torch.float32
            )
            
            print("✅ DepthPro model loaded successfully!")
            
        except Exception as e:
            print(f"❌ DepthPro model loading failed: {e}")
            print("🔄 Falling back to simple depth estimation...")
            # If DepthPro fails to load, set a flag
            self.model = None
            self.transform = None
    
    def correct_orientation(self, image):
        """
        Automatically rotate the image based on the EXIF orientation tag.
        
        Args:
            image (Image.Image): A PIL Image object.
            
        Returns:
            Image.Image: The orientation-corrected image.
        """
        try:
            exif = image._getexif()
            if exif:
                orientation = exif.get(274)  # 274 corresponds to the Orientation tag
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
        except (AttributeError, KeyError):
            # The image might not have EXIF data, or the Orientation tag is missing.
            pass
        return image
    
    def estimate_depth(self, image):
        """
        Estimate the depth of a single image.
        
        Args:
            image (Image.Image or np.ndarray): Input image.
            
        Returns:
            np.ndarray: Depth map with shape (H, W) and values normalized to the range [0, 1].
        """
        # If the DepthPro model failed to load, use simple depth estimation as a fallback.
        if self.model is None or self.transform is None:
            print("⚠️ DepthPro is not available, using simple depth estimation.")
            return self._simple_depth_estimation(image)
        
        # Ensure the input is a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Correct image orientation based on EXIF data
        image = self.correct_orientation(image)
        
        try:
            # Perform depth estimation using DepthPro
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prediction = self.model.infer(image_tensor)
                
            # Get the depth map from the prediction
            if isinstance(prediction, dict) and 'depth' in prediction:
                depth = prediction['depth']
            else:
                depth = prediction
                
            # Convert to a numpy array
            if isinstance(depth, torch.Tensor):
                depth_np = depth.squeeze().detach().cpu().numpy()
            else:
                depth_np = np.array(depth)
            
            # Normalize to [0, 1]
            if depth_np.max() > depth_np.min():
                depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
            else:
                depth_normalized = np.ones_like(depth_np) * 0.5
            
            return depth_normalized
            
        except Exception as e:
            print(f"⚠️ DepthPro inference failed: {e}")
            print("🔄 Falling back to simple depth estimation.")
            return self._simple_depth_estimation(image)
    
    def _simple_depth_estimation(self, image):
        """
        A simple depth estimation method as a fallback.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to grayscale and perform simple processing
        gray = image.convert('L')
        depth_array = np.array(gray, dtype=np.float32) / 255.0
        
        # Simple depth simulation: brighter areas are considered closer (inverted depth)
        depth_inverted = 1.0 - depth_array
        
        return depth_inverted
    
    def create_point_cloud(self, image, depth_map, fx=None, fy=None, cx=None, cy=None, depth_scale=1.0, downsample=1):
        """
        Creates a 3D point cloud from an image and its depth map.

        Args:
            image (Image.Image or np.ndarray): The original color image.
            depth_map (np.ndarray): The depth map corresponding to the image.
            fx (float, optional): Focal length in x. Defaults to a reasonable guess.
            fy (float, optional): Focal length in y. Defaults to a reasonable guess.
            cx (float, optional): Principal point x. Defaults to image center.
            cy (float, optional): Principal point y. Defaults to image center.
            depth_scale (float, optional): Scale factor for the depth map. Defaults to 1.0.
            downsample (int, optional): Factor to downsample the point cloud for performance. Defaults to 1 (no downsampling).

        Returns:
            dict: A dictionary containing 'points', 'colors', and 'intrinsics'.
        """
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
        
        # Get image dimensions
        h, w = depth_map.shape
        
        # If camera intrinsics are not provided, use default values
        if fx is None:
            fx = w * 0.8  # Assuming a field of view of about 60 degrees
        if fy is None:
            fy = h * 0.8
        if cx is None:
            cx = w / 2.0
        if cy is None:
            cy = h / 2.0
        
        # Create a grid of pixel coordinates
        u, v = np.meshgrid(np.arange(0, w, downsample), np.arange(0, h, downsample))
        
        # Downsample the depth map and color map
        if downsample > 1:
            depth_sampled = depth_map[::downsample, ::downsample]
            color_sampled = image_array[::downsample, ::downsample]
        else:
            depth_sampled = depth_map
            color_sampled = image_array
        
        # Convert depth values to metric distances (simple scaling)
        z = depth_sampled * depth_scale + 0.1
        
        # Convert to 3D coordinates (back-projection)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        # Flatten the coordinate arrays
        points_3d = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        
        # Process color information
        if len(color_sampled.shape) == 3:
            colors = color_sampled.reshape(-1, 3) / 255.0  # Normalize to [0, 1]
        else:
            # If it's a grayscale image, convert to RGB
            gray_colors = color_sampled.flatten() / 255.0
            colors = np.stack([gray_colors, gray_colors, gray_colors], axis=1)
        
        # Filter out invalid points (where depth is close to zero or infinite)
        valid_mask = (z.flatten() > 0.1) & (z.flatten() < 100) & np.isfinite(z.flatten())
        points_3d = points_3d[valid_mask]
        colors = colors[valid_mask]
        
        return {
            'points': points_3d,
            'colors': colors,
            'intrinsics': {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}
        }
    
    def visualize_point_cloud(self, point_cloud_data, save_path=None, title="3D Point Cloud", 
                            point_size=0.5, elevation=20, azimuth=45):
        """
        Visualize a 3D point cloud.
        
        Args:
            point_cloud_data (dict): Dictionary returned by create_point_cloud.
            save_path (str, optional): Save path; if None, display the plot. Defaults to None.
            title (str, optional): Plot title. Defaults to "3D Point Cloud".
            point_size (float, optional): Size of the points. Defaults to 0.5.
            elevation (float, optional): Viewpoint elevation angle. Defaults to 20.
            azimuth (float, optional): Viewpoint azimuth angle. Defaults to 45.
        """
        points = point_cloud_data['points']
        colors = point_cloud_data['colors']
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the point cloud
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=colors, s=point_size, alpha=0.6, edgecolors='none')
        
        # Set axis labels
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(title)
        
        # Set the viewpoint
        ax.view_init(elev=elevation, azim=azimuth)
        
        # Set axis aspect ratio to be equal
        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                             points[:, 1].max() - points[:, 1].min(),
                             points[:, 2].max() - points[:, 2].min()]).max() / 2.0
        
        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"🎯 Point cloud visualization saved to: {save_path}")
        else:
            plt.show()
    
    def save_point_cloud_ply(self, point_cloud_data, save_path):
        """
        Save the point cloud in PLY format.
        
        Args:
            point_cloud_data (dict): Dictionary returned by create_point_cloud.
            save_path (str): Save path for the .ply file.
        """
        points = point_cloud_data['points']
        colors = point_cloud_data['colors']
        
        # Ensure color values are in the 0-255 range
        colors_255 = (colors * 255).astype(np.uint8)
        
        # Create the PLY file header
        header = f"""ply
format ascii 1.0
element vertex {len(points)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
        
        # Combine points and colors for efficient writing
        vertices = np.hstack([points, colors_255])

        # Write to the PLY file
        with open(save_path, 'w') as f:
            f.write(header)
            np.savetxt(f, vertices, fmt='%.6f %.6f %.6f %d %d %d')
        
        print(f"💾 Point cloud saved in PLY format to: {save_path}")
    
    def save_point_cloud_npz(self, point_cloud_data, save_path):
        """
        Save the point cloud in NPZ format (numpy compressed).
        
        Args:
            point_cloud_data (dict): Dictionary returned by create_point_cloud.
            save_path (str): Save path for the .npz file.
        """
        np.savez_compressed(save_path, 
                           points=point_cloud_data['points'],
                           colors=point_cloud_data['colors'],
                           intrinsics=point_cloud_data['intrinsics'])
        print(f"💾 Point cloud saved in NPZ format to: {save_path}")
    
    def process_image_to_point_cloud(self, image_path, output_dir=None, visualize=True, 
                                   save_ply=True, save_npz=True, downsample=2):
        """
        Complete pipeline for processing an image to a point cloud.
        
        Args:
            image_path (str): Input image path.
            output_dir (str, optional): Output directory; if None, uses the image's directory. Defaults to None.
            visualize (bool, optional): Whether to generate visualizations. Defaults to True.
            save_ply (bool, optional): Whether to save in PLY format. Defaults to True.
            save_npz (bool, optional): Whether to save in NPZ format. Defaults to True.
            downsample (int, optional): Downsampling factor for point cloud generation. Defaults to 2.
            
        Returns:
            dict: A dictionary containing the point cloud data.
        """
        # Determine the output directory
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the base name of the file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        print(f"🖼️ Processing image: {image_path}")
        
        # Load the image
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"❌ Error: Image file not found at {image_path}")
            return None
        
        # Estimate depth
        print("🔍 Estimating depth...")
        depth_map = self.estimate_depth(image)
        
        # Create the point cloud
        print("🎯 Creating 3D point cloud...")
        point_cloud_data = self.create_point_cloud(image, depth_map, downsample=downsample)
        
        if point_cloud_data and len(point_cloud_data['points']) > 0:
            print(f"📊 Point cloud stats: {len(point_cloud_data['points'])} points")
        else:
            print("⚠️ Warning: No valid points were generated for the point cloud.")
            return None

        # Visualize the depth map
        if visualize:
            depth_vis_path = os.path.join(output_dir, f"{base_name}_depth.png")
            self.visualize_depth_map(image, depth_map, depth_vis_path)
        
        # Visualize the point cloud
        if visualize:
            pc_vis_path = os.path.join(output_dir, f"{base_name}_pointcloud.png")
            self.visualize_point_cloud(point_cloud_data, pc_vis_path, 
                                     title=f"Point Cloud - {base_name}")
        
        # Save the point cloud files
        if save_ply:
            ply_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
            self.save_point_cloud_ply(point_cloud_data, ply_path)
        
        if save_npz:
            npz_path = os.path.join(output_dir, f"{base_name}_pointcloud.npz")
            self.save_point_cloud_npz(point_cloud_data, npz_path)
        
        print("✅ Point cloud processing complete!")
        return point_cloud_data

    def estimate_depth_batch(self, images, show_progress=True):
        """
        Estimate depth for a batch of images.
        
        Args:
            images (list): List of images, where each element can be a PIL Image or a numpy array.
            show_progress (bool, optional): Whether to display a progress bar. Defaults to True.
            
        Returns:
            list: A list of depth maps.
        """
        depths = []
        
        iterator = tqdm(images, desc="Estimating depths", unit="img") if show_progress else images
        
        for image in iterator:
            depth = self.estimate_depth(image)
            depths.append(depth)
        
        return depths
    
    def get_depth_at_points(self, image, points):
        """
        Get the depth values at specified pixel coordinates in an image.
        
        Args:
            image (Image.Image or np.ndarray): Input image.
            points (np.ndarray): Point coordinates, with shape (N, 2) for [x, y] pairs.
            
        Returns:
            np.ndarray: An array of depth values corresponding to the input points.
        """
        depth_map = self.estimate_depth(image)
        h, w = depth_map.shape
        
        # Ensure points are integers and within bounds
        points_int = np.round(points).astype(int)
        points_int[:, 0] = np.clip(points_int[:, 0], 0, w - 1)
        points_int[:, 1] = np.clip(points_int[:, 1], 0, h - 1)
        
        # Extract depth values using advanced indexing
        depth_values = depth_map[points_int[:, 1], points_int[:, 0]]
        
        return depth_values
    
    def visualize_depth_map(self, image, depth_map, save_path=None):
        """
        Visualize the original image and its corresponding depth map side-by-side.
        
        Args:
            image (Image.Image or np.ndarray): The original image.
            depth_map (np.ndarray): The depth map.
            save_path (str, optional): Save path for the visualization; if None, display the plot. Defaults to None.
        """
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display the original image
        ax1.imshow(image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Display the depth map
        im = ax2.imshow(depth_map, cmap='viridis')
        ax2.set_title("Depth Map")
        ax2.axis('off')
        
        # Add a color bar
        fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Normalized Depth")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"💾 Depth visualization saved to: {save_path}")
        else:
            plt.show()
    
    def __call__(self, image):
        """
        Convenience call interface to estimate depth.
        
        Args:
            image (Image.Image or np.ndarray): Input image.
            
        Returns:
            np.ndarray: The estimated depth map.
        """
        return self.estimate_depth(image)