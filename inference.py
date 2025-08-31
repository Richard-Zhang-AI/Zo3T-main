import os
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableVideoDiffusionPipeline, AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel, EulerDiscreteScheduler
from diffusers.utils import export_to_video

from src.utils import *
from src.pipeline import SGI2VPipe
from src.model import MyUNet

import argparse

def read_condition(input_dir, config):
    """
    Read input condition.
    input_dir/:
        ./img.png (first frame image)
        ./traj.npy (ndarray of shape [N, (2+F), 2], where first [N, 2, 2] specifies top-left/bottom-right coordinates of bounding boxes (i.e., [[w1, h1], [w2, h2]]), while the rest of [N, F, 2] specifies trajectories of center coordinates of each bounding box across frames (in order of (w, h))

    Note: N is the number of bounding boxes placed on the first frame, F is the number of frames.
    """
    
    image_path = os.path.join(input_dir, "img.png")
    
    #Load image
    image = Image.open(image_path).convert('RGB')
    original_width, original_height = image.size
    image = image.resize((config.width, config.height))

    #load bounding_box, center_traj
    ret = np.load(os.path.join(input_dir, "traj.npy")).astype(np.float32) #N x (2+F) x 2
    ret[:,:,0] = ret[:,:,0]*config.width/original_width 
    ret[:,:,1] = ret[:,:,1]*config.height/original_height
    bounding_box = ret[:, :2].reshape(-1, 4) # N x 4
    center_traj = ret[:, 2:] #N x F x 2
    
    # Preprocess trajectory
    original_frames = center_traj.shape[1]  # Original number of frames in trajectory
    target_frames = config.num_frames       # Target number of frames
    
    print(f"🔄 Interpolating trajectory: from {original_frames} to {target_frames} frames")
    
    trajectory_points = [] # N x frames x 4 (i.e., h1, w1, h2, w2) : trajectory of bounding boxes overparameterized by top-left/bottom-right coordinates for each frame for convenience
    for j, trajectory in enumerate(center_traj):
        trajectory = trajectory*1.2 # Scale up trajectory
        # Interpolate/downsample the trajectory to match the target number of frames
        if target_frames != original_frames:
            # Create interpolation indices
            original_indices = np.linspace(0, original_frames - 1, original_frames)
            target_indices = np.linspace(0, original_frames - 1, target_frames)
            
            # Interpolate x and y coordinates separately
            interp_x = np.interp(target_indices, original_indices, trajectory[:, 0])
            interp_y = np.interp(target_indices, original_indices, trajectory[:, 1])
            
            # Recombine the interpolated trajectory
            interpolated_trajectory = np.column_stack([interp_x, interp_y])
        else:
            interpolated_trajectory = trajectory
        
        #For normal use
        box_traj = [] # frames x 4
        for i in range(target_frames):
            d = interpolated_trajectory[i] - interpolated_trajectory[0]
            dx, dy = d[0], d[1]
            box_traj.append(np.array([bounding_box[j][1] + dy, bounding_box[j][0] + dx, bounding_box[j][3] + dy, bounding_box[j][2] + dx], dtype=np.float32))
        trajectory_points.append(box_traj)
    return image, trajectory_points

#Approx. 4 minutes on A100 with default config
def run(pipe, config, image, trajectory_points, depth_map=None):
    pipe.unet.num_inference_steps = config.num_inference_steps
    pipe.unet.optimize_zero_initialize_param = True
    height, width = config.height, config.width
    motion_bucket_id = 127
    fps = 7
    num_frames = config.num_frames
    seed = config.seed
    pipe.unet.heatmap_sigma = config.heatmap_sigma
    pipe.unet.latent_fft_post_merge = config.latent_fft_post_merge
    pipe.unet.latent_fft_ratio = config.fft_ratio #range : 0.0 - 1.0
    pipe.unet.optimize_latent_iter = config.optimize_latent_iter
    pipe.unet.optimize_latent_lr = config.optimize_latent_lr
    pipe.unet.optimize_latent_time = config.optimize_latent_time
    pipe.unet.record_layer_sublayer =  config.record_layer_sublayer
    # LoRA configuration
    pipe.unet.lora_rank = config.lora_rank
    pipe.unet.lora_alpha = config.lora_alpha
    pipe.unet.enable_lora = config.enable_lora
    
    # 🎯 Control field vector optimization configuration
    pipe.unet.enable_control_force_optimization = config.enable_control_force_optimization
    generator = torch.manual_seed(seed)
    frames = pipe(image, trajectory_points, height=height, width=width, num_frames = num_frames, decode_chunk_size=8, generator=generator, fps=fps, motion_bucket_id=motion_bucket_id, noise_aug_strength=0.02, depth_map=depth_map).frames[0]
    return frames

def main(config, input_dir, output_dir):

    #Path check
    assert(os.path.exists(args.input_dir))
    os.makedirs(args.output_dir, exist_ok=True)
    
    #Read input condition
    print("Reading input condition..")
    image, trajectory_points = read_condition(input_dir, config)
    v_id = input_dir.split("/")[-1]
    # Create folder to save results
    os.makedirs(os.path.join(output_dir, v_id), exist_ok=True)
    
    # 🎯 Using DepthPro for depth estimation to generate 3D enhanced trajectories
    depth_map = None
    if config.enable_depth_scaling:
        print("🔍 Estimating depth using DepthPro...")
        try:
            from depthestimator import DepthProEstimator
            depth_estimator = DepthProEstimator()
            depth_map = depth_estimator.estimate_depth(image)  # (H, W)
            
            # Save the depth map here (depth_map is in np array format)
            plt.imsave(os.path.join(output_dir, v_id, "depth_map.png"), depth_map, cmap='plasma')
            print("✅ DepthPro depth estimation complete.")
            
            # Release depth estimator memory
            del depth_estimator
            torch.cuda.empty_cache()
            print("🗑️ Depth estimator memory released.")
            
        except ImportError:
            print("⚠️ DepthPro is unavailable, skipping 3D mapping feature.")
        except Exception as e:
            print(f"⚠️ Depth estimation failed: {e}, skipping 3D mapping feature.")
    
    #Visualize
    visualize_control(image, trajectory_points=trajectory_points, save_path = os.path.join(output_dir, v_id, "condition_vis.png"))
    
    #Load pre-trained image-to-video diffusion models
    print("Loading Stable Video Diffusion from local path..")
    svd_dir = "./stable-video-diffusion-img2vid"
        
    feature_extractor = CLIPImageProcessor.from_pretrained(svd_dir, subfolder="feature_extractor", torch_dtype=torch.float16, variant="fp16")
    vae = AutoencoderKLTemporalDecoder.from_pretrained(svd_dir, subfolder="vae", torch_dtype=torch.float16, variant="fp16").to("cuda")
    requires_grad(vae, False)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(svd_dir, subfolder="image_encoder", torch_dtype=torch.float16, variant="fp16").to("cuda")
    requires_grad(image_encoder, False)
    unet = MyUNet.from_pretrained(svd_dir, subfolder="unet", torch_dtype=torch.float16, variant="fp16").to("cuda")
    requires_grad(unet, False)
    scheduler = EulerDiscreteScheduler.from_pretrained(svd_dir, subfolder="scheduler", torch_dtype=torch.float16, variant="fp16")
            
    unet.inject() #inject module
    
    print("Stable Video Diffusion loaded!")
        
    #Set up pipeline
    pipe = SGI2VPipe(vae, image_encoder, unet, scheduler, feature_extractor)
    pipe = pipe.to(device="cuda")

    #Generate video
    frames = run(pipe, config, image, trajectory_points, depth_map)
    
    #Save video
    export_to_video(frames, os.path.join(output_dir, v_id, "result.mp4"), fps=7)
    print(f"🎉 Video generation complete and saved to: {os.path.join(output_dir, v_id, 'result.mp4')}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="")
    
    parser.add_argument('--input_dir', type=str, required=True, help='Path directory for input conditions.')
    parser.add_argument('--output_dir', type=str, required=True, help='Saving path directory for generated videos.')
    
    args = parser.parse_args()

    #Set up config
    class Config:
        """
        Hyperparameters
        """
        seed = 1206
        height, width = 576, 1024 #resolution of generated video
        num_frames = 14
        num_inference_steps = 50 #total number of inference steps
        optimize_latent_time = list(range(30,46)) #set of timesteps to perform optimization
        optimize_latent_iter = 5 #number of optimization iterations to perform for each timestep
        optimize_latent_lr = 0.25 #learning rate for optimization
        record_layer_sublayer = [(2, 1), (2, 2)] #extract feature maps from 1st and 2nd self-attention (note: 0-indexed base) located at 2nd resolution-level of upsampling layer
        heatmap_sigma = 0.5 #standard deviation of gaussian heatmap
        fft_ratio = 0.6 #fft mix ratio
        latent_fft_post_merge = True #fft-based post-processing is enabled iff True
        
        # LoRA configuration parameters
        enable_lora = True  # Enable LoRA training during optimization
        lora_rank = 4       # Rank for LoRA decomposition (higher = more capacity, lower = more efficient)
        lora_alpha = 1.0    # LoRA scaling factor (controls adaptation strength)
        
        # 🎯 Depth scaling enhancement feature
        enable_depth_scaling = True  # Enable depth-based trajectory scaling enhancement
        
        # 🎯 Control field vector optimization feature
        enable_control_force_optimization = True  # Enable control field vector optimization (activated at the same steps as latent optimization)

    #Run
    main(config = Config(), input_dir = args.input_dir, output_dir = args.output_dir)