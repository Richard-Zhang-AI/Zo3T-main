import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion import StableVideoDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _append_dims
from src.utils import *
from src.utils import butterworth_low_pass_filter


"""
LoRA Implementation for UNet Upsampling Layers
"""

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for fine-tuning specific modules
    """
    def __init__(self, original_module, rank=4, alpha=1.0):
        super().__init__()
        self.original_module = original_module
        self.rank = rank
        self.alpha = alpha
        
        # Get input and output dimensions
        if hasattr(original_module, 'weight'):
            out_features, in_features = original_module.weight.shape
        else:
            raise ValueError("Original module must have weight attribute")
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = self.alpha / self.rank
        
        # Flag to enable/disable LoRA - start disabled
        self.lora_enabled = False
        
    def forward(self, x):
        # Original forward pass
        result = self.original_module(x)
        
        # Add LoRA adaptation if enabled
        if self.lora_enabled and self.training:
            # Ensure LoRA parameters are on the same device as input
            if self.lora_A.device != x.device:
                self.lora_A.data = self.lora_A.data.to(x.device)
                self.lora_B.data = self.lora_B.data.to(x.device)
            lora_result = F.linear(x, (self.lora_B @ self.lora_A) * self.scaling)
            result = result + lora_result
            
        return result
    
    def enable_lora(self):
        self.lora_enabled = True
        
    def disable_lora(self):
        self.lora_enabled = False


class LoRAManager:
    """
    Manager for LoRA layers in UNet upsampling blocks
    """
    def __init__(self, unet, target_layers=None, rank=4, alpha=1.0):
        self.unet = unet
        self.rank = rank
        self.alpha = alpha
        self.lora_layers = {}
        self.original_modules = {}
        # 🚀 Reduce LoRA modules to a quarter of the original (from 8 to 2) to lower computational overhead
        # Only keep the core attention modules of layer 2: to_q (query) and to_k (key)
        self.target_layers = target_layers or [(2, 'to_q'), (2, 'to_k')]
        
    def inject_lora(self):
        """
        Inject LoRA layers into specified UNet upsampling attention modules
        """
        for layer_idx, module_name in self.target_layers:
            if layer_idx < len(self.unet.up_blocks):
                up_block = self.unet.up_blocks[layer_idx]
                
                for sublayer_idx, attention in enumerate(up_block.attentions):
                    attn_module = attention.transformer_blocks[0].attn1
                    
                    # Get the target module (e.g., to_q, to_k, to_v, to_out.0)
                    if '.' in module_name:
                        module_parts = module_name.split('.')
                        target_module = attn_module
                        for part in module_parts:
                            if part.isdigit():
                                target_module = target_module[int(part)]
                            else:
                                target_module = getattr(target_module, part)
                    else:
                        target_module = getattr(attn_module, module_name)
                    
                    # Create LoRA layer
                    lora_key = f"up_{layer_idx}_{sublayer_idx}_{module_name}"
                    self.original_modules[lora_key] = target_module
                    lora_layer = LoRALayer(target_module, self.rank, self.alpha)
                    self.lora_layers[lora_key] = lora_layer
                    
                    # Replace original module with LoRA layer
                    if '.' in module_name:
                        module_parts = module_name.split('.')
                        parent_module = attn_module
                        for part in module_parts[:-1]:
                            if part.isdigit():
                                parent_module = parent_module[int(part)]
                            else:
                                parent_module = getattr(parent_module, part)
                        setattr(parent_module, module_parts[-1], lora_layer)
                    else:
                        setattr(attn_module, module_name, lora_layer)
    
    def remove_lora(self):
        """
        Remove LoRA layers and restore original modules
        """
        for layer_idx, module_name in self.target_layers:
            if layer_idx < len(self.unet.up_blocks):
                up_block = self.unet.up_blocks[layer_idx]
                
                for sublayer_idx, attention in enumerate(up_block.attentions):
                    attn_module = attention.transformer_blocks[0].attn1
                    lora_key = f"up_{layer_idx}_{sublayer_idx}_{module_name}"
                    
                    if lora_key in self.original_modules:
                        # Restore original module
                        if '.' in module_name:
                            module_parts = module_name.split('.')
                            parent_module = attn_module
                            for part in module_parts[:-1]:
                                if part.isdigit():
                                    parent_module = parent_module[int(part)]
                                else:
                                    parent_module = getattr(parent_module, part)
                            setattr(parent_module, module_parts[-1], self.original_modules[lora_key])
                        else:
                            setattr(attn_module, module_name, self.original_modules[lora_key])
    
    def get_lora_parameters(self):
        """
        Get all LoRA parameters for optimization
        """
        lora_params = []
        for lora_layer in self.lora_layers.values():
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
        return lora_params
    
    def enable_lora(self):
        """
        Enable LoRA for all layers
        """
        for lora_layer in self.lora_layers.values():
            lora_layer.enable_lora()
    
    def disable_lora(self):
        """
        Disable LoRA for all layers
        """
        for lora_layer in self.lora_layers.values():
            lora_layer.disable_lora()


"""
Define Pipeline
"""

class SGI2VPipe(StableVideoDiffusionPipeline):
    """
    Modified from the original SVD pipeline
    ref: https://github.com/huggingface/diffusers/blob/24c7d578baf6a8b79890101dd280278fff031d12/src/diffusers/pipelines/stable_video_diffusion/pipeline_stable_video_diffusion.py#L139
    """
    heatmap = {}
    
    def get_depth_at_point(self, depth_map, x, y):
        """Get the depth value at a specified point."""
        h, w = depth_map.shape
        x = int(np.clip(x, 0, w-1))
        y = int(np.clip(y, 0, h-1))
        return depth_map[y, x]
    
    def smooth_depth_changes(self, depths, max_change_percent=3.0):
        """Smooth depth changes to prevent sudden jumps between adjacent frames."""
        if len(depths) <= 1:
            return depths
        
        smoothed = [depths[0]]  # Keep the depth of the first frame as is
        max_change_ratio = max_change_percent / 100.0
        
        for i in range(1, len(depths)):
            prev_depth = smoothed[i-1]
            current_depth = depths[i]
            
            # Calculate the change ratio
            if prev_depth > 0:
                change_ratio = abs(current_depth - prev_depth) / prev_depth
                
                # If the change exceeds the limit, adjust the current depth
                if change_ratio > max_change_ratio:
                    if current_depth > prev_depth:
                        # Depth increased too much, limit the growth
                        adjusted_depth = prev_depth * (1 + max_change_ratio)
                    else:
                        # Depth decreased too much, limit the reduction
                        adjusted_depth = prev_depth * (1 - max_change_ratio)
                    
                    smoothed.append(adjusted_depth)
                else:
                    smoothed.append(current_depth)
            else:
                # If the previous frame's depth is 0, use the current depth directly
                smoothed.append(current_depth)
        
        return smoothed
    
    def calculate_scale_from_depth(self, depth, base_depth, scale_factor=1.5):
        """Calculate a scaling factor based on depth."""
        if base_depth <= 0:
            return 1.0
        
        # Greater depth (further away) results in smaller scale; smaller depth (closer) results in larger scale.
        relative_depth = depth / base_depth
        scale = 1.0 / (1.0 + scale_factor * (relative_depth - 1.0))
        return np.clip(scale, 0.7, 1.5)  # Limit the scaling range to avoid excessive changes
    
    def apply_depth_scaling_to_trajectory(self, trajectory_points, depth_map, image_height, image_width):
        """Apply depth scaling to trajectories to generate 3D-enhanced trajectories."""
        if depth_map is None:
            return trajectory_points  # Return original trajectories if no depth map is provided
        
        # Resize depth map to match the image dimensions
        depth_map_resized = torch.nn.functional.interpolate(
            torch.from_numpy(depth_map).unsqueeze(0).unsqueeze(0).float(),
            size=(image_height, image_width),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        scaled_trajectory_points = []
        
        for obj_idx, obj_trajectory in enumerate(trajectory_points):
            # Analyze depth changes for each object
            obj_depths = []
            for frame_idx, bbox in enumerate(obj_trajectory):
                # Calculate the center of the bounding box
                y1, x1, y2, x2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Get the depth at the center point
                depth = self.get_depth_at_point(depth_map_resized, center_x, center_y)
                obj_depths.append(depth)
            
            # Apply depth smoothing (limit change between adjacent frames to 3%)
            smoothed_depths = self.smooth_depth_changes(obj_depths, max_change_percent=3.0)
            base_depth = smoothed_depths[0]  # Use the first frame's depth as a baseline
            
            # Generate scaled trajectories
            scaled_obj_trajectory = []
            for frame_idx, bbox in enumerate(obj_trajectory):
                current_depth = smoothed_depths[frame_idx]
                scale = self.calculate_scale_from_depth(current_depth, base_depth)
                
                # Apply scaling to the bounding box
                y1, x1, y2, x2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # New dimensions after scaling
                new_width = width * scale
                new_height = height * scale
                
                # Calculate new bounding box coordinates
                new_x1 = center_x - new_width / 2
                new_y1 = center_y - new_height / 2
                new_x2 = center_x + new_width / 2
                new_y2 = center_y + new_height / 2
                
                # Ensure coordinates are within image bounds
                new_x1 = max(0, min(image_width-1, new_x1))
                new_y1 = max(0, min(image_height-1, new_y1))
                new_x2 = max(0, min(image_width-1, new_x2))
                new_y2 = max(0, min(image_height-1, new_y2))
                
                scaled_obj_trajectory.append(np.array([new_y1, new_x1, new_y2, new_x2], dtype=np.float32))
            
            scaled_trajectory_points.append(scaled_obj_trajectory)
            print(f"   🎯 Object {obj_idx+1}: Depth range [{min(smoothed_depths):.3f}, {max(smoothed_depths):.3f}], Base depth {base_depth:.3f}")
        
        # print("scaled_trajectory_points_len:", len(scaled_trajectory_points))
        # print("scaled_trajectory_points[0]:", len(scaled_trajectory_points[0]))
        # print("scaled_trajectory_points[0][0]:", scaled_trajectory_points[0][0].shape)
        '''Trajectory format
        obj1_traj1 = [
            np.array([490, 100, 590, 200], dtype=np.float32), # Frame 0: [y1, x1, y2, x2]
            np.array([490, 500, 590, 600], dtype=np.float32), # Frame 1
            np.array([490, 900, 590, 1000], dtype=np.float32) # Frame 2
        ]
        obj1_traj2 = [
            np.array([490, 100, 590, 200], dtype=np.float32), # Frame 0: [y1, x1, y2, x2]
            np.array([490, 500, 590, 600], dtype=np.float32), # Frame 1
            np.array([490, 900, 590, 1000], dtype=np.float32) # Frame 2
        ]
        '''

        return scaled_trajectory_points
    
    def __init__(
        self,
        vae,
        image_encoder,
        unet,
        scheduler,
        feature_extractor,
    ):
        super().__init__(
            vae=vae,
            image_encoder=image_encoder,
            unet=unet,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )
        self.lora_manager = None
        
    def get_gaussian_heatmap(self, h, w):
        """
        Generate gaussian heatmap
        Modified from https://github.com/showlab/DragAnything/blob/main/demo.py#L380
        """
        if (h,w) in self.heatmap:
            isotropicGrayscaleImage = self.heatmap[(h,w)]
        else:
            sigy = self.unet.heatmap_sigma*(h/2)
            sigx = self.unet.heatmap_sigma*(w/2)

            cx = w/2
            cy = h/2
            isotropicGrayscaleImage = np.zeros((h, w), np.float32)
            for y in range(h):
                for x in range(w):
                    isotropicGrayscaleImage[y, x] = 1 / 2 / np.pi / (sigx*sigy) * np.exp(
                        -1 / 2 * ((x+0.5 - cx) ** 2 / (sigx ** 2) + (y+0.5 - cy) ** 2 / (sigy ** 2)))
            isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)
            self.heatmap[(h,w)] = isotropicGrayscaleImage
        return torch.from_numpy(isotropicGrayscaleImage).cuda()
    
    def optimize_latent(self, latents, trajectory_points, t, image_latents, image_embeddings, added_time_ids):
        """
        trajectory_points : N x frames x 4 (i.e. upper-left and bottom-right corners of bounding box : sx,sy,tx,ty)
        Enhanced with LoRA training for UNet upsampling layers
        """
        original_latents = latents.clone().detach()
        
        if self.unet.optimize_latent_iter > 0: 
            # Initialize LoRA manager for this video generation (if enabled)
            enable_lora = getattr(self.unet, 'enable_lora', False)
            if enable_lora and self.lora_manager is None:
                self.lora_manager = LoRAManager(
                    self.unet, 
                    rank=getattr(self.unet, 'lora_rank', 4),
                    alpha=getattr(self.unet, 'lora_alpha', 4)
                )
                self.lora_manager.inject_lora()
                # Ensure LoRA parameters are on the correct device
                for lora_layer in self.lora_manager.lora_layers.values():
                    lora_layer.lora_A.data = lora_layer.lora_A.data.to(latents.device)
                    lora_layer.lora_B.data = lora_layer.lora_B.data.to(latents.device)
                print("LoRA layers injected into UNet upsampling blocks")
            
            # Enable LoRA for optimization (if available)
            if self.lora_manager is not None:
                self.lora_manager.enable_lora()
            
            self.unet = self.unet.to(dtype=torch.float32)
            self.unet.enable_gradient_checkpointing()
            self.unet.train(True)
            latents = latents.to(dtype=torch.float32)
            image_latents = image_latents.to(dtype=torch.float32)
            image_embeddings = image_embeddings.to(dtype=torch.float32)
            added_time_ids = added_time_ids.to(dtype=torch.float32)
            t = t.to(dtype=torch.float32)
            
            with torch.enable_grad():
                latents = latents.clone().detach().requires_grad_(True)        
                
                # Get LoRA parameters for optimization (if LoRA is enabled)
                lora_params = self.lora_manager.get_lora_parameters() if self.lora_manager else []
                
                # Create optimizer for both latents and LoRA parameters
                optimizer = None 
                scaler = torch.cuda.amp.GradScaler()
                target_features = [None]*len(trajectory_points)
                
                for iter in range(self.unet.optimize_latent_iter):
                    
                    latent_model_input = self.scheduler.scale_model_input(latents, t)
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                    self.unet.record_value_ = []
                    with torch.autocast(device_type = "cuda", dtype=torch.float16):
                        noise_prior = self.unet(latent_model_input,t,encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False)[0]
                        #print("noise_prior.shape:", noise_prior.shape)

                    features = [] #list([frame, h, w, feature])

                    #Upsample recorded feature maps to the latent size
                    h, w = (self.unet.latent_shape[-2], self.unet.latent_shape[-1])
                    for i in range(len(self.unet.record_value_)):
                        fet = self.unet.record_value_[i].to(dtype=torch.float32).permute((0,3,1,2)) #[frame, feature, h, w]
                        fet = F.interpolate(fet, size=(h,w), mode="bilinear")
                        features.append(fet.permute((0,2,3,1)))

                    feature = torch.cat(features, dim=-1) #[frames, h, w, features]
                    
                    self.unet.record_value_ = []
                        
                    compress_factor = 8*self.unet.latent_shape[-2]//h
                        
                    frames = features[0].shape[0]
                        
                    loss = 0
                    loss_cnt = 0
                    
                    for j in range(frames):
                        #iterate over each control point
                        for point_idx in range(len(trajectory_points)):  
                            cur_point = (trajectory_points[point_idx][j]//compress_factor).astype(np.int32)
                            
                            sx, sy = cur_point[0], cur_point[1]
                            tx, ty = max(sx+1, cur_point[2]), max(sy+1, cur_point[3])
                            
                            #boundary check
                            sx_, sy_, tx_, ty_ = max(sx, 0), max(sy, 0), min(tx, feature.shape[1]), min(ty, feature.shape[2])
                            
                            #compute offset
                            osx, osy, otx, oty  = sx_ - sx, sy_ - sy, tx_ - sx, ty_ - sy

                            if sx_ >= tx_ or sy_ >= ty_:
                                #trajectory point goes beyond the image boundary
                                if j==0:
                                    print("Invalid trajectory, the initial boundaing box should not go beyond image boundary!!")
                                    exit(1)
                                continue
                            
                            if j == 0:
                                #Record feature maps of the first frame
                                target_features[point_idx] = feature[0,sx:tx,sy:ty].clone().detach().requires_grad_(False)
                            
                            #Compute loss
                            if j > 0:
                                    target = target_features[point_idx].unsqueeze(0) #[1, h, w, feature]
                                    target = F.interpolate(target.permute((0,3,1,2)),size=(tx-sx, ty-sy), mode="bilinear") #[1,feature,h,w]
                                    target = target.permute((0,2,3,1))[0] #[h, w, feature]
                                    target = target[osx:otx, osy:oty] #[h', w', feature]
                                    source = feature[j,sx_:tx_,sy_:ty_]
                                    
                                    #compute pixel-wise difference
                                    pixel_wise_loss = F.mse_loss(target, source, reduction="none").mean(dim=-1)
                                    
                                    #gaussian weight applied
                                    mask = self.get_gaussian_heatmap(tx-sx, ty-sy)
                                    mask = mask[osx:otx, osy:oty]
                                    assert(mask.shape == pixel_wise_loss.shape)

                                    #weight loss depending on the weight
                                    pixel_wise_loss = pixel_wise_loss
                                    
                                    #add up to the loss
                                    loss = loss + (mask*pixel_wise_loss).sum()
                                    loss_cnt += mask.sum()
                        
                    loss = loss/max(1e-8, loss_cnt)
                    
                    if optimizer == None:
                        #Initialize optimizer for both latents and LoRA parameters
                        param_groups = [{'params': [latents], 'lr': self.unet.optimize_latent_lr}]
                        if lora_params:  # Only add LoRA params if they exist
                            param_groups.append({'params': lora_params, 'lr': self.unet.optimize_latent_lr * 0.1})
                        optimizer = torch.optim.AdamW(param_groups) 
                    
                    self.unet.zero_grad()
                    optimizer.zero_grad()
                    if loss_cnt > 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    del loss
                    del feature
                    del self.unet.record_value_
                    torch.cuda.empty_cache()

                    if loss_cnt==0:
                        break #nothing to optimize
            
            # Keep LoRA enabled during optimization - it will be disabled in the main loop
            
            # FFT post-processing
            fft_axis = (-2, -1) #H and W
            with torch.no_grad():
                #latents: [1, frames = 14, channel = 4, h, w]               
                #create low pass filter
                LPF = butterworth_low_pass_filter(latents, d_s = self.unet.latent_fft_ratio)
                
                #FFT
                latents_freq = torch.fft.fftn(latents, dim=fft_axis)
                latents_freq = torch.fft.fftshift(latents_freq, dim=fft_axis)
                original_latents_freq = torch.fft.fftn(original_latents.to(dtype=torch.float32), dim=fft_axis)
                original_latents_freq = torch.fft.fftshift(original_latents_freq, dim=fft_axis)
    
                #frequency mix
                HPF = 1 - LPF
                new_freq = latents_freq*LPF + original_latents_freq*HPF
    
                #IFFT
                new_freq = torch.fft.ifftshift(new_freq, dim=fft_axis)
                latents = torch.fft.ifftn(new_freq, dim=fft_axis).real
            
            self.unet = self.unet.to(dtype=torch.float16)
            latents = latents.to(dtype=torch.float16)
            image_latents = image_latents.to(dtype=torch.float16)
            image_embeddings = image_embeddings.to(dtype=torch.float16)
            added_time_ids = added_time_ids.to(dtype=torch.float16)
            t = t.to(dtype=torch.float16)
            self.unet.train(False)
        
        latents = latents.detach().requires_grad_(False)
        return latents
    
    def optimize_control_force(self, control_force, latents, trajectory_points, t, noise_pred_uncond):
        """
        Optimize the control force vector using a regional loss in the latent space.
        control_force: [1, frames, 4, h, w] Control force vector
        latents: [1, frames, 4, h, w] Current latent
        trajectory_points: Trajectory point information
        """
        if self.unet.optimize_latent_iter <= 0:
            return control_force
            
        print("🎯 Starting control force vector optimization...")
        original_control_force = control_force.clone().detach()
        
        # Set training mode and precision
        control_force = control_force.to(dtype=torch.float32)
        latents = latents.to(dtype=torch.float32)
        noise_pred_uncond = noise_pred_uncond.to(dtype=torch.float32)
        t = t.to(dtype=torch.float32)
        
        with torch.enable_grad():
            control_force = control_force.clone().detach().requires_grad_(True)
            
            optimizer = None
            scaler = torch.cuda.amp.GradScaler()
            target_latent_features = [None] * len(trajectory_points)
            
            # Control force vector optimization iterations (3 times)
            for iter in range(3):
                # Reconstruct noise_pred_cond
                noise_pred_cond = control_force + noise_pred_uncond
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Manually calculate the pre-denoised result (without changing the scheduler state)
                # Get the sigma value for the current timestep
                sigma = self.scheduler.sigmas[self.scheduler.step_index]
                predicted_latents = (latents - 0.1*sigma * noise_pred) / (1 + sigma)
                
                # Calculate the regional loss in the latent space
                loss = 0
                loss_cnt = 0
                
                # Latent feature dimensions: [1, frames, 4, h, w]
                frames = predicted_latents.shape[1]
                h, w = predicted_latents.shape[-2], predicted_latents.shape[-1]
                
                # Calculate compression factor (from image space to latent space)
                compress_factor = 8  # VAE downsampling factor
                
                for j in range(frames):
                    # Iterate over each control point
                    for point_idx in range(len(trajectory_points)):
                        cur_point = (trajectory_points[point_idx][j] // compress_factor).astype(np.int32)
                        
                        sx, sy = cur_point[0], cur_point[1]
                        tx, ty = max(sx+1, cur_point[2]), max(sy+1, cur_point[3])
                        
                        # Boundary check
                        sx_, sy_, tx_, ty_ = max(sx, 0), max(sy, 0), min(tx, h), min(ty, w)
                        
                        # Calculate offset
                        osx, osy, otx, oty = sx_ - sx, sy_ - sy, tx_ - sx, ty_ - sy
                        
                        if sx_ >= tx_ or sy_ >= ty_:
                            if j == 0:
                                print("Invalid trajectory in latent space!")
                                continue
                            continue
                        
                        if j == 0:
                            # Record the latent features of the first frame as a reference
                            target_latent_features[point_idx] = predicted_latents[0, 0, :, sx_:tx_, sy_:ty_].clone().detach().requires_grad_(False)
                        
                        # Calculate loss (starting from the second frame)
                        if j > 0:
                            target = target_latent_features[point_idx]  # [4, h', w']
                            
                            # Adjust target feature size to match the current region
                            target = target.unsqueeze(0)  # [1, 4, h', w']
                            target = F.interpolate(target, size=(tx-sx, ty-sy), mode="bilinear")  # [1, 4, h'', w'']
                            target = target[0]  # [4, h'', w'']
                            target = target[:, osx:otx, osy:oty]  # [4, h''', w''']
                            
                            # Latent features of the corresponding region in the current frame
                            source = predicted_latents[0, j, :, sx_:tx_, sy_:ty_]  # [4, h''', w''']
                            
                            # Calculate pixel-wise latent feature difference
                            pixel_wise_loss = F.mse_loss(target, source, reduction="none").mean(dim=0)  # [h''', w''']
                            
                            # Apply Gaussian weights
                            mask = self.get_gaussian_heatmap(tx-sx, ty-sy)
                            mask = mask[osx:otx, osy:oty]
                            
                            if mask.shape == pixel_wise_loss.shape:
                                # Accumulate loss
                                loss = loss + (mask * pixel_wise_loss).sum()
                                loss_cnt += mask.sum()
                
                # Normalize the loss
                loss = 0.01 * loss / max(1e-8, loss_cnt)
                
                if optimizer is None:
                    # Initialize optimizer (only for the control force vector)
                    optimizer = torch.optim.AdamW([control_force], lr=self.unet.optimize_latent_lr * 0.01)
                
                # Backpropagate and update the control force vector
                optimizer.zero_grad()
                if loss_cnt > 0:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                
                # Clear memory
                del loss
                del predicted_latents
                torch.cuda.empty_cache()
                
                if loss_cnt == 0:
                    break
        
        # FFT post-processing (optional)
        fft_axis = (-2, -1)
        with torch.no_grad():
            # Create a low-pass filter
            LPF = butterworth_low_pass_filter(control_force, d_s=self.unet.latent_fft_ratio)
            
            # FFT transform
            control_force_freq = torch.fft.fftn(control_force, dim=fft_axis)
            control_force_freq = torch.fft.fftshift(control_force_freq, dim=fft_axis)
            original_control_force_freq = torch.fft.fftn(original_control_force.to(dtype=torch.float32), dim=fft_axis)
            original_control_force_freq = torch.fft.fftshift(original_control_force_freq, dim=fft_axis)
            
            # Frequency domain mixing
            HPF = 1 - LPF
            new_freq = control_force_freq * LPF + original_control_force_freq * HPF
            
            # Inverse FFT transform
            new_freq = torch.fft.ifftshift(new_freq, dim=fft_axis)
            control_force = torch.fft.ifftn(new_freq, dim=fft_axis).real
        
        # Restore precision
        control_force = control_force.to(dtype=torch.float16)
        control_force = control_force.detach().requires_grad_(False)
        
        print("✅ Control force vector optimization complete.")
        return control_force
    
    def cleanup_lora(self):
        """
        Clean up LoRA layers after video generation
        """
        if self.lora_manager is not None:
            self.lora_manager.remove_lora()
            self.lora_manager = None
            print("LoRA layers removed and cleaned up")
    
    def __call__(self, image, trajectory_points, height, width, num_frames, min_guidance_scale = 1.0, max_guidance_scale = 3.0, fps = 7,
                 generator = None, motion_bucket_id = 127, noise_aug_strength = 0.02, decode_chunk_size = 8, depth_map = None):
        #Modified from the original implementaion such that the pipeline incorporates our latent optimization procedure
        batch_size = 1
        fps = fps - 1
        self._guidance_scale = max_guidance_scale
        
        # 🔧 Ensure all model components are on the correct device (fixes device mismatch issues in batch processing)
        self.vae = self.vae.to("cuda")
        self.image_encoder = self.image_encoder.to("cuda")
        
        # 🎯 Apply depth scaling to enhance trajectories (if a depth map is provided)
        if depth_map is not None:
            print("🔍 Applying depth scaling to enhance trajectories...")
            trajectory_points = self.apply_depth_scaling_to_trajectory(trajectory_points, depth_map, height, width)
            print("✅ Depth scaling enhancement complete.")
        
        image_embeddings = self._encode_image(image, "cuda", 1, self.do_classifier_free_guidance)
        try: 
            image = self.image_processor.preprocess(image, height=height, width=width).to("cuda")
        except:
            self.image_processor = self.video_processor 
            image = self.image_processor.preprocess(image, height=height, width=width).to("cuda")
        noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype).to("cuda")
        image = image + noise_aug_strength * noise
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if needs_upcasting:
            self.vae.to(dtype=torch.float32)
        image_latents = self._encode_vae_image(
            image,
            device="cuda",
            num_videos_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
        )
        image_latents = image_latents.to(image_embeddings.dtype)
        if needs_upcasting:
            self.vae.to(dtype=torch.float16)
        image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            1,
            self.do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to("cuda")
        self.scheduler.set_timesteps(self.unet.num_inference_steps, device="cuda")
        timesteps = self.scheduler.timesteps
        num_channels_latents = self.unet.config.in_channels
        shape = (
            batch_size,
            num_frames,
            num_channels_latents // 2,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = randn_tensor(shape, generator=generator, device=image.device, dtype=image_embeddings.dtype).to("cuda")
        latents = latents * self.scheduler.init_noise_sigma # scale the initial noise by the standard deviation required by the scheduler
        self.unet.latent_shape = latents.shape
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to("cuda", latents.dtype)
        guidance_scale = guidance_scale.repeat(batch_size, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)
        self._guidance_scale = guidance_scale

        #Free cuda memory
        self.vae = self.vae.to("cpu")
        self.image_encoder = self.image_encoder.to("cpu")
        torch.cuda.empty_cache()
            
        #Denoising loop
        num_warmup_steps = len(timesteps) - self.unet.num_inference_steps * self.scheduler.order #num_warmup_steps = 0 in our setting
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=self.unet.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                self.unet.cur_timestep = len(timesteps) - i
                
                # Ensure LoRA is disabled during regular inference
                if self.lora_manager is not None:
                    self.lora_manager.disable_lora()
                
                # Latent optimization and LoRA training happens here
                if (self.unet.cur_timestep in self.unet.optimize_latent_time):
                    #update latent through optimization (now includes LoRA training)
                    latents = self.optimize_latent(latents, trajectory_points, t, image_latents[1:], image_embeddings[1:], added_time_ids[1:])
                    # Disable LoRA again after optimization
                    if self.lora_manager is not None:
                        self.lora_manager.disable_lora()
                
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

                # Officially predict noise using the updated latent 
                # (The previous pass through UNet was only for feature extraction and latent updating, not for noise prediction)
                with torch.no_grad():
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=image_embeddings,
                        added_time_ids=added_time_ids,
                        return_dict=False,
                    )[0]
                    #print("noise_pred.shape:", noise_pred.shape)
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    
                    # 🎯 Control force vector optimization (activated in the same optimization steps)
                    enable_control_force_opt = getattr(self.unet, 'enable_control_force_optimization', False)
                    if enable_control_force_opt and (self.unet.cur_timestep in self.unet.optimize_latent_time):
                        # Extract the control force vector
                        control_force = noise_pred_cond - noise_pred_uncond
                        print(f"🎯 Step {i+1}/{self.unet.num_inference_steps}: Initiating control force vector optimization (Control force shape: {control_force.shape})")
                        
                        # Optimize the control force vector
                        optimized_control_force = self.optimize_control_force(
                            control_force, latents, trajectory_points, t, noise_pred_uncond
                        )
                        
                        # Reconstruct the optimized noise_pred_cond
                        noise_pred_cond = optimized_control_force + noise_pred_uncond
                        print("✅ Control force vector optimization complete, noise prediction updated.")
                    
                    # Calculate the final noise prediction
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)
                
                # Official denoising step
                latents = self.scheduler.step(noise_pred, t, latents,  s_churn = 0.0).prev_sample
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        # Clean up LoRA after generation
        self.cleanup_lora()
        
        # 🔧 Ensure models are restored to the correct device and data type
        self.vae = self.vae.to("cuda")
        self.image_encoder = self.image_encoder.to("cuda")
        if needs_upcasting:
            self.vae = self.vae.to(dtype=torch.float16)
        frames = self.decode_latents(latents, num_frames, decode_chunk_size)
        frames = tensor2vid(frames, self.image_processor, output_type="np")
        self.maybe_free_model_hooks()
        return StableVideoDiffusionPipelineOutput(frames=frames)