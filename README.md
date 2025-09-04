# Zo3T: Zero-shot 3D-Aware Trajectory-Guided image-to-video generation via Test-Time Training

<p align="center">
  <strong>Ruicheng Zhang</strong><sup>1,2*</sup>,
  <strong>Jun Zhou</strong><sup>1*</sup>,
  <strong>Zunnan Xu</strong><sup>1*</sup>,
  <strong>Zihao Liu</strong><sup>1</sup>,
  <strong>Jiehui Huang</strong><sup>3</sup>,
  <strong>Mingyang Zhang</strong><sup>4</sup>,
  <strong>Yu Sun</strong><sup>2</sup>,
  <strong>Xiu Li</strong><sup>1†</sup>

<p align="center"><sub>
  <sup>1</sup> Tsinghua University, <sup>2</sup> Sun Yat-sen University<br>
  <sup>3</sup> The Hong Kong University of Science and Technology, <sup>4</sup> China University of Mining and Technology
</sub></p>

<p align="center"><sub>* Equal contribution. † Corresponding author.</sub></p>

## Framework Overview

![Framework Diagram](asserts/images/framework.png)

<p><small><i>
An overview of our zero-shot trajectory-guided video generation framework.
Our method optimizes a pre-trained video diffusion model at specific denoising timesteps via two key stages.
First, <b>Test-Time Training (TTT)</b> adapts the latent state and an ephemeral adapter to maintain semantic consistency along the trajectory.
Second, <b>Guidance Field Rectification</b> refines the denoising direction using a one-step lookahead optimization to ensure precise path execution.
</i></small></p>

## Results

### Qualitative Results

#### Object Control (Examples 1-8)

<table>
  <tr>
    <td align="center">
      <img src="asserts/videos/1-compare/condition_vis.png" width="150" alt="Condition 1"/>
      <br><small>Condition 1</small>
      <br>
      <a href="asserts/videos/1-compare/Ours.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 1"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/2-compare/condition_vis.png" width="150" alt="Condition 2"/>
      <br><small>Condition 2</small>
      <br>
      <a href="asserts/videos/2-compare/Ours.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 2"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/3/condition_vis.png" width="150" alt="Condition 3"/>
      <br><small>Condition 3</small>
      <br>
      <a href="asserts/videos/3/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 3"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/4/condition_vis.png" width="150" alt="Condition 4"/>
      <br><small>Condition 4</small>
      <br>
      <a href="asserts/videos/4/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 4"/>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="asserts/videos/5/condition_vis.png" width="150" alt="Condition 5"/>
      <br><small>Condition 5</small>
      <br>
      <a href="asserts/videos/5/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 5"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/6/condition_vis.png" width="150" alt="Condition 6"/>
      <br><small>Condition 6</small>
      <br>
      <a href="asserts/videos/6/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 6"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/7/condition_vis.png" width="150" alt="Condition 7"/>
      <br><small>Condition 7</small>
      <br>
      <a href="asserts/videos/7/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 7"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/8/condition_vis.png" width="150" alt="Condition 8"/>
      <br><small>Condition 8</small>
      <br>
      <a href="asserts/videos/8/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 8"/>
      </a>
    </td>
  </tr>
</table>

#### Camera Control (Examples 9-16)

<table>
  <tr>
    <td align="center">
      <img src="asserts/videos/9/condition_vis.png" width="150" alt="Condition 9"/>
      <br><small>Condition 9</small>
      <br>
      <a href="asserts/videos/9/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 9"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/10/condition_vis.png" width="150" alt="Condition 10"/>
      <br><small>Condition 10</small>
      <br>
      <a href="asserts/videos/10/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 10"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/11/condition_vis.png" width="150" alt="Condition 11"/>
      <br><small>Condition 11</small>
      <br>
      <a href="asserts/videos/11/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 11"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/12/condition_vis.png" width="150" alt="Condition 12"/>
      <br><small>Condition 12</small>
      <br>
      <a href="asserts/videos/12/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 12"/>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="asserts/videos/13/condition_vis.png" width="150" alt="Condition 13"/>
      <br><small>Condition 13</small>
      <br>
      <a href="asserts/videos/13/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 13"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/14/condition_vis.png" width="150" alt="Condition 14"/>
      <br><small>Condition 14</small>
      <br>
      <a href="asserts/videos/14/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 14"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/15/condition_vis.png" width="150" alt="Condition 15"/>
      <br><small>Condition 15</small>
      <br>
      <a href="asserts/videos/15/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 15"/>
      </a>
    </td>
    <td align="center">
      <img src="asserts/videos/16/condition_vis.png" width="150" alt="Condition 16"/>
      <br><small>Condition 16</small>
      <br>
      <a href="asserts/videos/16/result.mp4">
        <img src="https://img.shields.io/badge/▶️-Watch_Video-blue" alt="Video 16"/>
      </a>
    </td>
  </tr>
</table>

### Qualitative Comparisons

#### Example 1

<table>
  <tr>
    <td align="center">
      <img src="asserts/videos/1-compare/condition_vis.png" width="180" alt="Condition"/>
      <br><strong>Condition</strong>
    </td>
    <td align="center">
      <a href="asserts/videos/1-compare/Ours.mp4">
        <img src="https://img.shields.io/badge/▶️-Ours-green?style=for-the-badge" alt="Our Result"/>
      </a>
    </td>
    <td align="center">
      <a href="asserts/videos/1-compare/DragAnything.mp4">
        <img src="https://img.shields.io/badge/▶️-DragAnything-orange?style=for-the-badge" alt="DragAnything"/>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="asserts/videos/1-compare/DragNUWA.mp4">
        <img src="https://img.shields.io/badge/▶️-DragNUWA-purple?style=for-the-badge" alt="DragNUWA"/>
      </a>
    </td>
    <td align="center">
      <a href="asserts/videos/1-compare/SG-I2V.mp4">
        <img src="https://img.shields.io/badge/▶️-SG--I2V-blue?style=for-the-badge" alt="SG-I2V"/>
      </a>
    </td>
    <td align="center">
      <a href="asserts/videos/1-compare/ObjCtrl-2.5D.mp4">
        <img src="https://img.shields.io/badge/▶️-ObjCtrl--2.5D-red?style=for-the-badge" alt="ObjCtrl-2.5D"/>
      </a>
    </td>
  </tr>
</table>

#### Example 2

<table>
  <tr>
    <td align="center">
      <img src="asserts/videos/2-compare/condition_vis.png" width="180" alt="Condition"/>
      <br><strong>Condition</strong>
    </td>
    <td align="center">
      <a href="asserts/videos/2-compare/Ours.mp4">
        <img src="https://img.shields.io/badge/▶️-Ours-green?style=for-the-badge" alt="Our Result"/>
      </a>
    </td>
    <td align="center">
      <a href="asserts/videos/2-compare/DragAnything.mp4">
        <img src="https://img.shields.io/badge/▶️-DragAnything-orange?style=for-the-badge" alt="DragAnything"/>
      </a>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="asserts/videos/2-compare/DragNUWA.mp4">
        <img src="https://img.shields.io/badge/▶️-DragNUWA-purple?style=for-the-badge" alt="DragNUWA"/>
      </a>
    </td>
    <td align="center">
      <a href="asserts/videos/2-compare/SG-I2V.mp4">
        <img src="https://img.shields.io/badge/▶️-SG--I2V-blue?style=for-the-badge" alt="SG-I2V"/>
      </a>
    </td>
    <td align="center">
      <a href="asserts/videos/2-compare/ObjCtrl-2.5D.mp4">
        <img src="https://img.shields.io/badge/▶️-ObjCtrl--2.5D-red?style=for-the-badge" alt="ObjCtrl-2.5D"/>
      </a>
    </td>
  </tr>
</table>

> **注意**: 点击上方的视频标志可以下载并播放相应的视频文件。

### 替代方案：GIF展示

如果您希望直接在README中显示动态内容，建议将重要的MP4视频转换为GIF格式：

```bash
# 使用 ffmpeg 将 MP4 转换为 GIF
ffmpeg -i asserts/videos/1-compare/Ours.mp4 -vf "fps=10,scale=320:-1:flags=lanczos" -loop 0 asserts/videos/1-compare/Ours.gif
```

然后可以直接使用：
```markdown
![Our Result](asserts/videos/1-compare/Ours.gif)
```

## Getting Started

### Prerequisites

- Python 3.12
- PyTorch
- `diffusers`, `transformers`, `accelerate`
- `numpy`, `opencv-python`, `matplotlib`, `Pillow`
- A pre-trained Stable Video Diffusion model.

### Installation

Follow these steps to set up the environment and install all necessary dependencies.

**1. Clone the Repository**

First, clone the Zo3T repository to your local machine:

```bash
git clone https://github.com/your-username/Zo3T-main.git
cd Zo3T-main
```

**2. Create and Activate a Conda Environment**

We recommend using `conda` to manage dependencies. Create a new environment and activate it:

```bash
conda create -n zo3t python=3.12 -y
conda activate zo3t
```

**3. Install All Dependencies**

All required packages are listed in `requirements.txt`. Install them using a single `pip` command:

```bash
pip install -r requirements.txt
```

**4. Download the Stable Video Diffusion Model**

The pipeline requires the weights for the Stable Video Diffusion model. You need to download the `stable-video-diffusion-img2vid` model checkpoint.

- You can download it from the [official Hugging Face repository](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid).
- Place the downloaded model folder in a convenient location.

Then, update the `svd_dir` variable in the `inference.py` script to point to the directory where you saved the model:

```python
# in inference.py
...
#Load pre-trained image-to-video diffusion models
print("Loading Stable Video Diffusion from local path..")
svd_dir = "/path/to/your/stable-video-diffusion-img2vid" # ⬅️ UPDATE THIS PATH
...
```

You are now ready to run the inference script.

### Usage

Prepare your input directory with the following structure:

```
/path/to/your/input_dir/
├── img.png
└── traj.npy
```

- `img.png`: The first frame of the video.
- `traj.npy`: A NumPy array of shape `[N, (2+F), 2]`, where:
  - `N` is the number of objects to track.
  - The first slice `[:, :2, :]` contains the top-left and bottom-right coordinates `[[w1, h1], [w2, h2]]` of the initial bounding boxes.
  - The second slice `[:, 2:, :]` contains the trajectory of the center point for each bounding box over `F` frames.

Run the inference script:

```bash
python inference.py --input_dir /path/to/your/input_dir/ --output_dir /path/to/your/output_dir/
```

### Configuration

Hyperparameters can be adjusted within the `Config` class in `inference.py`:

- `seed`: Random seed for reproducibility.
- `height`, `width`: Resolution of the generated video.
- `num_frames`: Number of frames to generate.
- `num_inference_steps`: Total number of denoising steps.
- `optimize_latent_time`: A list of timesteps at which to perform optimization.
- `optimize_latent_iter`: Number of optimization iterations per timestep.
- `optimize_latent_lr`: Learning rate for latent optimization.
- `enable_lora`: Set to `True` to use LoRA during optimization.
- `enable_depth_scaling`: Set to `True` to enable depth-aware trajectory scaling.
- `enable_control_force_optimization`: Set to `True` to enable control force optimization.
