# app.py (Full code, corrected for Kaggle environment)
import os
import random
from pathlib import Path
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from PIL import Image
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d_emo import EMOUNet3DConditionModel
from src.models.whisper.audio2feature import load_audio_model
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline
from src.utils.util import save_videos_grid
from src.models.pose_encoder import PoseEncoder
from src.utils.dwpose_util import draw_pose_select_v2
from moviepy.editor import VideoFileClip, AudioFileClip

import gradio as gr
from datetime import datetime
from torchao.quantization import quantize_, int8_weight_only
import gc

# --- Global Configuration (Define these as global variables) ---
# These variables were causing the NameError because they were not explicitly passed
# or defined in the scope where generate_button.click was called.
# By defining them globally, they are accessible.
width, height = 768, 768
sample_rate = 16000
cfg = 1.0 # Changed to float for consistency
fps = 24
context_frames = 12
context_overlap = 3

# --- Device and VRAM Info ---
if torch.cuda.is_available():
    device = "cuda"
else:
    print("CUDA not available, using cpu")
    device = "cpu"

total_vram_in_gb = torch.cuda.get_device_properties(0).total_memory / 1073741824 if torch.cuda.is_available() else 0
print(f'\033[32mCUDA version: {torch.version.cuda}\033[0m')
print(f'\033[32mPytorch version: {torch.__version__}\033[0m')
print(f'\033[32mGPU Model: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}\033[0m')
print(f'\033[32mVRAM: {total_vram_in_gb:.2f}GB\033[0m')
print(f'\033[32mPrecision: float16\033[0m')
dtype = torch.float16

# --- FFmpeg Path Check ---
# This part is fine, but make sure ffmpeg is in the PATH in Kaggle environment.
ffmpeg_path = os.getenv('FFMPEG_PATH')
if ffmpeg_path is None:
    print("Please download ffmpeg-static and export to FFMPEG_PATH. \nFor example: export FFMPEG_PATH=./ffmpeg-4.4-amd64-static")
elif ffmpeg_path not in os.environ.get('PATH', ''): # Use .get for robustness
    print("Adding ffmpeg to PATH")
    os.environ["PATH"] = f"{ffmpeg_path}:{os.environ['PATH']}"

# --- Main Generation Function ---
def generate(image_input,
             audio_input,
             pose_input,
             # These parameters now correctly use the global variables defined above
             # or are passed directly as constant values from the Gradio interface setup.
             # They are arguments of this function, not Gradio components directly.
             # Gradio will map the passed values from its inputs= list to these arguments.
             input_width, # Renamed to avoid confusion with global 'width' in function scope
             input_height, # Renamed
             length,
             steps,
             input_sample_rate, # Renamed
             input_cfg, # Renamed
             input_fps, # Renamed
             input_context_frames, # Renamed
             input_context_overlap, # Renamed
             quantization_input,
             seed):

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("./outputs") # Ensure output path is relative to project root
    save_dir.mkdir(exist_ok=True, parents=True)

    # ############# model_init started #############
    print("Initializing models...")
    ## vae init
    vae = AutoencoderKL.from_pretrained("./pretrained_weights/sd-vae-ft-mse").to(device, dtype=dtype)
    if quantization_input:
        quantize_(vae, int8_weight_only())
        print("int8 quantization enabled for VAE.")

    ## reference net init
    reference_unet = UNet2DConditionModel.from_pretrained("./pretrained_weights/sd-image-variations-diffusers", subfolder="unet", use_safetensors=False).to(dtype=dtype, device=device)
    reference_unet.load_state_dict(torch.load("./pretrained_weights/reference_unet.pth", weights_only=True))
    if quantization_input:
        quantize_(reference_unet, int8_weight_only())
        print("int8 quantization enabled for Reference UNet.")

    ## denoising net init
    motion_module_path = "./pretrained_weights/motion_module_acc.pth"
    denoising_unet_path = "./pretrained_weights/denoising_unet_acc.pth"

    if not os.path.exists(motion_module_path):
        gr.Warning(f"Error: Motion module '{motion_module_path}' not found. Please check your pretrained weights download.")
        raise FileNotFoundError(f"Motion module not found at: {motion_module_path}")
    print(f'Using motion module from {motion_module_path}')

    denoising_unet = EMOUNet3DConditionModel.from_pretrained_2d(
        "./pretrained_weights/sd-image-variations-diffusers",
        motion_module_path,
        subfolder="unet",
        unet_additional_kwargs = {
            "use_inflated_groupnorm": True,
            "unet_use_cross_frame_attention": False,
            "unet_use_temporal_attention": False,
            "use_motion_module": True,
            "cross_attention_dim": 384,
            "motion_module_resolutions": [1, 2, 4, 8],
            "motion_module_mid_block": True ,
            "motion_module_decoder_only": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs":{
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ['Temporal_Self', 'Temporal_Self'],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 32,
                "temporal_attention_dim_div": 1,
            }
        },
    ).to(dtype=dtype, device=device)
    denoising_unet.load_state_dict(torch.load(denoising_unet_path, weights_only=True),strict=False)
    if quantization_input:
        quantize_(denoising_unet, int8_weight_only())
        print("int8 quantization enabled for Denoising UNet.")


    # pose net init
    pose_net = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to(dtype=dtype, device=device)
    pose_net.load_state_dict(torch.load("./pretrained_weights/pose_encoder.pth", weights_only=True))

    ### load audio processor params
    audio_processor = load_audio_model(model_path="./pretrained_weights/audio_processor/tiny.pt", device=device)

    # ############# model_init finished #############
    print("Models initialized.")

    sched_kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear",
        "clip_sample": False,
        "steps_offset": 1,
        "prediction_type": "v_prediction",
        "rescale_betas_zero_snr": True,
        "timestep_spacing": "trailing"
    }
    scheduler = DDIMScheduler(**sched_kwargs)

    pipe = EchoMimicV2Pipeline(
        vae=vae,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        audio_guider=audio_processor,
        pose_encoder=pose_net,
        scheduler=scheduler,
    )

    pipe = pipe.to(device, dtype=dtype)

    if seed is not None and seed > -1:
        generator = torch.manual_seed(int(seed)) # Ensure seed is an integer
    else:
        seed = random.randint(100, 1000000)
        generator = torch.manual_seed(seed)
    print(f"Using seed: {seed}")

    print('Reference Image:', image_input)
    print('Audio File:', audio_input)
    print('Pose Directory:', pose_input)

    save_name = f"{save_dir}/{timestamp}"

    # Use the input_width and input_height from Gradio
    ref_image_pil = Image.open(image_input).resize((input_width, input_height))
    audio_clip = AudioFileClip(audio_input)

    # Check and adjust length based on audio duration and available pose files
    if not os.path.isdir(pose_input):
        gr.Warning(f"Error: Pose directory '{pose_input}' does not exist. Please check the path.")
        raise NotADirectoryError(f"Pose directory not found: {pose_input}")

    pose_files = sorted([f for f in os.listdir(pose_input) if f.endswith('.npy')], key=lambda x: int(x.split('.')[0]))
    max_pose_length = len(pose_files)

    # Ensure length is reasonable
    if length <= 0:
        gr.Warning("Error: Calculated video length is zero or negative. Please check audio and pose files.")
        raise ValueError("Invalid video length.")

    # Limit length based on audio duration and actual number of pose files
    effective_length = min(length, int(audio_clip.duration * input_fps), max_pose_length)
    if effective_length <= 0:
        gr.Warning("Error: Effective video length is too short. Check audio duration and pose files.")
        return None, gr.update(visible=True, value="Video generation failed due to insufficient length.")

    start_idx = 0 # Currently fixed at 0

    pose_list = []
    print(f"Preparing {effective_length} pose frames from index {start_idx}...")
    for index in range(start_idx, start_idx + effective_length):
        tgt_musk = np.zeros((input_width, input_height, 3)).astype('uint8')
        tgt_musk_path = os.path.join(pose_input, f"{index}.npy")
        if not os.path.exists(tgt_musk_path):
            gr.Warning(f"Warning: Pose file '{tgt_musk_path}' not found. Adjusting video length to {len(pose_list)} frames.")
            effective_length = len(pose_list) # Adjust length to what's available
            break # Exit loop if a pose file is missing
        detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
        imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']
        im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
        im = np.transpose(np.array(im),(1, 2, 0))
        tgt_musk[rb:re,cb:ce,:] = im

        tgt_musk_pil = Image.fromarray(np.array(tgt_musk)).convert('RGB')
        pose_list.append(torch.Tensor(np.array(tgt_musk_pil)).to(dtype=dtype, device=device).permute(2,0,1) / 255.0)

    if not pose_list:
        gr.Warning("Error: No pose frames were loaded. Video generation failed.")
        return None, gr.update(visible=True, value="Video generation failed: No pose frames.")

    poses_tensor = torch.stack(pose_list, dim=1).unsqueeze(0)
    audio_clip = audio_clip.set_duration(effective_length / input_fps) # Trim audio to match video length

    print("Starting video generation pipeline...")
    video = pipe(
        ref_image_pil,
        audio_input, # Path to audio file
        poses_tensor[:,:,:effective_length,...],
        input_width,
        input_height,
        effective_length, # Use adjusted length
        steps,
        input_cfg,
        generator=generator,
        audio_sample_rate=input_sample_rate,
        context_frames=input_context_frames,
        fps=input_fps,
        context_overlap=input_context_overlap,
        start_idx=start_idx,
    ).videos

    final_length = min(video.shape[2], poses_tensor.shape[2], effective_length)
    video_sig = video[:, :, :final_length, :, :]

    output_video_path_wo_audio = str(save_name + "_woa_sig.mp4")
    output_video_path_final = str(save_name + "_sig.mp4")

    print(f"Saving video without audio: {output_video_path_wo_audio}")
    save_videos_grid(
        video_sig,
        output_video_path_wo_audio,
        n_rows=1,
        fps=input_fps,
    )

    print(f"Combining audio with video: {output_video_path_final}")
    try:
        video_clip_sig = VideoFileClip(output_video_path_wo_audio)
        video_clip_sig = video_clip_sig.set_audio(audio_clip)
        video_clip_sig.write_videofile(output_video_path_final, codec="libx264", audio_codec="aac", threads=2)
    except Exception as e:
        gr.Warning(f"Error combining audio and video: {e}. Outputting video without audio.")
        output_video_path_final = output_video_path_wo_audio # Fallback to video without audio
    
    print(f"Final video saved at: {output_video_path_final}")
    seed_text = gr.update(visible=True, value=seed)
    return output_video_path_final, seed_text


# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
            <div>
                <h2 style="font-size: 30px;text-align: center;">EchoMimicV2-ACC</h2>
            </div>
            <div style="text-align: center;">
                <a href="https://github.com/antgroup/echomimic_v2">🌐 Github</a> |
                <a href="https://arxiv.org/abs/2411.10061">📜 arXiv </a>
            </div>
            <div style="text-align: center; font-weight: bold; color: red;">
                ⚠️ This demo is for academic research and experience only
            </div>
            """)
    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Group():
                    image_input = gr.Image(label="Input Image (auto-resized to 768x768)", type="filepath")
                    audio_input = gr.Audio(label="Input Audio", type="filepath")
                    pose_input = gr.Textbox(label="Input Pose (directory path)", placeholder="Enter the directory path for pose data", value="assets/halfbody_demo/pose/fight")
                with gr.Group():
                    length = gr.Number(label="Video Length (frames, recommended 120)", value=120)
                    steps = gr.Number(label="Steps (default 6)", value=6)
                    quantization_input = gr.Checkbox(label="int8 Quantization (recommended for 12GB+ VRAM & audio <= 5s)", value=False)
                    seed = gr.Number(label="Seed (-1 for random)", value=-1)
                generate_button = gr.Button("🎬 Generate Video")
            with gr.Column():
                video_output = gr.Video(label="Output Video")
                seed_text = gr.Textbox(label="Seed", interactive=False, visible=False)

        gr.Examples(
            examples=[
                # Example paths relative to the project root
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0003.png", "assets/halfbody_demo/audio/chinese/fighting.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0033.png", "assets/halfbody_demo/audio/chinese/good.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0010.png", "assets/halfbody_demo/audio/chinese/news.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/1168.png", "assets/halfbody_demo/audio/chinese/no_smoking.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0057.png", "assets/halfbody_demo/audio/chinese/ultraman.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/man/0001.png", "assets/halfbody_demo/audio/chinese/echomimicv2_man.wav"],
                ["EMTD_dataset/ref_imgs_by_FLUX/woman/0077.png", "assets/halfbody_demo/audio/chinese/echomimicv2_woman.wav"],
            ],
            inputs=[image_input, audio_input],
            label="Preset Characters and Audio Examples",
        )

    generate_button.click(
        generate,
        inputs=[
            image_input,
            audio_input,
            pose_input,
            # Explicitly pass the global constants for these parameters
            # They are not user-adjustable in the current Gradio layout,
            # so we pass their defined values.
            width,          # Global constant
            height,         # Global constant
            length,         # Gradio Number input
            steps,          # Gradio Number input
            sample_rate,    # Global constant
            cfg,            # Global constant
            fps,            # Global constant
            context_frames, # Global constant
            context_overlap,# Global constant
            quantization_input, # Gradio Checkbox input
            seed            # Gradio Number input
        ],
        outputs=[video_output, seed_text],
    )

# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure current working directory is /kaggle/working/echomimic2
    # This is crucial for relative paths (./pretrained_weights/, ./outputs/, assets/...)
    current_working_directory = os.getcwd()
    expected_directory = "/kaggle/working/echomimic2"
    if not current_working_directory.endswith("echomimic2") and os.path.exists(expected_directory):
        print(f"Warning: Current working directory is not '{expected_directory}' ({current_working_directory}). Changing to it...")
        try:
            os.chdir(expected_directory)
            print(f"Changed to: {os.getcwd()}")
        except FileNotFoundError:
            print(f"Error: Directory '{expected_directory}' not found. Please ensure you have cloned the repo into it.")
            exit(1) # Exit if cannot change directory
    elif not os.path.exists(expected_directory):
        print(f"Error: The expected project directory '{expected_directory}' does not exist. Please ensure the repository is cloned correctly.")
        exit(1)

    print("Launching Gradio demo...")
    demo.queue()
    # Use share=True to get a public URL for Kaggle notebooks
    demo.launch(inbrowser=True, share=True)
