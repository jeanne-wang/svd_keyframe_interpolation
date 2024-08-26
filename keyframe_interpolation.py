import os
import torch
import argparse
import copy
from diffusers.utils import load_image, export_to_video
from diffusers import UNetSpatioTemporalConditionModel
from custom_diffusers.pipelines.pipeline_frame_interpolation_with_noise_injection import FrameInterpolationWithNoiseInjectionPipeline
from custom_diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler
from attn_ctrl.attention_control import (AttentionStore, 
                                         register_temporal_self_attention_control, 
                                         register_temporal_self_attention_flip_control,
)

def main(args):

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipe = FrameInterpolationWithNoiseInjectionPipeline.from_pretrained(
        args.pretrained_model_name_or_path, 
        scheduler=noise_scheduler,
        variant="fp16",
        torch_dtype=torch.float16, 
    )
    ref_unet = pipe.ori_unet

    
    state_dict = pipe.unet.state_dict()
    # computing delta w
    finetuned_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        args.checkpoint_dir,
        subfolder="unet",
        torch_dtype=torch.float16,
    ) 
    assert finetuned_unet.config.num_frames==14
    ori_unet = UNetSpatioTemporalConditionModel.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        subfolder="unet",
        variant='fp16',
        torch_dtype=torch.float16,
    )
  
    finetuned_state_dict = finetuned_unet.state_dict()
    ori_state_dict = ori_unet.state_dict()
    for name, param in finetuned_state_dict.items():
        if 'temporal_transformer_blocks.0.attn1.to_v' in name or "temporal_transformer_blocks.0.attn1.to_out.0" in name:
            delta_w = param - ori_state_dict[name]
            state_dict[name] = state_dict[name] + delta_w
    pipe.unet.load_state_dict(state_dict)

    controller_ref= AttentionStore()
    register_temporal_self_attention_control(ref_unet, controller_ref)

    controller = AttentionStore()
    register_temporal_self_attention_flip_control(pipe.unet, controller, controller_ref)

    pipe = pipe.to(args.device)

    # run inference
    generator = torch.Generator(device=args.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    

    frame1 = load_image(args.frame1_path)
    frame1 = frame1.resize((1024, 576))

    frame2 = load_image(args.frame2_path)
    frame2 = frame2.resize((1024, 576))

    frames = pipe(image1=frame1, image2=frame2, 
                num_inference_steps=args.num_inference_steps, 
                generator=generator,
                weighted_average=args.weighted_average,
                noise_injection_steps=args.noise_injection_steps,
                noise_injection_ratio= args.noise_injection_ratio,
    ).frames[0]
    
    if args.out_path.endswith('.gif'):
        frames[0].save(args.out_path, save_all=True, append_images=frames[1:], duration=142, loop=0)
    else:
        export_to_video(frames, args.out_path, fps=7)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-video-diffusion-img2vid-xt")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument('--frame1_path', type=str, required=True)
    parser.add_argument('--frame2_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--weighted_average', action='store_true')
    parser.add_argument('--noise_injection_steps', type=int, default=0)
    parser.add_argument('--noise_injection_ratio', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()
    out_dir = os.path.dirname(args.out_path)
    os.makedirs(out_dir, exist_ok=True)
    main(args)
