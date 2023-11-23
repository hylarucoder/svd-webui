import math
import os
from glob import glob
from typing import Optional

import cv2
import gradio as gr
import numpy as np
import torch
import typer
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from sgm.util import instantiate_from_config


def cli(
    image: Image,
    num_frames: int,
    num_steps: int,
    checkpoint: str,
    fps_id: int,
    motion_bucket_id,
    cond_aug,
    seed,
    decoding_t,
    progress: gr.Progress,
    device: str = "cuda",
):
    progress(0.01, "Processing Image")
    model_config = f"configs/{checkpoint}.yaml"
    output_folder = "./outputs"
    if not image:
        raise ValueError("Something went wrong")
    if image.mode == "RGBA":
        image = image.convert("RGB")
    w, h = image.size

    if h % 64 != 0 or w % 64 != 0:
        width, height = map(lambda x: x - x % 64, (w, h))
        image = image.resize((width, height))
        print(
            f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
        )

    image = ToTensor()(image)
    image = image * 2.0 - 1.0

    image = image.unsqueeze(0).to(device)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    shape = (num_frames, C, H // F, W // F)
    if (H, W) != (576, 1024):
        print(
            "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
        )

    progress(0.02, "Download model")
    if checkpoint not in ["svd", "svd_image_decoder", "svd_xt", "svd_xt_image_decoder"]:
        raise ValueError("Invalid checkpoint")
    ckpt_dir = get_ckpt_dir()
    if checkpoint in ["svd", "svd_image_decoder"]:
        download_hf_model("stabilityai/stable-video-diffusion-img2vid", ckpt_dir, checkpoint)
    if checkpoint in ["svd", "svd_image_decoder"]:
        download_hf_model("stabilityai/stable-video-diffusion-img2vid-xt", ckpt_dir, checkpoint)

    progress(0.03, "Loading model")
    model = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        lowvram_mode=True,
    )
    torch.manual_seed(seed)

    if motion_bucket_id > 255:
        print("WARNING: High motion bucket! This may lead to suboptimal performance.")

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")

    value_dict = {
        "motion_bucket_id": motion_bucket_id,
        "fps_id": fps_id,
        "cond_aug": cond_aug,
        "cond_frames_without_noise": image,
        "cond_frames": image + cond_aug * torch.randn_like(image),
    }

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model.conditioner),
                value_dict,
                [1, num_frames],
                T=num_frames,
                device=device,
            )
            c, uc = model.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)

            randn = torch.randn(shape, device=device)

            additional_model_inputs = {
                "image_only_indicator": torch.zeros(2, num_frames).to(device),
                "num_video_frames": batch["num_video_frames"],
            }

            def denoiser(input, sigma, c):
                return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

            samples_z = model.sampler(denoiser, randn, cond=c, uc=uc)
            model.en_and_decode_n_samples_a_time = decoding_t
            samples_x = model.decode_first_stage(samples_z)
            samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

            os.makedirs(output_folder, exist_ok=True)
            base_count = len(glob(os.path.join(output_folder, "*.mp4")))
            video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"MP4V"),
                fps_id + 1,
                (samples.shape[-1], samples.shape[-2]),
            )

            vid = (rearrange(samples, "t c h w -> t h w c") * 255).cpu().numpy().astype(np.uint8)
            for frame in vid:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(frame)
            writer.release()

        return video_path


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = torch.tensor([value_dict["fps_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "motion_bucket_id":
            batch[key] = torch.tensor([value_dict["motion_bucket_id"]]).to(device).repeat(int(math.prod(N)))
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_ckpt_dir():
    return os.environ.get("SVD_CKPT_PATH", "models/checkpoints/")


def download_hf_model(repo_id, local_dir, f):
    from huggingface_hub import hf_hub_download

    hf_hub_download(
        repo_id=repo_id,
        filename=f,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    lowvram_mode: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
    config.model.params.ckpt_path = get_ckpt_dir() + config.model.params.ckpt_path

    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    if lowvram_mode:
        model.model.half()
    return model


if __name__ == "__main__":
    typer.run(cli)
