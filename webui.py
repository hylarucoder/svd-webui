import os

import gradio as gr
import pydantic as pt

from svd_webui.cli import cli


def group_by_n(l, n):
    for i in range(0, len(l), n):
        yield l[i: i + n]


class TProjectSetting(pt.BaseModel):
    ...


class Preset(pt.BaseModel):
    name: str
    frames: int = 14
    steps: int = 20
    checkpoint: str = "svd"
    fps: int = 6
    motion_bucket: int = 127
    cond_aug: int = 0.02
    seed: int = 23
    decoded_at_once: int = 1
    device: str = "cuda"


preset_default = Preset(
    name="SVD",
    checkpoint="svd",
    frames=14,
    steps=25,
)

presets = [
    preset_default,
    Preset(
        name="SVD XT",
        checkpoint="svd_xt",
        frames=25,
        steps=30,
    ),
    Preset(
        name="SVD Image Decoder",
        checkpoint="svd_image_decoder",
        frames=14,
        steps=25,
    ),
    Preset(
        name="SVD XT Image Decoder",
        checkpoint="svd_xt_image_decoder",
        frames=25,
        steps=30,
    ),
]


def get_presets():
    return [preset.name for preset in presets]


def render_ui():
    # main render
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column():
                    ip_preset = gr.Dropdown(
                        label="Preset",
                        choices=get_presets(),
                        value=preset_default.name,
                        interactive=True,
                    )
                with gr.Column():
                    ...

        with gr.Column(scale=1):
            ...
            # btn_new_project = gr.Button("New", variant="secondary")

    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tab(label="Preview"):
                preview_video = gr.Video(height=504, show_label=False, label="Preview", interactive=False)

            with gr.Row():
                with gr.Column(scale=3):
                    ip_input_image = gr.Image(type="pil", label="Input Image")
                with gr.Column(scale=1):
                    generate_button = gr.Button(value="Generate", visible=True)
                    stop_button = gr.Button(
                        "Stop TODO: not implemented yet",
                        interactive=False,
                    )

        with gr.Column(scale=1):
            with gr.Tab(label="Setting"):
                with gr.Row():
                    ip_checkpoint = gr.Dropdown(
                        choices=[
                            "svd",
                            "svd_xt",
                            "svd_image_decoder",
                            "svd_xt_image_decoder",
                        ],
                        label="Checkpoint",
                        value=preset_default.checkpoint,
                        allow_custom_value=True,
                        interactive=True,
                    )
                with gr.Row():
                    with gr.Group():
                        ip_frames = gr.Number(
                            minimum=12,
                            maximum=36,
                            label="Frames",
                            value=preset_default.frames,
                            precision=0,
                            interactive=True,
                        )
                        ip_fps = gr.Number(
                            minimum=5,
                            maximum=30,
                            label="FPS",
                            value=preset_default.fps,
                            precision=0,
                            interactive=True,
                        )
                with gr.Row():
                    ip_steps = gr.Number(
                        minimum=20,
                        maximum=40,
                        label="Steps",
                        value=preset_default.steps,
                        precision=0,
                        interactive=True,
                    )

                with gr.Row():
                    ip_cond_aug = gr.Number(
                        preset_default.cond_aug,
                        minimum=0.0,
                        maximum=1.0,
                        label="Cond Aug",
                        value=preset_default.cond_aug,
                        precision=2,
                        interactive=True,
                    )
                with gr.Row():
                    ip_motion_bucket = gr.Number(
                        preset_default.motion_bucket,
                        minimum=10,
                        maximum=255,
                        label="Motion bucket",
                        value=preset_default.motion_bucket,
                        precision=2,
                        interactive=True,
                    )
                    ip_decoded_at_once = gr.Number(
                        preset_default.decoded_at_once,
                        minimum=1,
                        maximum=10,
                        label="Decoded at once (NOTE: Increase may lead to VRAM OOM if you don't have enough )",
                        value=preset_default.decoded_at_once,
                        precision=2,
                        interactive=True,
                    )
                with gr.Row():
                    ip_seed = gr.Number(label="Seed", value=preset_default.seed, precision=0, interactive=True)

    def fn_generate(
            image,
            frames,
            steps,
            cond_aug,
            checkpoint,
            fps,
            motion_bucket,
            seed,
            decoded_at_once,
            data=None,
            progress=gr.Progress(
                track_tqdm=True,
            ),
    ):
        p = cli(
            image=image,
            num_frames=frames,
            num_steps=steps,
            checkpoint=checkpoint,
            fps_id=fps,
            motion_bucket_id=motion_bucket,
            cond_aug=cond_aug,
            seed=seed,
            decoding_t=decoded_at_once,
            progress=progress,
        )
        return p

    generate_button.click(
        lambda: (gr.update(visible=True, interactive=True), gr.update(visible=False)),
        inputs=[],
        outputs=[
            stop_button,
            generate_button,
        ],
    ).then(
        fn_generate,
        inputs=[
            ip_input_image,
            ip_frames,
            ip_steps,
            ip_cond_aug,
            ip_checkpoint,
            ip_fps,
            ip_motion_bucket,
            ip_seed,
            ip_decoded_at_once,
        ],
        outputs=[preview_video],
    ).then(
        lambda: (gr.update(visible=False, interactive=False), gr.update(visible=True)),
        inputs=[],
        outputs=[
            stop_button,
            generate_button,
        ],
    )

    def apply_preset(
            preset_name,
    ):
        preset = next((_ for _ in presets if _.name == preset_name), None)
        return (
            gr.update(
                value=preset.frames,
            ),
            gr.update(
                value=preset.steps,
            ),
            gr.update(
                value=preset.checkpoint,
            ),
            gr.update(
                value=preset.fps,
            ),
            gr.update(
                value=preset.motion_bucket,
            ),
            gr.update(
                value=preset.seed,
            ),
            gr.update(
                value=preset.decoded_at_once,
            ),
        )

    ip_preset.change(
        apply_preset,
        inputs=[
            ip_preset,
        ],
        outputs=[
            ip_frames,
            ip_steps,
            ip_checkpoint,
            ip_fps,
            ip_motion_bucket,
            ip_seed,
            ip_decoded_at_once,
        ],
    )


with gr.Blocks(
        title="SVD WebUI",
        css="""
        video {
            height: 504px !important;
        }
        """,
        theme=gr.themes.Default(
            spacing_size="sm",
            text_size="sm",
        ),
) as demo:
    render_ui()

server_port = 7860
if os.environ.get("SVD_PORT"):
    server_port = int(os.environ.get("SVD_PORT"))
demo.launch(server_name="0.0.0.0", server_port=server_port)
