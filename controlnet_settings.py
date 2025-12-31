from enum import IntEnum
import os
import json
from io import BytesIO
from PIL import Image

import ipywidgets as widgets
from diffusers.utils import load_image
from diffusers import ControlNetUnionModel
from IPython.display import display

from StableDiffusionXLColabUI.utils.get_controlnet_image import ControlNetImage
from StableDiffusionXLColabUI.utils import controlnet_loader, generate_prompt


# =========================================================
# ENUM DEFINITIONS
# =========================================================

class ControlNetType(IntEnum):
    OPENPOSE = 0
    DEPTH = 1
    CANNY = 2
    TILE = 3
    LINEART_ANIME = 4


CONTROLNET_MODE = {
    ControlNetType.OPENPOSE: 0,
    ControlNetType.DEPTH: 1,
    ControlNetType.CANNY: 3,
    ControlNetType.TILE: 4,
    ControlNetType.LINEART_ANIME: 5,
}


# =========================================================
# UTILS
# =========================================================

def load_param(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# =========================================================
# CONTROLNET SETTINGS UI
# =========================================================

class ControlNetSettings:

    # -----------------------------------------------------
    # INIT
    # -----------------------------------------------------
    def __init__(self, cfg, ideas_line, gpt2_pipe, base_path):
        self.base_path = base_path
        self.cn = ControlNetImage()
        self.cn.load_pipe()

        self.controlnet_dropdown_choice = [
            "Link",
            "Upload",
            "Last Generated Text2Img",
            "Last Generated ControlNet",
            "Last Generated Inpainting",
        ]

        self._init_prompt(cfg, ideas_line, gpt2_pipe)
        self._init_generation(cfg)
        self._init_scheduler(cfg)
        self._init_vae(cfg)
        self._init_controlnets()

    # -----------------------------------------------------
    # PROMPT
    # -----------------------------------------------------
    def _init_prompt(self, cfg, ideas_line, gpt2_pipe):
        layout = widgets.Layout(width="50%")

        self.prompt_widget = widgets.Textarea(value=cfg[0] if cfg else "", layout=layout)
        self.negative_prompt_widget = widgets.Textarea(value=cfg[1] if cfg else "", layout=layout)

        rand_btn = widgets.Button(description="🔄")
        rand_btn.on_click(
            lambda b: self._generate_prompt(ideas_line, gpt2_pipe)
        )

        self.prompts_section = widgets.VBox([
            widgets.HBox([self.prompt_widget, self.negative_prompt_widget]),
            widgets.HBox([rand_btn, widgets.Label("Generate / continue prompt")])
        ])

    def _generate_prompt(self, ideas_line, gpt2_pipe):
        self.prompt_widget.value = generate_prompt.generate(
            self.prompt_widget.value, ideas_line, gpt2_pipe
        )

    # -----------------------------------------------------
    # GENERATION
    # -----------------------------------------------------
    def _init_generation(self, cfg):
        self.width_slider = widgets.IntSlider(512, 512, 1536, 64)
        self.height_slider = widgets.IntSlider(512, 512, 1536, 64)
        self.steps = widgets.IntText(value=cfg[5] if cfg else 12)
        self.scale = widgets.FloatSlider(1, 1, 12, 0.1)
        self.batch_size = widgets.IntText(value=1)

        self.generation_parameter_section = widgets.VBox([
            widgets.HBox([self.width_slider, self.height_slider]),
            widgets.HBox([self.steps, self.batch_size]),
            self.scale,
        ])

    # -----------------------------------------------------
    # SCHEDULER
    # -----------------------------------------------------
    def _init_scheduler(self, cfg):
        self.scheduler_dropdown = widgets.Dropdown(
            options=["Default", "Euler", "Euler a", "DPM++ 2M"],
            value="Default"
        )
        self.scheduler_settings = widgets.VBox([self.scheduler_dropdown])

    # -----------------------------------------------------
    # VAE
    # -----------------------------------------------------
    def _init_vae(self, cfg):
        self.vae_link = widgets.Text(description="VAE")
        self.vae_section = widgets.HBox([self.vae_link])

    # -----------------------------------------------------
    # CONTROLNET UI FACTORY
    # -----------------------------------------------------
    def _make_controlnet_ui(self, label, cn_type):
        toggle = widgets.Checkbox(description=f"Enable {label}")
        link = widgets.Text(description=f"{label} Link")
        upload = widgets.FileUpload(accept="image/*", multiple=False)
        strength = widgets.FloatSlider(0.1, 0.1, 1, 0.1, description="Strength")
        preview_btn = widgets.Button(description="Preview")
        output = widgets.Output()

        def upload_handler(change):
            os.makedirs(f"/content/{label}", exist_ok=True)
            for f in upload.value.values():
                path = f"/content/{label}/temp.png"
                with open(path, "wb") as fp:
                    fp.write(f["content"])
                link.value = path

        upload.observe(upload_handler, names="value")
        preview_btn.on_click(lambda b: self.preview(link.value, cn_type, output))

        box = widgets.VBox([output, toggle])
        toggle.observe(
            lambda c: setattr(
                box,
                "children",
                [output, toggle, link, upload, strength, preview_btn]
                if c["new"] else [output, toggle]
            ),
            names="value"
        )

        return {
            "type": cn_type,
            "toggle": toggle,
            "link": link,
            "strength": strength,
            "ui": box,
        }

    # -----------------------------------------------------
    # CONTROLNETS
    # -----------------------------------------------------
    def _init_controlnets(self):
        self.controlnets = {
            ControlNetType.CANNY: self._make_controlnet_ui("Canny", ControlNetType.CANNY),
            ControlNetType.DEPTH: self._make_controlnet_ui("Depth", ControlNetType.DEPTH),
            ControlNetType.OPENPOSE: self._make_controlnet_ui("OpenPose", ControlNetType.OPENPOSE),
            ControlNetType.TILE: self._make_controlnet_ui("Tile", ControlNetType.TILE),
            ControlNetType.LINEART_ANIME: self._make_controlnet_ui("Lineart_Anime", ControlNetType.LINEART_ANIME),
        }

        self.controlnet_selections = widgets.VBox(
            [widgets.HTML("<hr>")] +
            [cn["ui"] for cn in self.controlnets.values()] +
            [widgets.HTML("<hr>")]
        )

    # -----------------------------------------------------
    # PREVIEW
    # -----------------------------------------------------
    def preview(self, path, cn_type, output):
        output.clear_output()
        try:
            img = load_image(path)

            if cn_type == ControlNetType.CANNY:
                img = self.cn.get_canny(img, 100, 240)
            elif cn_type == ControlNetType.DEPTH:
                img = self.cn.get_depth(img, "display")
            elif cn_type == ControlNetType.OPENPOSE:
                img = self.cn.get_openpose(img)
            elif cn_type == ControlNetType.LINEART_ANIME:
                img = self.cn.get_lineart_anime(img)

            display(img.resize((256, 256)))
        except Exception as e:
            with output:
                print("Preview error:", e)

    # -----------------------------------------------------
    # COLLECT VALUES
    # -----------------------------------------------------
    def collect_controlnet(self):
        enabled, links, strengths = {}, {}, {}
        for k, cn in self.controlnets.items():
            enabled[k] = cn["toggle"].value
            links[k] = cn["link"].value
            strengths[k] = cn["strength"].value
        return enabled, links, strengths

    # -----------------------------------------------------
    # WRAP ALL
    # -----------------------------------------------------
    def wrap_settings(self):
        return widgets.VBox([
            self.prompts_section,
            self.generation_parameter_section,
            self.scheduler_settings,
            self.vae_section,
            self.controlnet_selections,
        ])
