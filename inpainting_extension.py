import os
from PIL import Image, ImageFilter, ImageOps
import ipywidgets as widgets

class InpaintingExtension:
    def __init__(self):
        # Default values
        self.mask_blur = 8
        self.outpaint_pixels = 0
        self.invert_mask = False
        self.override_strength = None

        # ---------- UI ----------
        self.blur_slider = widgets.IntSlider(
            value=self.mask_blur,
            min=0,
            max=64,
            step=1,
            description="Mask Blur",
            continuous_update=False
        )

        self.outpaint_slider = widgets.IntSlider(
            value=self.outpaint_pixels,
            min=0,
            max=512,
            step=8,
            description="Outpaint px",
            continuous_update=False
        )

        self.invert_checkbox = widgets.Checkbox(
            value=self.invert_mask,
            description="Invert Mask"
        )

        self.ui = widgets.VBox([
            widgets.HTML("<b>Inpainting Enhancement</b>"),
            self.blur_slider,
            self.outpaint_slider,
            self.invert_checkbox,
        ])

        # Sync UI → logic
        self.blur_slider.observe(self._sync, names="value")
        self.outpaint_slider.observe(self._sync, names="value")
        self.invert_checkbox.observe(self._sync, names="value")

    # -------------------------
    def _sync(self, change=None):
        self.mask_blur = self.blur_slider.value
        self.outpaint_pixels = self.outpaint_slider.value
        self.invert_mask = self.invert_checkbox.value

    # -------------------------
    def process(self, value_list, inpaint_ui):
        try:
            new_values = list(value_list)

            image_path = new_values[3]
            mask_path = new_values[4]

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # Mask processing
            if self.mask_blur > 0:
                mask = mask.filter(ImageFilter.GaussianBlur(self.mask_blur))

            if self.invert_mask:
                mask = ImageOps.invert(mask)

            # Outpainting
            if self.outpaint_pixels > 0:
                image, mask = self._outpaint(image, mask, self.outpaint_pixels)

            out_dir = "/content/inpaint_ext"
            os.makedirs(out_dir, exist_ok=True)

            new_img = os.path.join(out_dir, "image.png")
            new_mask = os.path.join(out_dir, "mask.png")

            image.save(new_img)
            mask.save(new_mask)

            new_values[3] = new_img
            new_values[4] = new_mask

            return new_values

        except Exception as e:
            print("[InpaintingExtension] fallback:", e)
            return value_list

    # -------------------------
    def _outpaint(self, image, mask, px):
        w, h = image.size
        new_w = w + px * 2
        new_h = h + px * 2

        new_image = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        new_mask = Image.new("L", (new_w, new_h), 255)

        new_image.paste(image, (px, px))
        new_mask.paste(mask, (px, px))

        return new_image, new_mask
