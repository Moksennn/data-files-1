from diffusers.utils import load_image, make_image_grid
from diffusers import ControlNetUnionModel
from IPython.display import display
import torch
import json
import os

# Loading the path of the latest generated images
def load_last(filename, type):
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
            return data.get(type, None)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def controlnet_path_selector(path, type, base_path):
    last_generation_loading = os.path.join(f"{base_path}/Saved Parameters", "last_generation.json")
    try:
        if path == "inpaint":
            cn_path = load_last(last_generation_loading, 'inpaint')
        elif path == "controlnet":
            cn_path = load_last(last_generation_loading, 'controlnet')
        elif not path:
            cn_path = load_last(last_generation_loading, 'text2img')
        else:
            cn_path = path
        cn_image = load_image(cn_path)
        pipeline_type = "controlnet"
    except Exception as e:
        print(f"Couldn't load image. Reason: {e}")
        cn_image = ""
        pipeline_type = type
    return cn_image, pipeline_type

def flush(index, images, controlnets_scale, controlnet_modes):
    for element in [images, controlnets_scale, controlnet_modes]:
        element[index] = None

# ==========================
# MAIN CONTROLNET LOADER
# ==========================
def load(
    # Canny
    Canny,
    Canny_link,
    minimum_canny_threshold,
    maximum_canny_threshold,
    Canny_Strength,

    # Depth
    Depth_Map,
    Depthmap_Link,
    Depth_Strength,

    # OpenPose
    Open_Pose,
    Openpose_Link,
    Open_Pose_Strength,

    # Tile
    Tile,
    Tile_Link,
    Tile_Strength,

    # Lineart Anime
    Lineart_Anime,
    Lineart_Anime_Link,
    Lineart_Anime_Strength,

    # Shared
    controlnet,
    images,
    controlnets_scale,
    controlnet_modes,
    get_image_class,
):
    controlnet_weight = controlnet
    image = images
    controlnet_scale = controlnets_scale
    controlnet_mode = controlnet_modes

    # Cleanup disabled controlnets
    if not Canny and images[0]:
        flush(0, image, controlnet_scale, controlnet_mode)
    if not Depth_Map and images[1]:
        flush(1, image, controlnet_scale, controlnet_mode)
    if not Open_Pose and images[2]:
        flush(2, image, controlnet_scale, controlnet_mode)
    if not Tile and images[3]:
        flush(3, image, controlnet_scale, controlnet_mode)
    if not Lineart_Anime and images[4]:
        flush(4, image, controlnet_scale, controlnet_mode)

    # Load ControlNetUnion
    if any([Canny, Depth_Map, Open_Pose, Tile, Lineart_Anime]):
        if not controlnet:
            print("Loading ControlNetUnion...")
            controlnet_weight = ControlNetUnionModel.from_pretrained(
                "xinsir/controlnet-union-sdxl-1.0",
                torch_dtype=torch.float16
            )

        # -------- CANNY --------
        if Canny and Canny_link is not None:
            canny_image = get_image_class.get_canny(
                Canny_link,
                minimum_canny_threshold,
                maximum_canny_threshold
            )
            image[0] = canny_image.resize((1024, 1024))
            controlnet_scale[0] = Canny_Strength
            controlnet_mode[0] = 3  # canny / lineart / mlsd

            display(make_image_grid([Canny_link, canny_image], 1, 2))

        # -------- DEPTH --------
        if Depth_Map and Depthmap_Link is not None:
            depth_image = Depthmap_Link.resize((1024, 1024))
            depth_map = get_image_class.get_depth(depth_image).unsqueeze(0).half().to("cpu")

            image[1] = depth_map
            controlnet_scale[1] = Depth_Strength
            controlnet_mode[1] = 1  # depth

            display(make_image_grid([
                Depthmap_Link,
                get_image_class.get_depth(depth_image, "display")
            ], 1, 2))

        # -------- OPENPOSE --------
        if Open_Pose and Openpose_Link is not None:
            openpose_image = get_image_class.get_openpose(Openpose_Link)

            image[2] = openpose_image.resize((1024, 1024))
            controlnet_scale[2] = Open_Pose_Strength
            controlnet_mode[2] = 0  # openpose

            display(make_image_grid([Openpose_Link, openpose_image], 1, 2))

        # -------- TILE --------
        if Tile and Tile_Link is not None:
            tile_image = Tile_Link.resize((1024, 1024))

            image[3] = tile_image
            controlnet_scale[3] = Tile_Strength
            controlnet_mode[3] = 4  # tile

            display(make_image_grid([Tile_Link, tile_image], 1, 2))

        # -------- LINEART ANIME --------
        if Lineart_Anime and Lineart_Anime_Link is not None:
            lineart_image = get_image_class.get_lineart_anime(Lineart_Anime_Link)

            image[4] = lineart_image.resize((1024, 1024))
            controlnet_scale[4] = Lineart_Anime_Strength
            controlnet_mode[4] = 5  # anime_lineart

            display(make_image_grid([Lineart_Anime_Link, lineart_image], 1, 2))

    return controlnet_weight, image, controlnet_scale, controlnet_mode
