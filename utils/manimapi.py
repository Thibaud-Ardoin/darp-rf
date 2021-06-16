import manimlib
import os
from pathlib import Path
import colour
import copy

from manimlib.constants import (
    LOW_QUALITY_CAMERA_CONFIG,
    MEDIUM_QUALITY_CAMERA_CONFIG,
    HIGH_QUALITY_CAMERA_CONFIG,
    PRODUCTION_QUALITY_CAMERA_CONFIG
)

def _resolve_path(p : Path) -> str:
    return str(p.resolve())

def create_camera_config(height : int, width : int =None, fps : int = 30, bg_color=None):
    if width is None:
        width = 16 * height // 9

    return {
        "pixel_height" : int(height),
        "pixel_width" : int(width),
        "frame_rate" : int(fps),
        "background_color" : colour.Color(bg_color)
    }

def create_output_path_config(video_dir : Path, tex_dir : Path, text_dir : Path):
    video_dir = Path(video_dir)
    tex_dir = Path(tex_dir)
    text_dir = Path(text_dir)
    return {
        "video_dir" : _resolve_path(video_dir),
        "tex_dir" : _resolve_path(tex_dir),
        "text_dir" : _resolve_path(text_dir)
    }

_DEFAULT_MEDIA_DIR = Path("media")
DEFAULT_PATH_CONFIG = create_output_path_config(_DEFAULT_MEDIA_DIR / "videos",
                                                _DEFAULT_MEDIA_DIR / "Tex",
                                                _DEFAULT_MEDIA_DIR / "texts")

def create_config(scene_group_name,
             leave_progbars=False,
             transparent=False,
             last_frame_only=False,
             ignore_waits=False,
             camera_config=LOW_QUALITY_CAMERA_CONFIG,
             output_config=DEFAULT_PATH_CONFIG
             ):
    if transparent:
        camera_config['background_opacity'] = 0

    file_writer_config = {
        # "write_to_movie": self.write_to_movie or not self.save_last_frame,
        "write_to_movie": not last_frame_only,
        "save_last_frame": last_frame_only,
        "save_pngs": False,  # doesn't seem to have an affect
        "save_as_gif": False,  # appears to be not implemented
        "png_mode": "RGBA" if transparent else "RGB",
        "movie_file_extension": ".mov" if transparent else ".mp4",
        "file_name": None,
        "input_file_path": None,
        "output_directory": scene_group_name
    }
    config = {
        "module": None,
        "scene_names": [],
        "open_video_upon_completion": False,
        "show_file_in_finder": False,
        "file_writer_config": file_writer_config,
        "quiet": True,
        "ignore_waits": ignore_waits,
        "write_all": True,
        "start_at_animation_number": None,
        "end_at_animation_number": None,
        # "sound": self.sound,
        "leave_progress_bars": leave_progbars,
        "media_dir": None,
        "video_output_dir": None,
        "camera_config": camera_config,
        "skip_animations": last_frame_only
    }
    config.update(output_config)
    return config


def _initialize_directories(config):
    manimlib.constants.TEX_DIR = config["tex_dir"]
    manimlib.constants.TEXT_DIR = config["text_dir"]
    manimlib.constants.VIDEO_DIR = config["video_dir"]

    for folder in [manimlib.constants.VIDEO_DIR, manimlib.constants.VIDEO_OUTPUT_DIR,
                   manimlib.constants.TEX_DIR, manimlib.constants.TEXT_DIR]:
        if folder != "" and not os.path.exists(folder):
            os.makedirs(folder)

def render(scenes, config, scene_names=None, name_getter=None):
    if scene_names is not None:
        scene_names = list(scene_names)

    _initialize_directories(config)

    scene_kwargs_template = dict([
        (key, config[key])
        for key in [
            "camera_config",
            "file_writer_config",
            "skip_animations",
            "start_at_animation_number",
            "end_at_animation_number",
            "leave_progress_bars",
        ]
    ])
    output_files = []
    for idx, SceneClass in enumerate(scenes):
        if scene_names is not None:
            name = scene_names[idx]
        elif name_getter is not None:
            name = name_getter(SceneClass)
        else:
            name = None

        scene_kw = copy.deepcopy(scene_kwargs_template)
        scene_kw['file_writer_config']['file_name'] = name

        scene_files = {}

        # By invoking, this renders the full scene
        scene = SceneClass(**scene_kw)
        if scene.file_writer.save_last_frame:
            scene_files['image'] = scene.file_writer.image_file_path
        if scene.file_writer.write_to_movie:
            scene_files['movie'] = scene.file_writer.movie_file_path
        if scene.file_writer.save_as_gif:
            scene_files['gif'] = scene.file_writer.gif_file_path

        output_files.append(scene_files)

    return output_files


def cli():
    import example_scenes
    scenes = [
        example_scenes.WarpSquare,
        example_scenes.SquareToCircle,
        example_scenes.WriteStuff
    ]
    config = create_config("test", camera_config=HIGH_QUALITY_CAMERA_CONFIG)
    render(scenes, config)



if __name__ == '__main__':
    cli()