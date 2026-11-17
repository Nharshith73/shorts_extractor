"""Pure filter-graph construction for fixed 1080x1920 output."""

from dataclasses import dataclass
import logging

from .config import RenderConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LayoutGraph:
    graph: str
    effective_layout: str


def _blurred_panel(height: int, prefix: str) -> str:
    return (f"[{prefix}bg]scale=1080:{height}:force_original_aspect_ratio=increase,crop=1080:{height},"
            f"gblur=sigma=30,eq=brightness=-0.15[{prefix}background];"
            f"[{prefix}fg]scale=1080:{height}:force_original_aspect_ratio=decrease[{prefix}foreground];"
            f"[{prefix}background][{prefix}foreground]overlay=(W-w)/2:(H-h)/2[{prefix}panel]")


def build_layout(config: RenderConfig, width: int, height: int, captions: bool) -> LayoutGraph:
    layout = config.layout
    region = config.facecam_region
    if layout == "stacked" and region is None:
        LOGGER.warning("stacked layout has no facecam_region; falling back to blurred_background")
        layout = "blurred_background"
    if layout == "blurred_background":
        graph = "[0:v:0]split=2[gamebg][gamefg];" + _blurred_panel(1920, "game") + ";[gamepanel]null[layout]"
    elif layout == "stacked":
        assert region is not None
        if region.x + region.w > width or region.y + region.h > height:
            raise ValueError("facecam_region is outside the source frame")
        graph = ("[0:v:0]split=3[gamebg][gamefg][cam];" + _blurred_panel(1080, "game") + ";"
                 f"[cam]crop={region.w}:{region.h}:{region.x}:{region.y},"
                 "scale=1080:840:force_original_aspect_ratio=increase,crop=1080:840[facecam];"
                 "[gamepanel][facecam]vstack=inputs=2[layout]")
    else:
        graph = ("[0:v:0]scale=1080:1920:force_original_aspect_ratio=increase,"
                 f"crop=1080:1920:x='max(0,min(iw-1080,iw*{config.crop_center_x}-540))':y=(ih-1920)/2[layout]")
    graph += ";[layout]fps=30,setsar=1,format=yuv420p"
    if captions:
        # These are fixed relative names. No Windows drive/path enters the filter.
        graph += ",subtitles=filename=captions.ass:fontsdir=fonts"
    return LayoutGraph(graph + "[video]", layout)
