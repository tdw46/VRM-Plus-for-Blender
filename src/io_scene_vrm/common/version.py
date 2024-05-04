from dataclasses import dataclass
from sys import float_info
from typing import Optional

import bpy
from bpy.app.translations import pgettext

from .blender_manifest import BlenderManifest
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class Cache:
    use: bool
    last_blender_restart_required: bool
    last_blender_manifest_modification_time: float
    initial_blender_manifest_content: bytes


cache = Cache(
    use=False,
    last_blender_restart_required=False,
    last_blender_manifest_modification_time=0.0,
    initial_blender_manifest_content=BlenderManifest.default_blender_manifest_path().read_bytes(),
)


def clear_addon_version_cache() -> Optional[float]:  # pylint: disable=useless-return
    cache.use = False
    return None


def trigger_clear_addon_version_cache() -> None:
    if bpy.app.timers.is_registered(clear_addon_version_cache):
        return
    bpy.app.timers.register(clear_addon_version_cache, first_interval=0.5)


def max_supported_blender_major_minor_version() -> tuple[int, int]:
    blender_version_max = BlenderManifest.read().blender_version_max
    return (blender_version_max[0], blender_version_max[1])


def addon_version() -> tuple[int, int, int]:
    return BlenderManifest.read().version


def blender_restart_required() -> bool:
    if cache.use:
        return cache.last_blender_restart_required

    cache.use = True

    if cache.last_blender_restart_required:
        return True

    blender_manifest_path = BlenderManifest.default_blender_manifest_path()
    blender_manifest_modification_time = blender_manifest_path.stat().st_mtime
    if (
        abs(
            cache.last_blender_manifest_modification_time
            - blender_manifest_modification_time
        )
        < float_info.epsilon
    ):
        return False

    cache.last_blender_manifest_modification_time = blender_manifest_modification_time

    blender_manifest_content = blender_manifest_path.read_bytes()
    if blender_manifest_content == cache.initial_blender_manifest_content:
        return False

    cache.last_blender_restart_required = True
    return True


def stable_release() -> bool:
    return bpy.app.version_cycle in [
        "release",
        "rc",  # Windowsストアは3.3.11や3.6.3をRC版のままリリースしている
    ]


def supported() -> bool:
    return bpy.app.version[:2] <= max_supported_blender_major_minor_version()


def preferences_warning_message() -> Optional[str]:
    if blender_restart_required():
        return pgettext(
            "The VRM add-on has been updated."
            + " Please restart Blender to apply the changes."
        )
    if not stable_release():
        return pgettext(
            "VRM add-on is not compatible with Blender {blender_version_cycle}."
        ).format(blender_version_cycle=bpy.app.version_cycle.capitalize())
    if not supported():
        return pgettext(
            "The installed VRM add-on is not compatible with Blender {blender_version}."
            + " Please upgrade the add-on.",
        ).format(blender_version=".".join(map(str, bpy.app.version[:2])))
    return None


def panel_warning_message() -> Optional[str]:
    if blender_restart_required():
        return pgettext(
            "The VRM add-on has been\n"
            + "updated. Please restart Blender\n"
            + "to apply the changes."
        )
    if not stable_release():
        return pgettext(
            "VRM add-on is\n"
            + "not compatible with\n"
            + "Blender {blender_version_cycle}."
        ).format(blender_version_cycle=bpy.app.version_cycle.capitalize())
    if not supported():
        return pgettext(
            "The installed VRM add-\n"
            + "on is not compatible with\n"
            + "Blender {blender_version}. Please update."
        ).format(blender_version=".".join(map(str, bpy.app.version[:2])))
    return None


def validation_warning_message() -> Optional[str]:
    if blender_restart_required():
        return pgettext(
            "The VRM add-on has been updated."
            + " Please restart Blender to apply the changes."
        )
    if not stable_release():
        return pgettext(
            "VRM add-on is not compatible with Blender {blender_version_cycle}."
            + " The VRM may not be exported correctly.",
        ).format(blender_version_cycle=bpy.app.version_cycle.capitalize())
    if not supported():
        return pgettext(
            "The installed VRM add-on is not compatible with Blender {blender_version}."
            + " The VRM may not be exported correctly."
        ).format(blender_version=".".join(map(str, bpy.app.version)))
    return None
