# SPDX-License-Identifier: MIT OR GPL-3.0-or-later
import itertools
import struct
from os import environ
from pathlib import Path
from typing import Optional

from bpy.types import Armature, Context, Object, PoseBone
from mathutils import Euler, Matrix, Quaternion, Vector

from ..common import version
from ..common.convert import Json
from ..common.deep import make_json
from ..common.gl import GL_FLOAT
from ..common.gltf import pack_glb
from ..common.logging import get_logger
from ..common.rotation import (
    ROTATION_MODE_AXIS_ANGLE,
    ROTATION_MODE_EULER,
    ROTATION_MODE_QUATERNION,
    get_rotation_as_quaternion,
)
from ..common.vrm1.human_bone import HumanBoneName, HumanBoneSpecification
from ..common.workspace import save_workspace
from ..editor.extension import get_armature_extension
from ..editor.t_pose import setup_humanoid_t_pose
from ..editor.vrm1.property_group import Vrm1PropertyGroup

logger = get_logger(__name__)


class VrmAnimationExporter:
    @staticmethod
    def execute(context: Context, path: Path, armature: Object) -> set[str]:
        armature_data = armature.data
        if not isinstance(armature_data, Armature):
            return {"CANCELLED"}

        with (
            setup_humanoid_t_pose(context, armature),
            save_workspace(context, armature, mode="POSE"),
        ):
            output_bytes = export_vrm_animation(context, armature)

        path.write_bytes(output_bytes)
        return {"FINISHED"}


def connect_humanoid_node_dicts(
    human_bone_specification: HumanBoneSpecification,
    human_bone_name_to_node_dict: dict[HumanBoneName, dict[str, Json]],
    parent_node_dict: Optional[dict[str, Json]],
    node_dicts: list[dict[str, Json]],
    human_bone_name_to_node_index: dict[HumanBoneName, int],
) -> None:
    current_node_dict = human_bone_name_to_node_dict.get(human_bone_specification.name)
    if isinstance(current_node_dict, dict):
        node_index = len(node_dicts)
        human_bone_name_to_node_index[human_bone_specification.name] = node_index
        node_dicts.append(current_node_dict)
        if parent_node_dict is not None:
            children = parent_node_dict.get("children")
            if not isinstance(children, list):
                children = []
                parent_node_dict["children"] = children
            children.append(node_index)
        parent_node_dict = current_node_dict
    for child in human_bone_specification.children():
        connect_humanoid_node_dicts(
            child,
            human_bone_name_to_node_dict,
            parent_node_dict,
            node_dicts,
            human_bone_name_to_node_index,
        )


def create_node_dicts(
    bone: PoseBone,
    parent_bone: Optional[PoseBone],
    node_dicts: list[dict[str, Json]],
    bone_name_to_node_index: dict[str, int],
    armature_world_matrix: Matrix,
    bone_name_to_world_rest_matrix: dict[
        str, Matrix
    ],  # Holds each bone's rest WS matrix
) -> int:
    bone_world_matrix = armature_world_matrix @ bone.matrix

    # Store this bone's world-space rest transform for consistent export
    bone_name_to_world_rest_matrix[bone.name] = bone_world_matrix.copy()

    node_index = len(node_dicts)
    node_dict: dict[str, Json] = {"name": bone.name}
    node_dicts.append(node_dict)
    bone_name_to_node_index[bone.name] = node_index

    # -------------------------------------------------------
    # FIX for local transform so it matches how the importer
    # expects to read translation/rotation as a local matrix.
    if parent_bone:
        parent_world = armature_world_matrix @ parent_bone.matrix
        local_matrix = parent_world.inverted_safe() @ bone_world_matrix

        # If this child bone is "disconnected" (use_connect=False),
        # preserve its original offset from parent's tail
        arm_data = parent_bone.id_data  # The Armature data block
        if not arm_data.data.bones[bone.name].use_connect:
            edit_parent = arm_data.data.bones[parent_bone.name]
            edit_child = arm_data.data.bones[bone.name]

            # The parent's tail is at (0, length, 0) in its own space
            parent_tail = Vector((0.0, edit_parent.length, 0.0))

            # Child's head position relative to parent's head in parent's space
            child_head = edit_parent.matrix_local.inverted() @ edit_child.head_local

            # The offset we need is the negative of what we had before
            # This makes it relative to tail instead of being mirrored around the head
            offset = -(child_head - parent_tail)

            # Add parent_tail to shift the reference point from head to tail
            offset -= parent_tail

            # Add this offset to preserve the original spacing
            local_matrix.translation += offset
    else:
        # Root bone local relative to the armature object's own world transform
        local_matrix = armature_world_matrix.inverted_safe() @ bone_world_matrix

        # Invert the transfrom direction
        local_matrix.translation *= -1
    # -------------------------------------------------------

    translation = local_matrix.to_translation()
    rotation = local_matrix.to_quaternion()

    node_dict["translation"] = [
        translation.x,
        translation.z,
        -translation.y,
    ]
    # ---- FIX: corrected quaternion axis mapping below ----
    node_dict["rotation"] = [
        rotation.x,
        rotation.z,
        -rotation.y,
        rotation.w,
    ]

    # We keep the existing scale logic if needed:
    local_scale = local_matrix.to_scale()
    if local_scale != Vector((1.0, 1.0, 1.0)):
        node_dict["scale"] = [
            local_scale.x,
            local_scale.z,
            local_scale.y,
        ]

    children = [
        create_node_dicts(
            child_bone,
            bone,
            node_dicts,
            bone_name_to_node_index,
            armature_world_matrix,
            bone_name_to_world_rest_matrix,  # pass along
        )
        for child_bone in bone.children
    ]
    if children:
        node_dict["children"] = make_json(children)

    return node_index


def export_vrm_animation(context: Context, armature: Object) -> bytes:
    armature_data = armature.data
    if not isinstance(armature_data, Armature):
        message = "Armature data is not an Armature"
        raise TypeError(message)
    vrm1 = get_armature_extension(armature_data).vrm1
    human_bones = vrm1.humanoid.human_bones

    node_dicts: list[dict[str, Json]] = []
    bone_name_to_node_index: dict[str, int] = {}
    bone_name_to_base_quaternion: dict[str, Quaternion] = {}
    bone_name_to_world_rest_matrix: dict[str, Matrix] = {}
    scene_node_indices: list[int] = [0]
    data_path_to_bone_and_property_name: dict[str, tuple[PoseBone, str]] = {}

    root_node_translation = armature.matrix_world.to_translation()
    root_node_rotation = armature.matrix_world.to_quaternion()
    root_node_scale = armature.matrix_world.to_scale()
    root_node_dict: dict[str, Json] = {
        "name": armature.name,
        "translation": [
            root_node_translation.x,
            root_node_translation.z,
            -root_node_translation.y,
        ],
        # ---- FIX: corrected quaternion axis mapping below ----
        "rotation": [
            root_node_rotation.x,
            root_node_rotation.z,
            -root_node_rotation.y,
            root_node_rotation.w,
        ],
        "scale": [
            root_node_scale.x,
            root_node_scale.z,
            root_node_scale.y,
        ],
    }
    root_node_child_indices: list[int] = []
    node_dicts.append(root_node_dict)

    # Now build the node hierarchy in local transforms
    for bone in armature.pose.bones:
        if not bone.parent:
            cnode_idx = create_node_dicts(
                bone,
                None,
                node_dicts,
                bone_name_to_node_index,
                armature.matrix_world,
                bone_name_to_world_rest_matrix,
            )
            root_node_child_indices.append(cnode_idx)

        # Compute base_quaternion ignoring localBone.inverted
        if bone.parent:
            base_q = (bone.parent.matrix.inverted_safe() @ bone.matrix).to_quaternion()
        else:
            base_q = bone.matrix.to_quaternion()
        bone_name_to_base_quaternion[bone.name] = base_q

        # gather data_path references
        if bone.rotation_mode == ROTATION_MODE_QUATERNION:
            data_path_to_bone_and_property_name[
                bone.path_from_id("rotation_quaternion")
            ] = (bone, "rotation_quaternion")
        elif bone.rotation_mode == ROTATION_MODE_AXIS_ANGLE:
            data_path_to_bone_and_property_name[
                bone.path_from_id("rotation_axis_angle")
            ] = (bone, "rotation_axis_angle")
        elif bone.rotation_mode in ROTATION_MODE_EULER:
            data_path_to_bone_and_property_name[bone.path_from_id("rotation_euler")] = (
                bone,
                "rotation_euler",
            )
        else:
            logger.error(
                "Unexpected rotation mode for bone %s: %s",
                bone.name,
                bone.rotation_mode,
            )

        if human_bones.hips.node.bone_name == bone.name:
            data_path_to_bone_and_property_name[bone.path_from_id("location")] = (
                bone,
                "location",
            )
        else:
            data_path_to_bone_and_property_name[bone.path_from_id("location")] = (
                bone,
                "location",
            )

    if root_node_child_indices:
        root_node_dict["children"] = make_json(root_node_child_indices)

    frame_to_timestamp_factor = context.scene.render.fps_base / float(
        context.scene.render.fps
    )

    buffer0_bytearray = bytearray()
    accessor_dicts: list[dict[str, Json]] = []
    buffer_view_dicts: list[dict[str, Json]] = []
    animation_sampler_dicts: list[dict[str, Json]] = []
    animation_channel_dicts: list[dict[str, Json]] = []
    preset_expression_dict: dict[str, dict[str, Json]] = {}
    custom_expression_dict: dict[str, dict[str, Json]] = {}

    frame_start = context.scene.frame_start
    frame_end = context.scene.frame_end

    create_expression_animation(
        vrm1,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_to_timestamp_factor=frame_to_timestamp_factor,
        armature_data=armature_data,
        node_dicts=node_dicts,
        accessor_dicts=accessor_dicts,
        buffer_view_dicts=buffer_view_dicts,
        animation_channel_dicts=animation_channel_dicts,
        animation_sampler_dicts=animation_sampler_dicts,
        scene_node_indices=scene_node_indices,
        buffer0_bytearray=buffer0_bytearray,
        preset_expression_dict=preset_expression_dict,
        custom_expression_dict=custom_expression_dict,
    )

    create_node_animation(
        vrm1,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_to_timestamp_factor=frame_to_timestamp_factor,
        armature=armature,
        data_path_to_bone_and_property_name=data_path_to_bone_and_property_name,
        bone_name_to_node_index=bone_name_to_node_index,
        bone_name_to_base_quaternion=bone_name_to_base_quaternion,
        buffer0_bytearray=buffer0_bytearray,
        buffer_view_dicts=buffer_view_dicts,
        accessor_dicts=accessor_dicts,
        animation_channel_dicts=animation_channel_dicts,
        animation_sampler_dicts=animation_sampler_dicts,
        bone_name_to_world_rest_matrix=bone_name_to_world_rest_matrix,
    )

    look_at_target_node_index = create_look_at_animation(
        vrm1,
        frame_start=frame_start,
        frame_end=frame_end,
        frame_to_timestamp_factor=frame_to_timestamp_factor,
        node_dicts=node_dicts,
        accessor_dicts=accessor_dicts,
        buffer_view_dicts=buffer_view_dicts,
        animation_channel_dicts=animation_channel_dicts,
        animation_sampler_dicts=animation_sampler_dicts,
        buffer0_bytearray=buffer0_bytearray,
    )

    buffer_dicts: list[dict[str, Json]] = [{"byteLength": len(buffer0_bytearray)}]

    human_bones_dict: dict[str, Json] = {}
    human_bone_name_to_human_bone = human_bones.human_bone_name_to_human_bone()
    for human_bone_name, human_bone in human_bone_name_to_human_bone.items():
        bone_name = human_bone.node.bone_name
        node_index = bone_name_to_node_index.get(bone_name)
        if not isinstance(node_index, int):
            continue
        human_bones_dict[human_bone_name.value] = {"node": node_index}

    addon_version = version.get_addon_version()
    if environ.get("BLENDER_VRM_USE_TEST_EXPORTER_VERSION") == "true":
        addon_version = (999, 999, 999)

    vrmc_vrm_animation_dict: dict[str, Json] = {
        "specVersion": "1.0",
        "humanoid": {
            "humanBones": human_bones_dict,
        },
        "expressions": make_json(
            {
                "preset": preset_expression_dict,
                "custom": custom_expression_dict,
            }
        ),
    }
    if look_at_target_node_index is not None:
        vrmc_vrm_animation_dict["lookAt"] = {
            "node": look_at_target_node_index,
            "offsetFromHeadBone": list(vrm1.look_at.offset_from_head_bone),
        }

    vrma_dict = make_json(
        {
            "asset": {
                "version": "2.0",
                "generator": "VRM Add-on for Blender v"
                + ".".join(map(str, addon_version)),
            },
            "nodes": node_dicts,
            "scenes": [{"nodes": scene_node_indices}],
            "buffers": buffer_dicts,
            "bufferViews": buffer_view_dicts,
            "accessors": accessor_dicts,
            "animations": [
                {
                    "channels": animation_channel_dicts,
                    "samplers": animation_sampler_dicts,
                }
            ],
            "extensionsUsed": ["VRMC_vrm_animation"],
            "extensions": {
                "VRMC_vrm_animation": vrmc_vrm_animation_dict,
            },
        }
    )

    if not isinstance(vrma_dict, dict):
        message = "vrma_dict is not dict"
        raise TypeError(message)

    return pack_glb(vrma_dict, buffer0_bytearray)


def create_look_at_animation(
    vrm1: Vrm1PropertyGroup,
    *,
    frame_start: int,
    frame_end: int,
    frame_to_timestamp_factor: float,
    node_dicts: list[dict[str, Json]],
    accessor_dicts: list[dict[str, Json]],
    buffer_view_dicts: list[dict[str, Json]],
    animation_channel_dicts: list[dict[str, Json]],
    animation_sampler_dicts: list[dict[str, Json]],
    buffer0_bytearray: bytearray,
) -> Optional[int]:
    look_at_target_object = vrm1.look_at.preview_target_bpy_object
    if not look_at_target_object:
        return None

    animation_data = look_at_target_object.animation_data
    if not animation_data:
        return None

    action = animation_data.action
    if not action:
        return None

    look_at_translation_offsets: list[Vector] = []
    data_path = look_at_target_object.path_from_id("location")
    for fcurve in action.fcurves:
        if fcurve.mute:
            continue
        if not fcurve.is_valid:
            continue
        if fcurve.data_path != data_path:
            continue
        for frame_i in range(frame_start, frame_end + 1):
            idx = frame_i - frame_start
            val = float(fcurve.evaluate(frame_i))
            while len(look_at_translation_offsets) <= idx:
                look_at_translation_offsets.append(Vector((0.0, 0.0, 0.0)))
            look_at_translation_offsets[idx][fcurve.array_index] = val

    if not look_at_translation_offsets:
        return None

    def_trans, def_rot, def_scale = look_at_target_object.matrix_world.decompose()
    def_matrix = def_rot.to_matrix().to_4x4() @ Matrix.Diagonal(def_scale).to_4x4()

    final_offsets = [vec @ def_matrix for vec in look_at_translation_offsets]

    look_at_target_node_index = len(node_dicts)
    node_dicts.append(
        {
            "name": look_at_target_object.name,
            "translation": [
                def_trans.x,
                def_trans.z,
                -def_trans.y,
            ],
        }
    )

    in_off = len(buffer0_bytearray)
    time_vals = [
        float(i) * frame_to_timestamp_factor for i in range(len(final_offsets))
    ]
    in_bytes = struct.pack("<" + "f" * len(time_vals), *time_vals)
    buffer0_bytearray.extend(in_bytes)
    while len(buffer0_bytearray) % 32 != 0:
        buffer0_bytearray.append(0)
    in_bv_idx = len(buffer_view_dicts)
    in_bv = {"buffer": 0, "byteLength": len(in_bytes)}
    if in_off > 0:
        in_bv["byteOffset"] = in_off
    buffer_view_dicts.append(in_bv)

    out_off = len(buffer0_bytearray)
    gltf_translations = [(v.x, v.z, -v.y) for v in final_offsets]
    trans_flat = list(itertools.chain(*gltf_translations))
    out_bytes = struct.pack("<" + "f" * len(trans_flat), *trans_flat)
    buffer0_bytearray.extend(out_bytes)
    while len(buffer0_bytearray) % 32 != 0:
        buffer0_bytearray.append(0)
    out_bv_idx = len(buffer_view_dicts)
    out_bv = {"buffer": 0, "byteLength": len(out_bytes)}
    if out_off > 0:
        out_bv["byteOffset"] = out_off
    buffer_view_dicts.append(out_bv)

    inp_acc_idx = len(accessor_dicts)
    accessor_dicts.append(
        {
            "bufferView": in_bv_idx,
            "componentType": GL_FLOAT,
            "count": len(time_vals),
            "type": "SCALAR",
            "min": [min(time_vals)],
            "max": [max(time_vals)],
        }
    )
    out_acc_idx = len(accessor_dicts)
    x_list = [t[0] for t in gltf_translations]
    y_list = [t[1] for t in gltf_translations]
    z_list = [t[2] for t in gltf_translations]
    accessor_dicts.append(
        {
            "bufferView": out_bv_idx,
            "componentType": GL_FLOAT,
            "count": len(gltf_translations),
            "type": "VEC3",
            "min": [min(x_list), min(y_list), min(z_list)],
            "max": [max(x_list), max(y_list), max(z_list)],
        }
    )

    smp_idx = len(animation_sampler_dicts)
    animation_sampler_dicts.append({"input": inp_acc_idx, "output": out_acc_idx})
    animation_channel_dicts.append(
        {
            "sampler": smp_idx,
            "target": {
                "node": look_at_target_node_index,
                "path": "translation",
            },
        }
    )

    return look_at_target_node_index


def create_expression_animation(
    vrm1: Vrm1PropertyGroup,
    *,
    frame_start: int,
    frame_end: int,
    frame_to_timestamp_factor: float,
    armature_data: Armature,
    node_dicts: list[dict[str, Json]],
    accessor_dicts: list[dict[str, Json]],
    buffer_view_dicts: list[dict[str, Json]],
    animation_channel_dicts: list[dict[str, Json]],
    animation_sampler_dicts: list[dict[str, Json]],
    scene_node_indices: list[int],
    buffer0_bytearray: bytearray,
    preset_expression_dict: dict[str, dict[str, Json]],
    custom_expression_dict: dict[str, dict[str, Json]],
) -> None:
    expression_animation_data = armature_data.animation_data
    if not expression_animation_data:
        return
    action = expression_animation_data.action
    if not action:
        return

    data_path_to_expression_name: dict[str, str] = {}
    for nm, expr in vrm1.expressions.all_name_to_expression_dict().items():
        if nm in ["lookUp", "lookDown", "lookLeft", "lookRight"]:
            continue
        data_path_to_expression_name[expr.path_from_id("preview")] = expr.name

    expr_name_to_trans_vals: dict[str, list[tuple[float, float, float]]] = {}

    idxer = 0
    for fcv in action.fcurves:
        if fcv.mute or not fcv.is_valid:
            continue
        nm = data_path_to_expression_name.get(fcv.data_path)
        if not nm:
            continue
        for fr in range(frame_start, frame_end + 1):
            valx = float(fcv.evaluate(fr))
            valx = max(0.0, min(1.0, valx))
            if nm not in expr_name_to_trans_vals:
                expr_name_to_trans_vals[nm] = []
            expr_name_to_trans_vals[nm].append((valx, 0.0, idxer / 8.0))
        idxer += 1

    for enm, tv in expr_name_to_trans_vals.items():
        ndx = len(node_dicts)
        node_dicts.append({"name": enm})
        if enm in vrm1.expressions.preset.name_to_expression_dict():
            preset_expression_dict[enm] = {"node": ndx}
        else:
            custom_expression_dict[enm] = {"node": ndx}
        scene_node_indices.append(ndx)

        in_off = len(buffer0_bytearray)
        tim_list = [float(i) * frame_to_timestamp_factor for i, _ in enumerate(tv)]
        in_bytes = struct.pack("<" + "f" * len(tim_list), *tim_list)
        buffer0_bytearray.extend(in_bytes)
        while len(buffer0_bytearray) % 32 != 0:
            buffer0_bytearray.append(0)
        in_bv_idx = len(buffer_view_dicts)
        in_bv_d = {"buffer": 0, "byteLength": len(in_bytes)}
        if in_off > 0:
            in_bv_d["byteOffset"] = in_off
        buffer_view_dicts.append(in_bv_d)

        out_off = len(buffer0_bytearray)
        flist = list(itertools.chain(*tv))
        out_bytes = struct.pack("<" + "f" * len(flist), *flist)
        buffer0_bytearray.extend(out_bytes)
        while len(buffer0_bytearray) % 32 != 0:
            buffer0_bytearray.append(0)
        out_bv_idx = len(buffer_view_dicts)
        out_bv_d = {"buffer": 0, "byteLength": len(out_bytes)}
        if out_off > 0:
            out_bv_d["byteOffset"] = out_off
        buffer_view_dicts.append(out_bv_d)

        inp_acc_idx = len(accessor_dicts)
        accessor_dicts.append(
            {
                "bufferView": in_bv_idx,
                "componentType": GL_FLOAT,
                "count": len(tim_list),
                "type": "SCALAR",
                "min": [min(tim_list)],
                "max": [max(tim_list)],
            }
        )
        out_acc_idx = len(accessor_dicts)
        xvals = [tup[0] for tup in tv]
        yvals = [tup[1] for tup in tv]
        zvals = [tup[2] for tup in tv]
        accessor_dicts.append(
            {
                "bufferView": out_bv_idx,
                "componentType": GL_FLOAT,
                "count": len(tv),
                "type": "VEC3",
                "min": [min(xvals), min(yvals), min(zvals)],
                "max": [max(xvals), max(yvals), max(zvals)],
            }
        )

        sidx = len(animation_sampler_dicts)
        animation_sampler_dicts.append({"input": inp_acc_idx, "output": out_acc_idx})
        animation_channel_dicts.append(
            {
                "sampler": sidx,
                "target": {"node": ndx, "path": "translation"},
            }
        )


def create_node_animation(
    vrm1: Vrm1PropertyGroup,
    *,
    frame_start: int,
    frame_end: int,
    frame_to_timestamp_factor: float,
    armature: Object,
    data_path_to_bone_and_property_name: dict[str, tuple[PoseBone, str]],
    bone_name_to_node_index: dict[str, int],
    bone_name_to_base_quaternion: dict[str, Quaternion],
    buffer0_bytearray: bytearray,
    buffer_view_dicts: list[dict[str, Json]],
    accessor_dicts: list[dict[str, Json]],
    animation_channel_dicts: list[dict[str, Json]],
    animation_sampler_dicts: list[dict[str, Json]],
    bone_name_to_world_rest_matrix: dict[str, Matrix],
) -> None:
    human_bones = vrm1.humanoid.human_bones
    human_bone_name_to_human_bone = human_bones.human_bone_name_to_human_bone()

    animation_data = armature.animation_data
    if not animation_data:
        return
    action = animation_data.action
    if not action:
        return

    bone_name_to_quaternion_offsets: dict[str, list[Quaternion]] = {}
    bone_name_to_euler_offsets: dict[str, list[Euler]] = {}
    bone_name_to_axis_angle_offsets: dict[str, list[list[float]]] = {}
    hips_translation_offsets: list[Vector] = []
    bone_name_to_location_offsets_for_non_humanoid: dict[str, list[Vector]] = {}
    for pb in armature.pose.bones:
        bone_name_to_location_offsets_for_non_humanoid[pb.name] = []

    for fcurve in action.fcurves:
        if fcurve.mute or not fcurve.is_valid:
            continue
        bone_and_property = data_path_to_bone_and_property_name.get(fcurve.data_path)
        if not bone_and_property:
            continue
        bone, prop_name = bone_and_property
        for frame_i in range(frame_start, frame_end + 1):
            idx = frame_i - frame_start
            val = float(fcurve.evaluate(frame_i))
            if prop_name == "rotation_quaternion":
                arr = bone_name_to_quaternion_offsets.setdefault(bone.name, [])
                while len(arr) <= idx:
                    arr.append(Quaternion((0.0, 0.0, 0.0, 1.0)))
                arr[idx][fcurve.array_index] = val
            elif prop_name == "rotation_axis_angle":
                arr = bone_name_to_axis_angle_offsets.setdefault(bone.name, [])
                while len(arr) <= idx:
                    arr.append([0.0, 0.0, 0.0, 0.0])
                arr[idx][fcurve.array_index] = val
            elif prop_name == "rotation_euler":
                arr = bone_name_to_euler_offsets.setdefault(bone.name, [])
                while len(arr) <= idx:
                    arr.append(Euler((0, 0, 0)))
                eul = arr[idx]
                idx_map = {
                    "XYZ": [0, 1, 2],
                    "XZY": [0, 2, 1],
                    "YXZ": [1, 0, 2],
                    "YZX": [1, 2, 0],
                    "ZXY": [2, 0, 1],
                    "ZYX": [2, 1, 0],
                }.get(bone.rotation_mode, [0, 1, 2])
                eul[idx_map[fcurve.array_index]] = val
            elif prop_name == "location":
                if bone.name == human_bones.hips.node.bone_name:
                    while len(hips_translation_offsets) <= idx:
                        hips_translation_offsets.append(Vector((0.0, 0.0, 0.0)))
                    hips_translation_offsets[idx][fcurve.array_index] = val
                else:
                    arr = bone_name_to_location_offsets_for_non_humanoid[bone.name]
                    while len(arr) <= idx:
                        arr.append(Vector((0.0, 0.0, 0.0)))
                    arr[idx][fcurve.array_index] = val

    total_frames = frame_end - frame_start + 1
    bone_name_to_quaternions: dict[str, list[Quaternion]] = {}
    for pb in armature.pose.bones:
        bname = pb.name
        base_q = bone_name_to_base_quaternion.get(bname, Quaternion())
        if bname in bone_name_to_quaternion_offsets:
            arr = bone_name_to_quaternion_offsets[bname]
            result_quats = []
            for qtemp in arr:
                result_quats.append(base_q @ qtemp.normalized())
            bone_name_to_quaternions[bname] = result_quats
        elif bname in bone_name_to_axis_angle_offsets:
            arr = bone_name_to_axis_angle_offsets[bname]
            result_quats = []
            for axang in arr:
                axq = Quaternion((axang[1], axang[2], axang[3]), axang[0]).normalized()
                result_quats.append(base_q @ axq)
            bone_name_to_quaternions[bname] = result_quats
        elif bname in bone_name_to_euler_offsets:
            arr = bone_name_to_euler_offsets[bname]
            result_quats = []
            for eul in arr:
                eq = eul.to_quaternion()
                result_quats.append(base_q @ eq)
            bone_name_to_quaternions[bname] = result_quats
        else:
            bone_name_to_quaternions[bname] = [base_q for _ in range(total_frames)]

    # rotation
    for bname, quats in bone_name_to_quaternions.items():
        node_idx = bone_name_to_node_index.get(bname)
        if node_idx is None:
            logger.error("Failed to find node index for bone %s", bname)
            continue
        maybe_hb_name = None
        for hh, hhbone in human_bone_name_to_human_bone.items():
            if hhbone.node.bone_name == bname:
                maybe_hb_name = hh
                break
        if maybe_hb_name in [HumanBoneName.LEFT_EYE, HumanBoneName.RIGHT_EYE]:
            continue

        in_off = len(buffer0_bytearray)
        times_arr = [float(i) * frame_to_timestamp_factor for i, _ in enumerate(quats)]
        inbytes = struct.pack("<" + "f" * len(times_arr), *times_arr)
        buffer0_bytearray.extend(inbytes)
        while len(buffer0_bytearray) % 32 != 0:
            buffer0_bytearray.append(0)
        in_bv_idx = len(buffer_view_dicts)
        in_bv_d = {"buffer": 0, "byteLength": len(inbytes)}
        if in_off > 0:
            in_bv_d["byteOffset"] = in_off
        buffer_view_dicts.append(in_bv_d)

        out_off = len(buffer0_bytearray)
        # ---- FIX: corrected quaternion axis mapping below ----
        gltf_quats = [(qq.x, qq.z, -qq.y, qq.w) for qq in quats]
        flatted = list(itertools.chain(*gltf_quats))
        out_b = struct.pack("<" + "f" * len(flatted), *flatted)
        buffer0_bytearray.extend(out_b)
        while len(buffer0_bytearray) % 32 != 0:
            buffer0_bytearray.append(0)
        out_bv_idx = len(buffer_view_dicts)
        out_bv_d = {"buffer": 0, "byteLength": len(out_b)}
        if out_off > 0:
            out_bv_d["byteOffset"] = out_off
        buffer_view_dicts.append(out_bv_d)

        inp_acc_idx = len(accessor_dicts)
        accessor_dicts.append(
            {
                "bufferView": in_bv_idx,
                "componentType": GL_FLOAT,
                "count": len(times_arr),
                "type": "SCALAR",
                "min": [min(times_arr)],
                "max": [max(times_arr)],
            }
        )
        out_acc_idx = len(accessor_dicts)
        xvs = [gg[0] for gg in gltf_quats]
        yvs = [gg[1] for gg in gltf_quats]
        zvs = [gg[2] for gg in gltf_quats]
        wvs = [gg[3] for gg in gltf_quats]
        accessor_dicts.append(
            {
                "bufferView": out_bv_idx,
                "componentType": GL_FLOAT,
                "count": len(quats),
                "type": "VEC4",
                "min": [min(xvs), min(yvs), min(zvs), min(wvs)],
                "max": [max(xvs), max(yvs), max(zvs), max(wvs)],
            }
        )

        smp_idx = len(animation_sampler_dicts)
        animation_sampler_dicts.append({"input": inp_acc_idx, "output": out_acc_idx})
        animation_channel_dicts.append(
            {
                "sampler": smp_idx,
                "target": {"node": node_idx, "path": "rotation"},
            }
        )

    # hips
    hips_bone_name = human_bones.hips.node.bone_name
    hips_bone = armature.pose.bones.get(hips_bone_name)
    if hips_bone:
        hips_node_idx = bone_name_to_node_index.get(hips_bone_name)
        if isinstance(hips_node_idx, int):
            hips_translation_frames = []
            for i in range(total_frames):
                # We'll treat the fcurve location as local offset
                # and just store it as-is in the channel, matching
                # the importer logic expecting local transforms.
                if i < len(hips_translation_offsets):
                    tloc = hips_translation_offsets[i]
                else:
                    tloc = Vector((0.0, 0.0, 0.0))
                hips_translation_frames.append(tloc)

            if hips_translation_frames:
                in_off = len(buffer0_bytearray)
                times_arr = [
                    float(i) * frame_to_timestamp_factor
                    for i, _ in enumerate(hips_translation_frames)
                ]
                in_b = struct.pack("<" + "f" * len(times_arr), *times_arr)
                buffer0_bytearray.extend(in_b)
                while len(buffer0_bytearray) % 32 != 0:
                    buffer0_bytearray.append(0)
                in_bv_idx = len(buffer_view_dicts)
                in_bv_d = {"buffer": 0, "byteLength": len(in_b)}
                if in_off > 0:
                    in_bv_d["byteOffset"] = in_off
                buffer_view_dicts.append(in_bv_d)

                out_off = len(buffer0_bytearray)
                gltf_off = [(ddd.x, ddd.z, -ddd.y) for ddd in hips_translation_frames]
                flt = list(itertools.chain(*gltf_off))
                out_b = struct.pack("<" + "f" * len(flt), *flt)
                buffer0_bytearray.extend(out_b)
                while len(buffer0_bytearray) % 32 != 0:
                    buffer0_bytearray.append(0)
                out_bv_idx = len(buffer_view_dicts)
                out_bv_d = {"buffer": 0, "byteLength": len(out_b)}
                if out_off > 0:
                    out_bv_d["byteOffset"] = out_off
                buffer_view_dicts.append(out_bv_d)

                inp_acc_idx = len(accessor_dicts)
                accessor_dicts.append(
                    {
                        "bufferView": in_bv_idx,
                        "componentType": GL_FLOAT,
                        "count": len(times_arr),
                        "type": "SCALAR",
                        "min": [min(times_arr)],
                        "max": [max(times_arr)],
                    }
                )
                out_acc_idx = len(accessor_dicts)
                xarr = [a[0] for a in gltf_off]
                yarr = [a[1] for a in gltf_off]
                zarr = [a[2] for a in gltf_off]
                accessor_dicts.append(
                    {
                        "bufferView": out_bv_idx,
                        "componentType": GL_FLOAT,
                        "count": len(hips_translation_frames),
                        "type": "VEC3",
                        "min": [min(xarr), min(yarr), min(zarr)],
                        "max": [max(xarr), max(yarr), max(zarr)],
                    }
                )

                sidx = len(animation_sampler_dicts)
                animation_sampler_dicts.append(
                    {"input": inp_acc_idx, "output": out_acc_idx}
                )
                animation_channel_dicts.append(
                    {
                        "sampler": sidx,
                        "target": {"node": hips_node_idx, "path": "translation"},
                    }
                )

    # non-humanoid translation
    for pb in armature.pose.bones:
        if pb.name == hips_bone_name:
            continue
        arr_loc = bone_name_to_location_offsets_for_non_humanoid[pb.name]
        if not arr_loc:
            continue
        nd_idx = bone_name_to_node_index.get(pb.name)
        if nd_idx is None:
            logger.error("Failed to find node index for bone %s", pb.name)
            continue

        frames = []
        for i in range(total_frames):
            if i < len(arr_loc):
                loc_ = arr_loc[i]
            else:
                loc_ = Vector((0.0, 0.0, 0.0))
            frames.append(loc_)

        if frames:
            in_off = len(buffer0_bytearray)
            tm_arr = [
                float(i) * frame_to_timestamp_factor for i, _ in enumerate(frames)
            ]
            inb = struct.pack("<" + "f" * len(tm_arr), *tm_arr)
            buffer0_bytearray.extend(inb)
            while len(buffer0_bytearray) % 32 != 0:
                buffer0_bytearray.append(0)
            in_bv_idx = len(buffer_view_dicts)
            in_bv_d = {"buffer": 0, "byteLength": len(inb)}
            if in_off > 0:
                in_bv_d["byteOffset"] = in_off
            buffer_view_dicts.append(in_bv_d)

            out_off = len(buffer0_bytearray)
            gltf_lc = [(rr.x, rr.z, -rr.y) for rr in frames]
            floc = list(itertools.chain(*gltf_lc))
            outb = struct.pack("<" + "f" * len(floc), *floc)
            buffer0_bytearray.extend(outb)
            while len(buffer0_bytearray) % 32 != 0:
                buffer0_bytearray.append(0)
            out_bv_idx = len(buffer_view_dicts)
            out_bv_d = {"buffer": 0, "byteLength": len(outb)}
            if out_off > 0:
                out_bv_d["byteOffset"] = out_off
            buffer_view_dicts.append(out_bv_d)

            inp_acc_idx = len(accessor_dicts)
            accessor_dicts.append(
                {
                    "bufferView": in_bv_idx,
                    "componentType": GL_FLOAT,
                    "count": len(tm_arr),
                    "type": "SCALAR",
                    "min": [min(tm_arr)],
                    "max": [max(tm_arr)],
                }
            )
            out_acc_idx = len(accessor_dicts)
            xxvals = [hw[0] for hw in gltf_lc]
            yyvals = [hw[1] for hw in gltf_lc]
            zzvals = [hw[2] for hw in gltf_lc]
            accessor_dicts.append(
                {
                    "bufferView": out_bv_idx,
                    "componentType": GL_FLOAT,
                    "count": len(frames),
                    "type": "VEC3",
                    "min": [min(xxvals), min(yyvals), min(zzvals)],
                    "max": [max(xxvals), max(yyvals), max(zzvals)],
                }
            )

            smp_i = len(animation_sampler_dicts)
            animation_sampler_dicts.append(
                {"input": inp_acc_idx, "output": out_acc_idx}
            )
            animation_channel_dicts.append(
                {
                    "sampler": smp_i,
                    "target": {"node": nd_idx, "path": "translation"},
                }
            )
