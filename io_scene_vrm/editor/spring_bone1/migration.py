import bpy


def migrate_blender_object(armature: bpy.types.Armature) -> None:
    ext = armature.vrm_addon_extension
    if tuple(ext.addon_version) >= (2, 3, 27):
        return

    for collider in ext.spring_bone1.colliders:
        bpy_object = collider.get("blender_object")
        if isinstance(bpy_object, bpy.types.Object):
            collider.bpy_object = bpy_object
        if "blender_object" in collider:
            del collider["blender_object"]


def fixup_gravity_dir(armature: bpy.types.Armature) -> None:
    ext = armature.vrm_addon_extension

    if tuple(ext.addon_version) <= (2, 14, 3):
        for spring in ext.spring_bone1.springs:
            for joint in spring.joints:
                joint.gravity_dir = [
                    joint.gravity_dir[0],
                    joint.gravity_dir[2],
                    joint.gravity_dir[1],
                ]

    if tuple(ext.addon_version) <= (2, 14, 10):
        for spring in ext.spring_bone1.springs:
            for joint in spring.joints:
                joint.gravity_dir = [
                    joint.gravity_dir[0],
                    -joint.gravity_dir[1],
                    joint.gravity_dir[2],
                ]

    if tuple(ext.addon_version) <= (2, 15, 3):
        for spring in ext.spring_bone1.springs:
            for joint in spring.joints:
                gravity_dir = list(joint.gravity_dir)
                joint.gravity_dir = (gravity_dir[0] + 1, 0, 0)  # Make a change
                joint.gravity_dir = gravity_dir


def migrate(armature: bpy.types.Object) -> None:
    armature_data = armature.data
    if not isinstance(armature_data, bpy.types.Armature):
        return
    migrate_blender_object(armature_data)
    fixup_gravity_dir(armature_data)
