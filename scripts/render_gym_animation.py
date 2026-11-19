"""Render the QAgent Gym README artwork with Blender 4.4+.

blender --background --python scripts/render_gym_animation.py -- --output .work/gym-animation
Add --preview to render one frame. Use ImageMagick to assemble the numbered PNGs.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

FRAME_COUNT = 72
FPS = 18


def material(name, color, *, metallic=0, roughness=0.35, glow=0):
    result = bpy.data.materials.new(name)
    result.diffuse_color = (*color, 1)
    result.use_nodes = True
    surface = result.node_tree.nodes.get("Principled BSDF")
    surface.inputs["Base Color"].default_value = (*color, 1)
    surface.inputs["Metallic"].default_value = metallic
    surface.inputs["Roughness"].default_value = roughness
    surface.inputs["Emission Color"].default_value = (*color, 1)
    surface.inputs["Emission Strength"].default_value = glow
    return result


def finish(obj, name, location, surface, parent=None):
    obj.name = name
    obj.parent = parent
    obj.location = location
    obj.data.materials.append(surface)
    return obj


def box(name, location, size, surface, radius=0.08, parent=None):
    bpy.ops.mesh.primitive_cube_add()
    obj = bpy.context.object
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if radius:
        bevel = obj.modifiers.new("Rounded edges", "BEVEL")
        bevel.width = radius
        bevel.segments = 4
        obj.modifiers.new("Surface normals", "WEIGHTED_NORMAL")
    return finish(obj, name, location, surface, parent)


def sphere(name, location, size, surface, parent=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16)
    obj = bpy.context.object
    obj.scale = size
    bpy.ops.object.shade_smooth()
    return finish(obj, name, location, surface, parent)


def cylinder(name, location, radius, depth, surface, parent=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=depth)
    obj = bpy.context.object
    bevel = obj.modifiers.new("Machined edges", "BEVEL")
    bevel.width = 0.035
    bevel.segments = 3
    obj.modifiers.new("Surface normals", "WEIGHTED_NORMAL")
    return finish(obj, name, location, surface, parent)


def text(name, value, location, size, surface, parent=None):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = value
    curve.size = size
    curve.extrude = 0.004
    curve.bevel_depth = 0.0015
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.rotation_euler = (math.pi / 2, 0, 0)
    return finish(obj, name, location, surface, parent)


def line(name, points, surface, radius=0.018, parent=None):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 3
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for point, co in zip(spline.points, points, strict=True):
        point.co = (*co, 1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    return finish(obj, name, (0, 0, 0), surface, parent)


def pivot(name, location, parent=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    obj.location = location
    return obj


def brand_material():
    surface = material("QAgents journal emblem", (1, 1, 1), glow=0.8)
    nodes = surface.node_tree.nodes
    shader = nodes.get("Principled BSDF")
    image = nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(
        str(Path(__file__).resolve().parents[1] / "assets/brand/qagents-icon-dark.png")
    )
    image.image.pack()
    for target in ("Base Color", "Emission Color"):
        surface.node_tree.links.new(image.outputs["Color"], shader.inputs[target])
    surface.node_tree.links.new(image.outputs["Alpha"], shader.inputs["Alpha"])
    return surface


def brand_mark(name, location, size, surface, parent=None):
    bpy.ops.mesh.primitive_plane_add(size=size)
    obj = bpy.context.object
    obj.rotation_euler.x = math.pi / 2
    return finish(obj, name, location, surface, parent)


def robot(name, location, accent, palette):
    root = pivot(name, location)
    white, dark, light = palette["shell"], palette["visor"], palette["light"]
    body = pivot(name + " body", (0, 0, 0), root)
    box("Torso", (0, 0, 1.0), (0.68, 0.48, 0.65), white, 0.19, body)
    box("Helmet", (0, 0, 1.75), (1.07, 0.83, 0.84), white, 0.3, body)
    box("Visor", (0, -0.414, 1.73), (0.88, 0.12, 0.49), dark, 0.2, body)
    for side in (-1, 1):
        ear = cylinder("Headphone", (side * 0.55, 0, 1.76), 0.23, 0.12, accent, body)
        ear.rotation_euler[1] = math.pi / 2
        eye_points = [
            (side * 0.19 + math.cos(a) * 0.068, -0.484, 1.74 + math.sin(a) * 0.075)
            for a in [math.pi * i / 16 for i in range(17)]
        ]
        line("Friendly eye", eye_points, light, 0.017, body)
    line(
        "Smile",
        [
            (math.cos(a) * 0.055, -0.485, 1.67 + math.sin(a) * 0.035)
            for a in [math.pi + math.pi * i / 16 for i in range(17)]
        ],
        light,
        0.009,
        body,
    )
    cylinder("Antenna stem", (0, 0, 2.24), 0.025, 0.19, palette["metal"], body)
    sphere("Antenna light", (0, 0, 2.35), (0.07, 0.07, 0.07), accent, body)
    brand_mark("Q chest emblem", (0, -0.258, 1.03), 0.43, palette["brand"], body)
    arms, legs = [], []
    for side in (-1, 1):
        arm = pivot("Shoulder", (side * 0.43, 0, 1.17), body)
        sphere("Shoulder joint", (0, 0, 0), (0.12, 0.12, 0.12), accent, arm)
        box("Arm", (0, 0, -0.22), (0.19, 0.21, 0.39), white, 0.085, arm)
        sphere("Hand", (0, -0.015, -0.46), (0.13, 0.13, 0.14), palette["joint"], arm)
        arms.append(arm)
        leg = pivot("Hip", (side * 0.18, 0, 0.65), root)
        cylinder("Leg", (0, 0, -0.17), 0.105, 0.3, accent, leg)
        box("Shoe", (0, -0.07, -0.37), (0.27, 0.43, 0.19), white, 0.08, leg)
        legs.append(leg)
    return root, body, arms, legs


def area_light(name, location, power, color, size, target):
    bpy.ops.object.light_add(type="AREA", location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.energy = power
    obj.data.color = color
    obj.data.shape = "DISK"
    obj.data.size = size
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def build_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    palette = {
        "shell": material("Pearl ceramic", (0.84, 0.91, 0.92), metallic=0.24),
        "visor": material("Obsidian glass", (0.008, 0.023, 0.035), metallic=0.5, roughness=0.13),
        "joint": material("Graphite joints", (0.04, 0.075, 0.1), metallic=0.55),
        "metal": material("Brushed titanium", (0.23, 0.37, 0.42), metallic=0.8),
        "teal": material("Q teal", (0.015, 0.47, 0.43), metallic=0.4),
        "coral": material("Coral", (0.92, 0.19, 0.1), metallic=0.15),
        "light": material("Mint LEDs", (0.18, 0.95, 0.77), glow=2.3),
        "gold": material("Amber LEDs", (1.0, 0.52, 0.14), glow=1.3),
        "floor": material("Deep ocean enamel", (0.018, 0.062, 0.09), metallic=0.35),
        "wall": material("Midnight backdrop", (0.012, 0.025, 0.045), roughness=0.65),
        "brand": brand_material(),
    }
    box("Display plinth", (0, 0, -0.16), (7.8, 5, 0.34), palette["floor"], 0.25)
    box("Studio ground", (0, 0, -0.44), (200, 200, 0.2), palette["wall"], 0)
    for x in (-3.7, 3.7):
        box("Runway light", (x, 0, 0.02), (0.027, 4.3, 0.025), palette["light"], 0.008)
    box("Front light", (0, -2.36, -0.08), (6.7, 0.03, 0.035), palette["light"], 0.01)
    # Every screen label is geometry, so the Q identity stays legible through the loop.
    box("Command screen frame", (0, 2.02, 2.66), (5.8, 0.22, 2.22), palette["metal"], 0.15)
    box("Command screen", (0, 1.89, 2.66), (5.56, 0.05, 2.0), palette["visor"], 0.1)
    text("Gym wordmark", "QAgent Gym", (-0.72, 1.84, 3.13), 0.45, palette["shell"])
    text(
        "Training display",
        "PRACTICE   /   REFLECT   /   EVALUATE",
        (-0.67, 1.84, 2.83),
        0.12,
        palette["light"],
    )
    brand = brand_mark("Q display emblem", (-1.65, 1.81, 2.65), 1.8, palette["brand"])
    orbit = pivot("Q orbit", (-1.65, 1.8, 2.65))
    for index in range(3):
        angle = index * 2 * math.pi / 3
        sphere(
            "Orbit particle",
            (math.cos(angle) * 0.78, 0, math.sin(angle) * 0.78),
            (0.045, 0.045, 0.045),
            palette["gold"],
            orbit,
        )
    chart_bars = []
    for index in range(12):
        x = -0.6 + index * 0.235
        bar = box("Live display bar", (x, 1.83, 2.29), (0.095, 0.035, 0.4), palette["light"], 0.025)
        chart_bars.append(bar)
    text("Screen footer", "CONTEXT LEARNING", (-0.65, 1.83, 1.96), 0.14, palette["shell"])
    # Memory station with floating data and a robot practicing a lift.
    box("Memory platform", (-2.35, 0, 0.09), (2.1, 2.7, 0.18), palette["teal"], 0.15)
    text("Memory label", "MEMORY", (-2.96, -1.39, 0.12), 0.22, palette["light"])
    memory = robot("Memory robot", (-2.28, -0.05, 0), palette["teal"], palette)
    data = pivot("Memory data", (-2.28, -0.22, 2.97))
    box("Data cube", (0, 0, 0), (0.45, 0.45, 0.45), palette["metal"], 0.09, data)
    brand_mark("Cube Q", (0, -0.231, 0), 0.43, palette["brand"], data)
    for index in range(3):
        box(
            "Memory cartridge",
            (-3.05, 0.9, 0.35 + index * 0.28),
            (0.35, 0.5, 0.2),
            palette["shell"],
            0.035,
        )
        box(
            "Cartridge status",
            (-3.05, 0.64, 0.35 + index * 0.28),
            (0.16, 0.015, 0.025),
            palette["light"],
            0.005,
        )
    # A treadmill makes the repeated running motion physical.
    box("Treadmill chassis", (0, -0.45, 0.13), (1.8, 2.75, 0.28), palette["metal"], 0.12)
    box("Treadmill belt", (0, -0.45, 0.29), (1.5, 2.52, 0.06), palette["visor"], 0.05)
    treads = [
        box("Belt tread", (0, -1.62 + i * 0.3, 0.33), (1.4, 0.025, 0.008), palette["metal"], 0.003)
        for i in range(8)
    ]
    for x in (-0.82, 0.82):
        line(
            "Treadmill rail",
            [(x, 0.45, 0.3), (x, 0.45, 1.1), (x, 0.0, 1.2)],
            palette["metal"],
            0.04,
        )
    runner = robot("Training robot", (0, -0.45, 0.15), palette["coral"], palette)
    text("Treadmill label", "TRAIN", (-0.36, -1.85, 0.12), 0.23, palette["light"])
    # The evaluation station repeats the telemetry screen's amber status lights.
    box("Evaluation platform", (2.34, 0, 0.09), (2.05, 2.7, 0.18), palette["teal"], 0.15)
    coach = robot("Evaluation robot", (2.3, 0, 0), palette["teal"], palette)
    text("Evaluation label", "EVALUATE", (1.65, -1.39, 0.12), 0.22, palette["light"])
    box("Side console", (3.12, -0.2, 0.85), (0.47, 0.65, 1.35), palette["floor"], 0.08)
    box("Console glass", (3.12, -0.54, 1.25), (0.41, 0.025, 0.44), palette["visor"], 0.025)
    for index in range(3):
        box(
            "Console meter",
            (3.12, -0.56, 1.12 + index * 0.1),
            (0.27, 0.012, 0.024),
            palette["gold"],
            0.008,
        )
    area_light("Softbox", (1, -5, 9), 1500, (0.8, 0.94, 1), 7, (0, 0, 1))
    area_light("Teal rim", (-5, 2, 5), 1100, (0.15, 1, 0.75), 5, (0, 0, 1))
    area_light("Warm rim", (5, 3, 6), 1250, (1, 0.6, 0.3), 5, (0, 0, 1))
    scene = bpy.context.scene
    scene.world.color = (0.12, 0.12, 0.12)
    bpy.ops.object.camera_add(location=(8.6, -13.8, 9.3))
    camera = bpy.context.object
    camera.rotation_euler = (
        (Vector((0, 0, 1.6)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 11.7
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 12
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 960
    scene.render.resolution_y = 600
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.fps = FPS
    scene.view_settings.view_transform = "AgX"
    scene.render.film_transparent = False
    return scene, memory, runner, coach, data, orbit, chart_bars, treads, brand


def animate(scene, memory, runner, coach, data, orbit, bars, treads, brand):
    for frame in range(1, FRAME_COUNT + 1):
        phase = 2 * math.pi * (frame - 1) / FRAME_COUNT
        brand.scale = (1 + 0.025 * math.sin(phase),) * 3
        brand.keyframe_insert("scale", frame=frame)
        for robot_rig, offset in ((memory, 0), (runner, 1.4), (coach, 2.2)):
            _, body, arms, legs = robot_rig
            body.location.z = 0.035 * math.sin(phase * 4 + offset)
            body.keyframe_insert("location", frame=frame)
            for index, arm in enumerate(arms):
                arm.rotation_euler.x = 0.3 * math.sin(phase * 4 + offset + index * math.pi)
                arm.rotation_euler.y = 0.12 * math.sin(phase * 2 + index)
                arm.keyframe_insert("rotation_euler", frame=frame)
        for index, leg in enumerate(runner[3]):
            leg.rotation_euler.x = 0.55 * math.sin(phase * 4 + index * math.pi)
            leg.keyframe_insert("rotation_euler", frame=frame)
        memory[2][0].rotation_euler.x = -2.2 + 0.18 * math.sin(phase * 2)
        memory[2][0].keyframe_insert("rotation_euler", frame=frame)
        data.location.z = 2.95 + 0.13 * math.sin(phase * 2)
        data.rotation_euler.z = 0.12 * math.sin(phase)
        data.keyframe_insert("location", frame=frame)
        data.keyframe_insert("rotation_euler", frame=frame)
        coach[2][1].rotation_euler.x = -1.8 + 0.25 * math.sin(phase * 2)
        coach[2][1].keyframe_insert("rotation_euler", frame=frame)
        orbit.rotation_euler.y = phase
        orbit.keyframe_insert("rotation_euler", frame=frame)
        for index, bar in enumerate(bars):
            height = 0.25 + 0.3 * (0.5 + 0.5 * math.sin(phase * 2 + index * 0.7))
            bar.scale.z = height / 0.4
            bar.location.z = 2.12 + height / 2
            bar.keyframe_insert("scale", frame=frame)
            bar.keyframe_insert("location", frame=frame)
        for index, tread in enumerate(treads):
            tread.location.y = -1.62 + ((index * 0.3 - phase / (2 * math.pi) * 2.4) % 2.4)
            tread.keyframe_insert("location", frame=frame)
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=True)
    setup = build_scene()
    animate(*setup)
    scene = setup[0]
    scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str((args.output / "qagent-gym.blend").resolve()))
    scene.render.filepath = str((args.output / "frame-").resolve())
    if args.preview:
        scene.render.filepath = str((args.output / "preview.png").resolve())
        bpy.ops.render.render(write_still=True)
    else:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
