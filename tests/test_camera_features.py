import math

import numpy as np
import pytest
from camera_features import (
    LABEL_SIZE,
    OUTPUT_CHANNELS,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    STRIDE,
    aim_at_centre,
    decode,
    depth_error,
    encode_labels,
    instance_ids,
    merge_cameras,
    pixels_to_ground,
    project,
)
from detection_features import MODEL_GROUND_Z, PEDESTRIAN, VEHICLE

CENTER = (100.0, 50.0, 0.0)
MOUNT = (14.1, 17.2, 7.0)


def carla_matrix(x, y, z, pitch, yaw):
    """carla.Transform.get_matrix() with no roll."""
    cp, sp, cy, sy = math.cos(math.radians(pitch)), math.sin(math.radians(pitch)), math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.array([[cp * cy, -sy, -cy * sp, x], [cp * sy, cy, -sy * sp, y], [sp, 0.0, cp, z], [0.0, 0.0, 0.0, 1.0]])


def mount_camera():
    to_world = carla_matrix(CENTER[0] + MOUNT[0], CENTER[1] + MOUNT[1], CENTER[2] + MOUNT[2], *aim_at_centre(MOUNT))
    return to_world, np.linalg.inv(to_world)


def test_a_camera_projects_what_it_faces_and_nothing_behind_it():
    pitch, yaw = aim_at_centre((10.0, 0.0, 10.0))
    assert pitch == pytest.approx(-45.0) and abs(yaw) == pytest.approx(180.0)  # looking back along -x, at the centre
    pixels = project(np.array([[10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [-5.0, 0.0, 0.0]]), np.eye(4), 640, 360, 90.0)
    assert pixels[0] == pytest.approx((320.0, 180.0))
    assert pixels[1] == pytest.approx((640.0, 180.0))  # 45 degrees right is the edge at a 90 degree field of view
    assert np.isnan(pixels[2]).all()


def test_the_mount_camera_sees_the_centre_mid_image_and_a_ground_pixel_comes_back_to_where_it_was():
    to_world, to_camera = mount_camera()
    assert project(np.array([CENTER]), to_camera)[0] == pytest.approx((320.0, 180.0), abs=1e-6)
    ground = np.array([[CENTER[0] - 20.0, CENTER[1] + 5.0, CENTER[2]], [CENTER[0] + 3.0, CENTER[1] - 30.0, CENTER[2]]])
    assert pixels_to_ground(project(ground, to_camera), to_world, CENTER[2]) == pytest.approx(ground[:, :2], abs=1e-6)
    assert np.isnan(pixels_to_ground([(320.0, 0.0)], to_world, CENTER[2])).all()  # the top edge is above the horizon


def test_instance_ids_read_the_low_byte_from_green_and_the_high_byte_from_blue():
    image = np.zeros((2, 3, 4), np.uint8)
    image[1, 2] = (1, 44, 7, 255)  # B, G, R, A: id 300 with semantic tag 7
    assert instance_ids(image.tobytes(), 3, 2)[1, 2] == 300 and instance_ids(image.tobytes(), 3, 2)[0, 0] == 0


def one_hot_output(labels):
    """An output that's exactly the target, with the heatmaps' peaks at 1, as a perfect model would give."""
    target, _, _ = encode_labels(labels)
    assert target.shape == (OUTPUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH)
    return target


def label(world_xy, to_camera, kind=VEHICLE, yaw=0.5, size=(4.5, 1.9, 1.5), pixels=500):
    u, v = project(np.array([[*world_xy, CENTER[2]]]), to_camera)[0]
    x, y = world_xy[0] - CENTER[0], -(world_xy[1] - CENTER[1])
    return [u, v, x, y, MODEL_GROUND_Z + size[2] / 2, *size, yaw, kind, pixels]


def test_decoding_a_perfect_output_gives_back_where_each_actor_stands():
    to_world, to_camera = mount_camera()
    labels = np.array([label((CENTER[0] - 10.0, CENTER[1] + 4.0), to_camera),
                       label((CENTER[0] + 2.0, CENTER[1] - 6.0), to_camera, PEDESTRIAN, -2.0, (0.6, 0.6, 1.8))], np.float32)
    rows = decode(one_hot_output(labels), 0.5, to_world, CENTER)
    rows = rows[np.argsort(rows[:, 8])]
    assert rows[:, 8].tolist() == [VEHICLE, PEDESTRIAN]
    assert rows[:, :2] == pytest.approx(labels[:, 2:4], abs=1e-3)
    assert rows[:, 6] == pytest.approx(labels[:, 8], abs=1e-5)
    assert rows[:, 3:6] == pytest.approx(labels[:, 5:8], abs=1e-4)


def test_a_barely_visible_actor_is_neither_a_target_nor_a_mistake_and_a_hidden_one_is_skipped():
    _, to_camera = mount_camera()
    barely, hidden = label((CENTER[0], CENTER[1]), to_camera, pixels=5), label((CENTER[0] - 8.0, CENTER[1]), to_camera, pixels=0)
    target, weight, centres = encode_labels(np.array([barely, hidden], np.float32).reshape(-1, LABEL_SIZE))
    assert target[:2].max() == 0.0 and centres.sum() == 0.0
    row, column = int(barely[1] / STRIDE), int(barely[0] / STRIDE)
    assert weight[VEHICLE, row, column] == 0.0 and weight[VEHICLE].mean() > 0.99


def test_the_depth_a_pixel_covers_matches_the_mount_cameras_geometry_and_grows_with_the_square_of_range():
    to_world, to_camera = mount_camera()
    far = np.array([[CENTER[0] - 40.0, CENTER[1] - 50.0, CENTER[2]]])  # about 85 m from the pole
    distance = math.hypot(far[0, 0] - CENTER[0] - MOUNT[0], far[0, 1] - CENTER[1] - MOUNT[1])
    pixel = project(far, to_camera)[0]
    moved = pixels_to_ground([pixel + (0.0, 1.0)], to_world, CENTER[2])[0]  # one pixel lower in the image
    one_pixel = math.hypot(*(moved - far[0, :2]))
    assert depth_error(distance, MOUNT[2], pixels=1) == pytest.approx(one_pixel, rel=0.25)
    assert depth_error(2 * distance, MOUNT[2]) == pytest.approx(4 * depth_error(distance, MOUNT[2]))


def test_two_cameras_finding_one_car_give_one_box_but_a_person_beside_it_stays():
    car_a = (10.0, 5.0, -0.9, 4.5, 1.9, 1.5, 0.0, 0.9, VEHICLE)
    car_b = (11.0, 5.5, -0.9, 4.5, 1.9, 1.5, 0.0, 0.7, VEHICLE)
    person = (11.0, 5.5, -0.9, 0.6, 0.6, 1.8, 0.0, 0.8, PEDESTRIAN)
    merged = merge_cameras([np.array([car_a]), np.array([car_b, person])])
    assert len(merged) == 2 and sorted(merged[:, 7].tolist()) == pytest.approx([0.8, 0.9])
    assert merge_cameras([]).shape == (0, 9)
