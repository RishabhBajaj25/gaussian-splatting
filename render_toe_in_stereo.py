#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import copy
import open3d as o3d

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False
def create_toe_in_stereo_cameras(center, left_cam, baseline=0.065):
    import numpy as np
    def look_at(eye, target, up=np.array([0, 1, 0], dtype=np.float32)):
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up_corrected = np.cross(forward, right)
        view_matrix = np.eye(4, dtype=np.float32)
        view_matrix[0, :3] = right
        view_matrix[1, :3] = up_corrected
        view_matrix[2, :3] = -forward
        view_matrix[:3, 3] = -eye @ np.stack([right, up_corrected, -forward], axis=1)
        return view_matrix    # Use the left camera's cam-to-world matrix to determine viewing direction
    cam_to_world = torch.inverse(left_cam.world_view_transform).cpu().numpy()
    view_dir = -cam_to_world[2, :3]
    up_dir = cam_to_world[1, :3]    # Center both cameras behind the focal point, offset left/right by half the baseline
    cam_center = center - view_dir * 5.0  # 1m behind the center, adjust as needed
    left_pos = cam_center - (baseline / 2.0) * np.cross(view_dir, up_dir)
    right_pos = cam_center + (baseline / 2.0) * np.cross(view_dir, up_dir)    # Construct left/right view matrices
    left_view = look_at(left_pos, center, up_dir)
    right_view = look_at(right_pos, center, up_dir)    # Copy left_cam structure
    left = copy.deepcopy(left_cam)
    right = copy.deepcopy(left_cam)    # Apply view transforms
    left.world_view_transform = torch.tensor(left_view, dtype=torch.float32, device=left_cam.data_device)
    right.world_view_transform = torch.tensor(right_view, dtype=torch.float32, device=left_cam.data_device)    # Keep projection matrix unchanged
    left.full_proj_transform = left.world_view_transform @ left.projection_matrix
    right.full_proj_transform = right.world_view_transform @ right.projection_matrix    # Update camera centers
    left.camera_center = torch.inverse(left.world_view_transform)[3, :3]
    right.camera_center = torch.inverse(right.world_view_transform)[3, :3]
    return left, right





def create_right_eye_camera(left_cam, baseline=0.065):
    # Step 1: Copy the camera
    right_cam = copy.deepcopy(left_cam)

    # Step 2: Invert world-to-view to get cam-to-world
    cam_to_world = torch.inverse(left_cam.world_view_transform).cpu().numpy()

    # Step 3: Move the camera along the right vector (X axis of cam space)
    # right_direction = cam_to_world[:3, 0]
    right_direction = cam_to_world[0, :3]
    ## right_direction = cam_to_world[2, :3]

    ## cam_to_world[:3, 3] += baseline * right_direction  # Shift along right direction
    cam_to_world[3, :3] += baseline * right_direction
    # cam_to_world[0, :3] += baseline * right_direction

    # Step 4: Recompute world_view_transform (camera to world)
    world_to_view = np.linalg.inv(cam_to_world)
    right_cam.world_view_transform = torch.tensor(world_to_view, dtype=torch.float32, device=left_cam.data_device)

    # Step 5: Keep projection matrix unchanged (do not alter FOV or depth settings)
    right_cam.full_proj_transform = right_cam.world_view_transform @ right_cam.projection_matrix

    # Step 6: Recompute camera center (camera position in world space)
    right_cam.camera_center = torch.inverse(right_cam.world_view_transform)[3, :3]

    return right_cam


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    t_in_lrs_path = os.path.join(model_path, name, "ours_{}".format(iteration), "LR_t_in")
    t_in_stereos_path = os.path.join(model_path, name, "ours_{}".format(iteration), "stereo_t_in")

    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(t_in_lrs_path, exist_ok=True)
    makedirs(t_in_stereos_path, exist_ok=True)

    ply_path = os.path.join(model_path, 'point_cloud', f'iteration_{iteration}', 'point_cloud.ply')


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        # right_view = create_right_eye_camera(view)
        ply_data = o3d.io.read_point_cloud(ply_path)
        center_point = ply_data.get_center()
        left_view, right_view = create_toe_in_stereo_cameras(center_point, view)

        left_img = render(left_view, gaussians, pipeline, background)["render"]
        right_img = render(right_view, gaussians, pipeline, background)["render"]

        # Combine stereo pair: e.g. side-by-side
        stereo_img = torch.cat([left_img, right_img], dim=-1)

        torchvision.utils.save_image(stereo_img, os.path.join(t_in_stereos_path, f"{idx:05d}_stereo.png"))
        torchvision.utils.save_image(left_img, os.path.join(t_in_lrs_path, f"{idx:05d}_left.png"))
        torchvision.utils.save_image(right_img, os.path.join(t_in_lrs_path, f"{idx:05d}_right.png"))

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)