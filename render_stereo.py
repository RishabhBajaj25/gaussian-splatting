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

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

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
    lrs_path = os.path.join(model_path, name, "ours_{}".format(iteration), "LR")
    stereos_path = os.path.join(model_path, name, "ours_{}".format(iteration), "stereo")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(lrs_path, exist_ok=True)
    makedirs(stereos_path, exist_ok=True)


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        right_view = create_right_eye_camera(view)

        left_img = render(view, gaussians, pipeline, background)["render"]
        right_img = render(right_view, gaussians, pipeline, background)["render"]

        # Combine stereo pair: e.g. side-by-side
        stereo_img = torch.cat([left_img, right_img], dim=-1)

        torchvision.utils.save_image(stereo_img, os.path.join(stereos_path, f"{idx:05d}_stereo.png"))
        torchvision.utils.save_image(left_img, os.path.join(lrs_path, f"{idx:05d}_left.png"))
        torchvision.utils.save_image(right_img, os.path.join(lrs_path, f"{idx:05d}_right.png"))

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