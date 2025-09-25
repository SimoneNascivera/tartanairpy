"""
Author: Yuchen Zhang
Date: 2024-07-01

Example script for sample new correspondence from the TartanAir dataset.
"""

# General imports.
import sys
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import flow_vis

# Local imports.
sys.path.append("..")
from tartanair.flow_calculation import *
from tartanair.flow_utils import *

import multiprocessing

# Create a TartanAir object.
tartanair_data_root = "/home/nasciver/tartanair/dataset"

# Create the requested camera models and their parameters.
cam_config = {
    "type": "pinhole",
    "params": {"fx": 320, "fy": 320, "cx": 320, "cy": 320, "width": 640, "height": 640},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

rotation = torch.from_numpy(
    np.array(
        [
            [1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 1.0000000],
            [0.0000000, -1.0000000, 0.0000000],
        ],
        dtype=np.float32,
    )
)

cam_model = generate_camera_model_object_from_config(cam_config)

loaded_poses = {}


def get_cam_pose(camera_path, index):
    if camera_path not in loaded_poses:
        poses = np.loadtxt(f"{camera_path.replace("image", "pose")}.txt")
        loaded_poses[camera_path] = poses

    return loaded_poses[camera_path][index]


def read_decode_depth(depthpath):
    depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)


def load_and_convert_tensor(camera_path: str, traj_index: dict, j: dict, device):
    rendered = {}
    rendered["image"] = cv2.imread(
        f"{camera_path}/{j:06d}_{traj_index["cam_side"]}_{traj_index["cam_orient"]}.png"
    )
    rendered["depth"] = read_decode_depth(
        f"{camera_path.replace("image", "depth")}/{j:06d}_{traj_index["cam_side"]}_{traj_index["cam_orient"]}_depth.png"
    )

    for key in rendered.keys():
        if key == "image":
            rendered[key] = cv2.cvtColor(rendered[key], cv2.COLOR_BGR2RGB)
            rendered[key] = (
                torch.from_numpy(rendered[key]).to(device).float() / 255.0
            ).permute(2, 0, 1)
        elif key == "depth":
            rendered[key] = (
                torch.from_numpy(rendered[key]).to(device).unsqueeze(0).float()
            )
        else:
            raise NotImplementedError("Unknown key: {}".format(key))
    mask = torch.zeros_like(rendered["depth"], dtype=torch.bool)

    return rendered, mask


def export_flow(camera_path: str, traj_index: dict, j: int):
    print(camera_path)
    flow_path = camera_path.replace("image", "flow")
    print(f"Processing sequence {traj_index} frames: {j} and {j+1}")

    if os.path.exists(
        os.path.join(
            flow_path,
            f"{j:06d}_{traj_index["cam_side"]}_{traj_index["cam_orient"]}_flow.npy",
        )
    ):
        print("Skpping, flow data exists")

    # get pose
    pose0 = get_cam_pose(camera_path, j)
    pose1 = get_cam_pose(camera_path, j + 1)

    rendered_0, mask_0 = load_and_convert_tensor(
        camera_path, traj_index, j, device=device
    )
    rendered_1, mask_1 = load_and_convert_tensor(
        camera_path, traj_index, j + 1, device=device
    )

    # compute correspondence
    (
        depth_value_gt,
        depth_error,
        fov_mask,
        valid_pixels_0,
        valid_pixels_1,
        flow_image,
        world_T_0,
        world_T_1,
    ) = calculate_pairwise_flow(
        pose0,
        rotation,
        rendered_0["depth"].to(device),
        mask_0.to(device),
        cam_model,
        pose1,
        rotation,
        rendered_1["depth"].to(device),
        mask_1.to(device),
        cam_model,
        device=device,
    )

    # compute occlusion
    non_occluded_prob = calculate_occlusion(
        rendered_0["depth"].to(device),
        rendered_1["depth"].to(device),
        valid_pixels_0,
        valid_pixels_1,
        depth_value_gt,
        device=device,
        depth_start_threshold=0.04,
        depth_temperature=0.02,
        apply_relative_error=True,
        relative_error_tol=0.01,
    )

    valid = torch.zeros(fov_mask.shape, device=fov_mask.device, dtype=torch.float32)
    valid[fov_mask] = non_occluded_prob  # probability of not occluded is less then 0.5

    flow = flow_image.cpu().numpy()
    np.save(os.path.join(flow_path, f"{j:06d}_lcam_bottom_flow.npy"), flow)


    import cv2
    image_0 = (rendered_0['image'].cpu().numpy() * 255).astype('uint8')
    image_1 = (rendered_1['image'].cpu().numpy() * 255).astype('uint8')
    image_0 = np.ascontiguousarray(image_0)
    image_1 = np.ascontiguousarray(image_1)
    
    point = np.array([600,300])
    cv2.rectangle(image_0, (point[0] - 1,point[1] - 1), (point[0] + 1, point[1] + 1), (255,0,0), 1)

    point1 = point + flow[point[1], point[0]]
    point1 = np.rint(point1).astype(int)
    
    cv2.rectangle(image_1, (point1[0] - 1,point1[1] - 1), (point1[0] + 1, point1[1] + 1), (255,0,0), 1)
    
    cv2.imwrite("output/a.png", np.hstack((image_0, image_1)))
    
    image_np = (rendered_0['image'].cpu().numpy() * 255).astype('uint8')
    cv2.imwrite("output/rgb_0.png", image_np)
    
    image_np = (rendered_1['image'].cpu().numpy() * 255).astype('uint8')
    cv2.imwrite("output/rgb_1.png", image_np)
    
    image_np = (mask_0.cpu().numpy() * 255).astype('uint8')
    cv2.imwrite("output/mask_0.png", image_np)
    
    image_np = (mask_1.cpu().numpy() * 255).astype('uint8')
    cv2.imwrite("output/mask_1.png", image_np)
    
    image_np = (np.log(rendered_0['depth'].cpu().numpy()) * 255).astype('uint8')
    cv2.imwrite("output/depth_0.png", image_np)
    
    image_np = (np.log(rendered_1['depth'].cpu().numpy()) * 255).astype('uint8')
    cv2.imwrite("output/depth_1.png", image_np)
    
    image_np = (flow_vis.flow_to_color(flow_image.cpu().numpy()) * 255).astype('uint8')
    cv2.imwrite("output/flow.png", image_np)
    
    image_np = (fov_mask.cpu().numpy() * 255).astype('uint8')
    cv2.imwrite("output/fov_mask.png", image_np)
    
    image_np = (valid.cpu().numpy() * 255).astype('uint8')
    cv2.imwrite("output/not_occluded.png", image_np)
        

if __name__ == "__main__":
    traj_index_list = []
    traj_length = []
    camera_orientation = []
    for i in os.walk(tartanair_data_root):
        if "image_lcam" in i[0].split("/")[-1]:
            env = i[0].split("/")[-4]
            difficulty = i[0].split("/")[-3].split("_")[1]
            id = i[0].split("/")[-2]
            count = len(i[2])

            traj_index = {
                "env": env,
                "difficulty": difficulty,
                "id": id,
                "cam_side": "lcam",
                "cam_orient": i[0].split("/")[-1].split("_")[-1],
            }
            traj_index_list.append(traj_index)
            traj_length.append(count)

    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    # for each sequence
    for i in range(len(traj_index_list)):
        print(
            f"Processing sequence {traj_index_list[i]["env"]} : {traj_index_list[i]["id"]}"
        )
        camera_path = os.path.join(
            tartanair_data_root,
            traj_index_list[i]["env"],
            f"Data_{traj_index_list[i]["difficulty"]}",
            traj_index_list[i]["id"],
            f"image_{traj_index_list[i]["cam_side"]}_{traj_index_list[i]["cam_orient"]}",
        )
        os.makedirs(camera_path, exist_ok=True)

        args = [(camera_path, traj_index_list[i], j) for j in range(traj_length[i] - 1)]

        export_flow(args[0][0], args[0][1], args[0][2])
        exit(0)
        # with multiprocessing.Pool(1) as pool:
        #    pool.starmap(export_flow, args)
        for arg in args:
            export_flow(*arg)
    """
    for j in range(traj_length[i]-1):
        print(f"Processing frames: {j} and {j+1}")
        # get cubemap images. This includes RGB, depth for all 6 faces of the cubemap.
        cubemap_images_0 = random_accessor.get_cubemap_images_parallel(
            traj_index_list[i],
            frame_idx=j
        )

        cubemap_images_1 = random_accessor.get_cubemap_images_parallel(
            traj_index_list[i],
            frame_idx=j+1
        )

        # get pose
        pose0 = random_accessor.get_front_cam_NED_pose(traj_index_list[i], j)
        pose1 = random_accessor.get_front_cam_NED_pose(traj_index_list[i], j+1)

        # move the camera models and images to GPU
        cam_model_obj.device = device # special setter will handle moving internal tensors to the device

        def convert_tensor(rendered, device):

            for key, images in rendered.items():
                if key == 'image':
                    for k, v in images.items():
                        v = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
                        rendered[key][k] = (torch.from_numpy(v).to(device).float() / 255.0).permute(2, 0, 1)
                elif key == 'depth':
                    for k, v in images.items():
                        rendered[key][k] = torch.from_numpy(v).to(device).unsqueeze(0).float()
                else:
                    raise NotImplementedError("Unknown key: {}".format(key))

            return rendered

        cubemap_images_0 = convert_tensor(cubemap_images_0, device=device)
        cubemap_images_1 = convert_tensor(cubemap_images_1, device=device)

        # render the images according to camera models. May select a rotation
        rendered_0, mask_0 = render_images_from_cubemap(cubemap_images_0, cam_model_obj, rotation=rotation, device=device)
        rendered_1, mask_1 = render_images_from_cubemap(cubemap_images_1, cam_model_obj, rotation=rotation, device=device)

        # compute correspondence
        depth_value_gt, depth_error, fov_mask, valid_pixels_0, valid_pixels_1, flow_image, world_T_0, world_T_1 = calculate_pairwise_flow(
            pose0, rotation, rendered_0['depth'].to(device), mask_0.to(device), cam_model_obj, 
            pose1, rotation, rendered_1['depth'].to(device), mask_1.to(device), cam_model_obj,
            device=device
        )

        # compute occlusion
        non_occluded_prob = calculate_occlusion(
            rendered_0['depth'].to(device), rendered_1['depth'].to(device), 
            valid_pixels_0, valid_pixels_1,
            depth_value_gt, device=device,
            depth_start_threshold=0.04,
            depth_temperature=0.02,
            apply_relative_error=True,
            relative_error_tol=0.01
        )

        valid = torch.zeros(fov_mask.shape, device=fov_mask.device, dtype=torch.float32)
        valid[fov_mask] = non_occluded_prob # probability of not occluded is less then 0.5

        flow = flow_image.cpu().numpy()
        np.save(os.path.join(flow_path, f"{j:06d}_lcam_bottom_flow.npy"), flow)
        
          # visualize everything
        #fig, axs = plt.subplots(3, 3, figsize=(20, 10))

        #import cv2
        #image_0 = (rendered_0['image'].cpu().numpy() * 255).astype('uint8')
        #image_1 = (rendered_1['image'].cpu().numpy() * 255).astype('uint8')
        #image_0 = np.ascontiguousarray(image_0)
        #image_1 = np.ascontiguousarray(image_1)
        #
        #point = np.array([600,300])
        #cv2.rectangle(image_0, (point[0] - 1,point[1] - 1), (point[0] + 1, point[1] + 1), (255,0,0), 1)
        
        #print(flow.shape)
        #point1 = point + flow[point[1], point[0]]
        #point1 = np.rint(point1).astype(int)
        #
        #cv2.rectangle(image_1, (point1[0] - 1,point1[1] - 1), (point1[0] + 1, point1[1] + 1), (255,0,0), 1)
        #
        #cv2.imwrite("output/a.png", np.hstack((image_0, image_1)))
        #
        #image_np = (rendered_0['image'].cpu().numpy() * 255).astype('uint8')
        #cv2.imwrite("output/rgb_0.png", image_np)
        #
        #image_np = (rendered_1['image'].cpu().numpy() * 255).astype('uint8')
        #cv2.imwrite("output/rgb_1.png", image_np)
        #
        #image_np = (mask_0.cpu().numpy() * 255).astype('uint8')
        #cv2.imwrite("output/mask_0.png", image_np)
        #
        #image_np = (mask_1.cpu().numpy() * 255).astype('uint8')
        #cv2.imwrite("output/mask_1.png", image_np)
        #
        #image_np = (np.log(rendered_0['depth'].cpu().numpy()) * 255).astype('uint8')
        #cv2.imwrite("output/depth_0.png", image_np)
        #
        #image_np = (np.log(rendered_1['depth'].cpu().numpy()) * 255).astype('uint8')
        #cv2.imwrite("output/depth_1.png", image_np)
        #
        #image_np = (flow_vis.flow_to_color(flow_image.cpu().numpy()) * 255).astype('uint8')
        #cv2.imwrite("output/flow.png", image_np)
        #
        #image_np = (fov_mask.cpu().numpy() * 255).astype('uint8')
        #cv2.imwrite("output/fov_mask.png", image_np)
        #
        #image_np = (valid.cpu().numpy() * 255).astype('uint8')
        #cv2.imwrite("output/not_occluded.png", image_np)
    """
