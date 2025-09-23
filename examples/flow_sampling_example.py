'''
Author: Yuchen Zhang
Date: 2024-07-01

Example script for sample new correspondence from the TartanAir dataset.
'''

# General imports.
import sys
import torch
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import flow_vis

# Local imports.
sys.path.append('..')
from tartanair.flow_calculation import *
from tartanair.flow_utils import *

import tartanair as ta

import multiprocessing

# Create a TartanAir object.
tartanair_data_root = '/home/nasciver/tartanair/dataset'
ta.init(tartanair_data_root)

# Create the requested camera models and their parameters.
cam_model = {'name': 'pinhole', 
               'params': 
                        {'fx': 320, 'fy': 320, 'cx': 320, 'cy': 320, 'width': 640, 'height': 640},
                }

device = "cuda" if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

random_accessor = ta.get_random_accessor()
random_accessor.cache_tartanair_pose()

cam_model_obj = random_accessor.generate_camera_model_object_from_config(cam_model)

rotation = torch.from_numpy(np.array(   [[1.0000000,  0.0000000, 0.0000000],
                                            [0.0000000,  0.0000000,  1.0000000],
                                            [0.0000000,  -1.0000000,  0.0000000]], dtype=np.float32))

def export_flow(flow_path, traj_index, j):
    print(f"Processing sequence {traj_index} frames: {j} and {j+1}")
    if os.path.exists(os.path.join(flow_path, f"{j:06d}_lcam_bottom_flow.npy")):
        print("Skpping, flow data exists")
        
    # get cubemap images. This includes RGB, depth for all 6 faces of the cubemap.
    cubemap_images_0 = random_accessor.get_cubemap_images_parallel(
        traj_index,
        frame_idx=j
    )

    cubemap_images_1 = random_accessor.get_cubemap_images_parallel(
        traj_index,
        frame_idx=j+1
    )

    # get pose
    pose0 = random_accessor.get_front_cam_NED_pose(traj_index, j)
    pose1 = random_accessor.get_front_cam_NED_pose(traj_index, j+1)

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
    
if __name__ == '__main__':
    traj_index_list = []
    traj_length = []
    for i in os.walk(tartanair_data_root):
        if "image_lcam_back" in i[0].split("/")[-1]:
            env = i[0].split("/")[-4]
            difficulty = i[0].split("/")[-3].split("_")[1]
            id = i[0].split("/")[-2]
            count = len(i[2])

            traj_index = {
                "env" : env,
                "difficulty": difficulty,
                "id": id,
                "cam_side": "lcam"
            }
            traj_index_list.append(traj_index)
            traj_length.append(count)

    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)

    # for each sequence
    for i in range(len(traj_index_list)):
        print(f"Processing sequence {traj_index_list[i]["env"]} : {traj_index_list[i]["id"]}")
        flow_path = os.path.join(tartanair_data_root, traj_index_list[i]["env"], f"Data_{traj_index_list[i]["difficulty"]}", traj_index_list[i]["id"], "flow_lcam_bottom")
        os.makedirs(flow_path, exist_ok=True)

        args = [(flow_path, traj_index_list[i], j) for j in range(traj_length[i]-1)]

        #with multiprocessing.Pool(1) as pool:
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