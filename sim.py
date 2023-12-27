import numpy as np
import torch
from experiment.environment import Environment
from models.graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
import open3d as o3d
import argparse
from utils.collision_detector import ModelFreeCollisionDetector

# Init the model
net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05, 
               hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# Load checkpoint
checkpoint_dir = "logs/log_rs_spotr/2023-10-06-23-21/checkpoint.tar"
checkpoint = torch.load(checkpoint_dir)
net.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print(f"-> loaded checkpoint {checkpoint_dir} (epoch: {start_epoch})")
net.eval()

parser = argparse.ArgumentParser()
parser.add_argument('--vis', type=int, default=0)
parser.add_argument('--num_objs', type=int, default=5)
parser.add_argument('--num_ites', type=int, default=10)
cfgs = parser.parse_args()

env = Environment(vis=cfgs.vis)

total_attempt = 0
total_success = 0
total_colli = 0
total_unstable = 0

for i in range(cfgs.num_ites):
    terminated = False
    num_attem = 0
    num_success = 0
    num_colli = 0
    num_unstable = 0
    num_scene = 0
    num_objects = cfgs.num_objs
    end_points = {}

    pcd_scene, pcd_obj_inds, o3d_scene, o3d_obj, seg_obj, info = env.reset()
    cloud = pcd_scene.copy()
    pcd_scene = np.expand_dims(pcd_scene, axis=0)
    pcd_scene = torch.Tensor(pcd_scene).to(device)
    pcd_obj_inds = np.expand_dims(pcd_obj_inds, axis=0)
    pcd_obj_inds = torch.Tensor(pcd_obj_inds).to(device)
    end_points.update({"point_clouds": pcd_scene})
    end_points.update({"pcd_obj_inds": pcd_obj_inds})
    while num_scene < 15 and terminated == False:
        num_scene += 1
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        # 取出第一个场景的预测结果 (1024，17)
        preds = grasp_preds[0].detach().cpu().numpy()

        # grasp_pts = preds[:, 13:16]
        # pcd_grasp_pts = toOpen3dCloud(grasp_pts)
        # o3d.io.write_point_cloud("grasp_pts.ply", pcd_grasp_pts)
        gg = GraspGroup(preds)
        gg.object_ids = seg_obj
        

        # grippers = gg.to_open3d_geometry_list()
        # if len(grippers) > 50:
        #     grippers = grippers[:50]
        # o3d.visualization.draw_geometries([o3d_scene, *grippers])

        # 按照点方向和地面的夹角重新打分
        # (3, )
        ground_normal = env.camera_from_ground_normal
        # (1024, 3)
        scene_normal = np.array(o3d_obj.normals)
        # (1024, )
        normal_score = scene_normal @ ground_normal
        # （1024, )
        quality_score = preds[:, 0]
        new_score = 0.7 * normal_score + 0.3 * quality_score
        gg.scores = new_score
        
        # grippers = gg.to_open3d_geometry_list()
        # if len(grippers) > 50:
        #     grippers = grippers[:50]
        # o3d.visualization.draw_geometries([o3d_scene, *grippers])

        nms_gg = gg.nms()
        # print(f"scene: {num_scene}. grasp nms rate: {len(nms_gg)} / {len(gg)}")
        
        # grippers = nms_gg.to_open3d_geometry_list()
        # if len(grippers) > 50:
        #     grippers = grippers[:50]
        # o3d.visualization.draw_geometries([o3d_scene, *grippers])

        # 碰撞检测
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01)
        collision_mask = mfcdetector.detect(nms_gg, approach_dist=0.05, collision_thresh=0.01)
        collision_free_gg = nms_gg[~collision_mask]
        # print(f"碰撞检测：{len(collision_free_gg)} / {len(nms_gg)}")

        bg_ids = []
        for g_id in range(len(collision_free_gg)):
            g = collision_free_gg[g_id]
            if g.object_id == 0:
                bg_ids.append(g_id)
        print(f"删除 {len(bg_ids)} 个背景抓取点")
        collision_free_gg.remove(bg_ids)

        # collision_free_grippers = collision_free_gg.to_open3d_geometry_list()
        # if len(collision_free_grippers) > 50:
        #     collision_free_grippers = collision_free_grippers[:50]
        # o3d.visualization.draw_geometries([o3d_scene, *collision_free_grippers])
        
        collision_free_gg = collision_free_gg.sort_by_score()
        pcd_scene, pcd_obj_inds, o3d_scene, o3d_obj, seg_obj, terminated, info = env.step(collision_free_gg)
        num_attem += info["num_attem"]
        num_colli += info["num_colli_grasp"]
        num_unstable += info["num_unstable_grasp"]
        if info["is_success"] == True:
            num_success += 1

        if terminated == True:
            break
        
        pcd_scene = np.expand_dims(pcd_scene, axis=0)
        pcd_scene = torch.Tensor(pcd_scene).to(device)
        pcd_obj_inds = np.expand_dims(pcd_obj_inds, axis=0)
        pcd_obj_inds = torch.Tensor(pcd_obj_inds).to(device)
        end_points.update({"point_clouds": pcd_scene})
        end_points.update({"pcd_obj_inds": pcd_obj_inds})

    eps = np.finfo(np.float32).eps
    complete_rate = num_success / num_objects
    # success_rate = num_success / (num_attem - num_colli + eps)
    success_rate = num_success / num_attem
    colli_rate = num_colli / num_attem
    unstable_rate = num_unstable / (num_attem - num_colli + eps)

    total_attempt += num_attem
    total_success += num_success
    total_colli += num_colli
    total_unstable += num_unstable

    print(f"num_objects: {num_objects}, num_attem: {num_attem}, num_colli: {num_colli}, num_success: {num_success},  num_unstable: {num_unstable}")
    print(f"Complete rate: {complete_rate:.2f}")
    print(f"Success rate: {success_rate:.2f}")
    print(f"Collision rate: {colli_rate:.2f}")
    print(f"Unstable rate: {unstable_rate:.2f}")
    # if total_attempt >= 20:
    #     break

print(f"total_attempt: {total_attempt}, total_success: {total_success}, success_rate: {(total_success/total_attempt):.2f}")
print(f"total_colli: {(total_colli/total_attempt):.2f}, total_unstable: {(total_unstable/(total_attempt-total_colli)):.2f}")
env.close()

#     complete_rate = num_success / num_objects
#     success_rate = num_success / (num_attem - num_colli + 0.0000001)
#     colli_rate = num_colli / num_attem
#     unstable_rate = num_unstable / (num_attem - num_colli + 0.0000001)


#     print(f"num_objects: {num_objects}, num_attem: {num_attem}, num_colli: {num_colli}, num_success: {num_success},  num_unstable: {num_unstable}")
#     print(f"Complete rate: {complete_rate:.2f}")
#     print(f"Success rate: {success_rate:.2f}")
#     print(f"Collision rate: {colli_rate:.2f}")
#     print(f"Unstable rate: {unstable_rate:.2f}")

# env.close()