""" PointNet2 backbone for feature learning.
    Author: Charles R. Qi
"""
import copy
import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule
from pointnet2.pointnet2_utils import  furthest_point_sample
from openpoints.models.build import MODELS, build_model_from_cfg
from knn.knn_modules import knn
import open3d as o3d
import numpy as np
import time


@MODELS.register_module()
class SpoTrBackbone(nn.Module):
    def __init__(self, encoder_args=None, decoder_args=None):
        super().__init__()
        
        self.encoder = build_model_from_cfg(encoder_args)
        decoder_args_merged_with_encoder = copy.deepcopy(encoder_args)
        decoder_args_merged_with_encoder.update(decoder_args)
        decoder_args_merged_with_encoder.encoder_channel_list = self.encoder.channel_list if hasattr(self.encoder,
                                                                                                    'channel_list') else None
        self.decoder = build_model_from_cfg(decoder_args_merged_with_encoder)
    
    # data (B, N, 3) 场景点云
    def forward(self, data, end_points=None):
        # p, f, idx = self.encoder.forward_seg_feat(data)
        # f = self.decoder(p, f).squeeze(-1)
        # num_seed = p[3].shape[1]
        # end_points['fp2_inds'] = idx[2][:, :num_seed] # (B, M)
        # end_points['fp2_features'] = f # (B, 256, M)
        # end_points['fp2_xyz'] = p[3] # (B, M, 3)
        # return f, end_points['fp2_xyz'], end_points

        p, f, idx, global_p = self.encoder.forward_seg_feat(data)
        # p, f, idx = self.encoder.forward_seg_feat(data, color0=end_points["cloud_colors"])
        # (B, 64, N)
        f = self.decoder(p, f)
        # toc = time.time()
        # print(f"autoencoder time: {(toc - tic) * 1000} ms.")
        # (B, 1024)
        obj_sampled_inds = end_points['pcd_obj_inds'].long()
        # (B, 64, 1024)
        obj_sampled_f = torch.gather(f.transpose(1, 2), 1, obj_sampled_inds.unsqueeze(-1).expand(-1, -1, f.shape[1])).transpose(1, 2)
        # (B, 1024, 3)
        obj_sampled_xyz = torch.gather(data, 1, obj_sampled_inds.unsqueeze(-1).expand(-1, -1, 3))

        # # 场景点云
        # last_pcd = data[-1]
        # # 物体样本点云
        # last_pcd_obj = obj_sampled_xyz[-1]
        # cloud = o3d.geometry.PointCloud()
        # cloud.points = o3d.utility.Vector3dVector(last_pcd.cpu().numpy().astype(np.float64))
        # o3d.io.write_point_cloud("test_pcd/pcd.ply", cloud)
        # cloud.points = o3d.utility.Vector3dVector(last_pcd_obj.cpu().numpy().astype(np.float64))
        # o3d.io.write_point_cloud("test_pcd/sampled_xyz.ply", cloud)

        end_points['fp2_inds'] = obj_sampled_inds
        end_points['fp2_features'] = obj_sampled_f
        end_points['fp2_xyz'] = obj_sampled_xyz
        end_points['global_p'] = global_p

        return end_points['fp2_features'], end_points['fp2_xyz'], end_points


class Pointnet2Backbone(nn.Module):
    r"""
       Backbone network for point cloud feature learning.
       Based on Pointnet++ single-scale grouping network. 
        
       Parameters
       ----------
       input_feature_dim: int
            Number of input channels in the feature descriptor for each point.
            e.g. 3 for RGB.
    """
    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.04,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=0.3,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

    def _break_up_pc(self, pc): # 分离坐标和其他特征（比如RGB通道）
        xyz = pc[..., 0:3].contiguous()
        features = (pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None)

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,D,K)
                XXX_inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        end_points['input_xyz'] = xyz
        end_points['input_features'] = features

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds # (B, 2048) 0, 14884,  1036,  2539, 17838, 15165, 18030, 14541, 15419,  4530....
        end_points['sa1_xyz'] = xyz 
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds

        return features, end_points['fp2_xyz'], end_points

