import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from glob import glob
import os
from loguru import logger
from pytorch3d.io import load_obj
from pytorch3d.renderer import RasterizationSettings, PointLights, MeshRenderer, MeshRasterizer, TexturesVertex, SoftPhongShader, look_at_view_transform, PerspectiveCameras, BlendParams
from pytorch3d.structures import Meshes
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from flame.FLAME import FLAME, FLAMETex
from renderer import Renderer
from configs.config import parse_args, get_cfg_defaults, update_cfg
from pathlib import Path


import matplotlib.pyplot as plt

class UVRenderer:
    
    def __init__(self, config, image_size, device):
        self.config = config
        self.device = device
        self.image_size = image_size
        self.setup_renderer()
        self.mask_image = torch.from_numpy(self.get_mask_image()).to(self.device)

    def get_mask_image(self):
        mask_image = cv2.imread(self.config.mask_image_path, cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.resize(mask_image, self.image_size[0])
        mask_image = mask_image.astype(np.float32) / 255.0
        return mask_image

    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][1].item()
    
    def setup_renderer(self):
        mesh_file = self.config.flame_template_path
        self.config.image_size = self.get_image_size()
        self.flame = FLAME(self.config).to(self.device)
        self.flametex = FLAMETex(self.config).to(self.device)
        self.diff_renderer = Renderer(self.image_size, obj_filename=mesh_file).to(self.device)
        self.faces = load_obj(mesh_file)[1]
        self.uvs = load_obj(mesh_file)[-1].verts_uvs

        self.uvs = self.uvs * 2 - 1
        self.uvs[:, 1] = -self.uvs[:, 1]

        raster_settings = RasterizationSettings(
            image_size=self.get_image_size(),
            faces_per_pixel=1,
            cull_backfaces=True,
            perspective_correct=True
        )

        self.blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=[0, 0, 0])

        self.lights = PointLights(
            device=self.device,
            location=((0.0, 0.0, 5.0),),
            ambient_color=((0.5, 0.5, 0.5),),
            diffuse_color=((0.5, 0.5, 0.5),)
        )

        self.mesh_rasterizer = MeshRasterizer(raster_settings=raster_settings)
        self.debug_renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(device=self.device, lights=self.lights)
        )
        
    def update_frame(self, frame_checkpoint):
        
        payload = torch.load(frame_checkpoint, map_location=self.device)

        camera_params = payload['camera']
        self.R = torch.from_numpy(camera_params['R']).to(self.device)
        self.t = torch.from_numpy(camera_params['t']).to(self.device)
        self.focal_length = torch.from_numpy(camera_params['fl']).to(self.device)
        self.principal_point = torch.from_numpy(camera_params['pp']).to(self.device)

        flame_params = payload['flame']
        self.tex = torch.from_numpy(flame_params['tex']).to(self.device)
        self.exp = torch.from_numpy(flame_params['exp']).to(self.device)
        self.sh = torch.from_numpy(flame_params['sh']).to(self.device)
        self.shape = torch.from_numpy(flame_params['shape']).to(self.device)
        self.mica_shape = torch.from_numpy(flame_params['shape']).to(self.device)
        self.eyes = torch.from_numpy(flame_params['eyes']).to(self.device)
        self.eyelids = torch.from_numpy(flame_params['eyelids']).to(self.device)
        self.jaw = torch.from_numpy(flame_params['jaw']).to(self.device)

        self.frame = int(payload['frame_id'])
        self.image_size = torch.from_numpy(payload['img_size'])[None].to(self.device)

    def get_uv(self):
        self.diff_renderer.rasterizer.reset()

        self.cameras = PerspectiveCameras(
            device=self.device,
            principal_point=self.principal_point,
            focal_length=self.focal_length,
            R=rotation_6d_to_matrix(self.R), T=self.t,
            image_size=self.image_size
        )
        vertices, lmk68, lmkMP = self.flame(
            cameras=torch.inverse(self.cameras.R),
            shape_params=self.shape,
            expression_params=self.exp,
            eye_pose_params=self.eyes,
            jaw_pose_params=self.jaw,
            eyelid_params=self.eyelids
        )
        B = vertices.shape[0]
        faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
        meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)])

        fragments = self.mesh_rasterizer(meshes_world, cameras=self.cameras)

        uv_map = self.fragment_to_uv(fragments)
        return uv_map[0].permute((1, 2, 0)).cpu().numpy()

    def get_mask(self, jaw_values=[0.1, -0.2, -0.4]):

        B = self.jaw.shape[0]
        texture_maps = (
            self.mask_image[None, ...].repeat(B, 1, 1, 1).to(self.device).float()
        )

        # Don't need to call self.cameras here because it's already been called in get_uv
        jaw = self.jaw.clone()
        output = torch.zeros(*self.get_image_size()).to(self.device)


        for jaw_value in [*jaw_values, jaw[0, 5]]:
            jaw = self.jaw.clone()
            jaw[..., 5] = jaw_value
            vertices, lmk68, lmkMP = self.flame(
                cameras=torch.inverse(self.cameras.R),
                shape_params=self.shape,
                expression_params=self.exp,
                eye_pose_params=self.eyes,
                jaw_pose_params=jaw,
                eyelid_params=self.eyelids
            )
            B = vertices.shape[0]
            faces = self.faces.verts_idx.cuda()[None].repeat(B, 1, 1)
            meshes_world = Meshes(verts=[vertices[i] for i in range(B)], faces=[faces[i] for i in range(B)])

            fragments = self.mesh_rasterizer(meshes_world, cameras=self.cameras)
            uv_map = self.fragment_to_uv(fragments)

            # Sample Mask Texture
            alpha = (1 - (uv_map == 0).all(dim=1, keepdim=True).float())
            mask = F.grid_sample(
                texture_maps,
                uv_map.permute(0, 2, 3, 1)[..., :2],
                align_corners=True,
                padding_mode="reflection"
            )

            mask = (mask * alpha).squeeze()
            output = torch.maximum(output, mask)

        return output[..., None].detach().cpu().numpy()

    def fragment_to_uv(self, fragments):
        N, H_out, W_out, K = fragments.pix_to_face.shape

        uvs, face_uvs = self.uvs, self.faces.textures_idx

        uvs = uvs[face_uvs, ..., :2].to(self.device)
        uvs = uvs[None].repeat(N, 1, 1, 1).reshape((-1, 3, 2))
        raster = interpolate_face_attributes(fragments.pix_to_face, fragments.bary_coords, uvs)

        raster = torch.cat((raster, torch.zeros_like(raster[..., 0:1])), dim=-1)

        render = hard_rgb_blend(raster, fragments, self.blend_params).permute((0, 3, 1, 2))
        return render[:, :3]

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    cfg.config_name = Path(args.cfg).stem

    checkpoints = glob(os.path.join(args.checkpoint_dir, '*.frame'))
    checkpoints = sorted(checkpoints)
    frame_checkpoint = checkpoints[0]
    image_size = [torch.load(frame_checkpoint, map_location='cpu')['img_size']]

    model = UVRenderer(cfg, image_size, 'cuda:0')

    for i, frame_checkpoint in enumerate(checkpoints):
        print(f'Frame {i}')
        model.update_frame(frame_checkpoint)
        uv = model.get_uv()

        # Map uvs to [0, 1]
        uvs_mask = (uv == 0).all(axis=-1)
        uv = (uv + 1) / 2
        uv[uvs_mask] = 1

        mask = model.get_mask()

        out = np.concatenate((uv[..., :2], mask), axis=-1)
        #out = out.permute((1, 2, 0)).cpu().numpy()
        out = (out * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.output_dir, f'{i:05d}.png'), out)
