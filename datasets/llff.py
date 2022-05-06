import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from pathlib import Path
import json

from .ray_utils import *


def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.
    
    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0) # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1) # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses) # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg # convert to homogeneous coordinate for faster computation
                                 # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1)) # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1) # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3] # (N_images, 3, 4)

    return poses_centered, pose_avg


def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4*np.pi, n_poses+1)[:-1]: # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5*t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))
        
        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0]) # (3)
        x = normalize(np.cross(y_, z)) # (3)
        y = np.cross(z, x) # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)] # (3, 4)

    return np.stack(poses_spiral, 0) # (n_poses, 3, 4)


def create_spheric_poses(radius, n_poses=120):
    """
    Create circular poses around z axis.
    Inputs:
        radius: the (negative) height and the radius of the circle.

    Outputs:
        spheric_poses: (n_poses, 3, 4) the poses in the circular path
    """
    def spheric_pose(theta, phi, radius):
        trans_t = lambda t : np.array([
            [1,0,0,0],
            [0,1,0,-0.9*t],
            [0,0,1,t],
            [0,0,0,1],
        ])

        rot_phi = lambda phi : np.array([
            [1,0,0,0],
            [0,np.cos(phi),-np.sin(phi),0],
            [0,np.sin(phi), np.cos(phi),0],
            [0,0,0,1],
        ])

        rot_theta = lambda th : np.array([
            [np.cos(th),0,-np.sin(th),0],
            [0,1,0,0],
            [np.sin(th),0, np.cos(th),0],
            [0,0,0,1],
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
        return c2w[:3]

    spheric_poses = []
    for th in np.linspace(0, 2*np.pi, n_poses+1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi/5, radius)] # 36 degree view downwards
    return np.stack(spheric_poses, 0)


class LLFFDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(504, 378), spheric_poses=False, val_num=1):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num) # at least 1
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        poses_bounds = np.load(os.path.join(self.root_dir,
                                            'poses_bounds.npy')) # (N_images, 17)

        # read metadata json if exists
        # meta_path = Path(self.root_dir / 'metadata.json')
        # if meta_path.exists():
        #     with meta_path.open('r') as f:
        #         metadata = json.load(f)
        # else:
        #     metadata = {}
        
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
                        # load full resolution image then resize
        if self.split in ['train', 'val', 'all_val']:
            assert len(poses_bounds) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        poses = poses_bounds[:, :15].reshape(-1, 3, 5) # (N_images, 3, 5)
        self.bounds = poses_bounds[:, -2:] # (N_images, 2)

        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1] # original intrinsics, same for all images
        # assert H*self.img_wh[0] == W*self.img_wh[1], \
        #     f'You must set @img_wh to have the same aspect ratio as ({W}, {H}) !'
        
        self.focal *= self.img_wh[0]/W

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        # [-u, r, -t] -> [r, u, -t]
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
                # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses)
        distances_from_center = np.linalg.norm(self.poses[..., 3], axis=1)
        val_ids = [np.argmin(distances_from_center)] # choose val image as the closest to
                                                   # center image

        # read split json if exists
        split_path = Path(self.root_dir) / 'split.json'
        if split_path.exists():
            with split_path.open('r') as f:
                split_data = json.load(f)
            val_ids = split_data['val_ids']
            
        self.val_ids = val_ids
        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = self.img_wh[0]/2
        self.K[1, 2] = self.img_wh[1]/2

        np.save(str(Path(self.root_dir) / 'centered_pose.npy'), self.poses)
        np.save(str(Path(self.root_dir) / 'pose.npy'), poses)


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.K) # (H, W, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i in val_ids: # exclude the val image
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = i * torch.ones(len(rays_o), 1)

                # img_name = Path(image_path).stem
                # # load time frame from metadata if exists
                # if img_name in metadata:
                #     rays_t = metadata[img_name] * torch.ones(len(rays_o), 1)

                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                                     # near plane is always at 1.0
                                     # near and far in NDC are always 0 and 1
                                     # See https://github.com/bmild/nerf/issues/34
                else:
                    near = self.bounds.min()
                    far = min(8 * near, self.bounds.max()) # focus on central object only

                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 8)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split == 'val':
            print('val image is', [self.image_paths[val_idx] for val_idx in val_ids])

        elif self.split == 'all_val':
            print('val image is', self.image_paths)

        else: # for testing, create a parametric rendering path
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.poses
            elif not self.spheric_poses:
                focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                  # given in the original repo. Mathematically if near=1
                                  # and far=infinity, then this number will converge to 4
                radii = np.percentile(np.abs(self.poses[..., 3]), 90, axis=0)
                self.poses_test = create_spiral_poses(radii, focus_depth)
            else:
                radius = 1.1 * self.bounds.min()
                self.poses_test = create_spheric_poses(radius)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return len(self.val_ids)
        if self.split == 'all_val':
            return len(self.image_paths)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx]}
        else:
            if self.split == 'val':
                idx = self.val_ids[idx]
            c2w = torch.FloatTensor(self.poses[idx])
            # if self.split == 'val':
            # else:
            #     c2w = torch.FloatTensor(self.poses_test[idx])

            t = 0 # transient embedding index, 0 for val and test (no perturbation)
            # if self.split == 'test_train' and idx != 0:
            #     t = idx

            rays_o, rays_d = get_rays(self.directions, c2w)
            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                near = self.bounds.min()
                far = min(8 * near, self.bounds.max())

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_paths[idx]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img
                sample['valid_mask'] = (img[-1]>0).flatten()

        # if self.split == 'all_val':
        #     # set ts=0 for all idx
        #     sample['ts'] = torch.zeros_like(sample['ts']).long()

        return sample



class KubricDataset(Dataset):
    def __init__(self, root_dir, split='train', image_scale=2.):
        """
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        image_scale: scaled to 1/2
        """
        self.root_dir = root_dir
        self.split = split
        self.image_scale = image_scale
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):
        # read metadata json if exists
        # meta_path = Path(self.root_dir / 'metadata.json')
        # if meta_path.exists():
        #     with meta_path.open('r') as f:
        #         metadata = json.load(f)
        # else:
        #     metadata = {}
        
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        self.cam_paths = list(sorted((Path(self.root_dir)/'cameras/').glob('*.json')))
                        # load full resolution image then resize
        if self.split in ['train', 'val', 'all_val']:
            assert len(self.cam_paths) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        with self.cam_paths[0].open('r') as f:
            cam0 = json.load(f)
        W, H = cam0['image_size']
        self.img_wh = [int(W/self.image_scale), int(H/self.image_scale)]
        self.focal = cam0['focal_length'] / self.image_scale
        
        self.bounds = np.zeros((len(self.image_paths), 2))
        self.bounds[:,0] = 1.
        self.bounds[:,1] = 71.4

        poses = np.zeros((len(self.image_paths), 3, 4))
        # load camera poses
        for i, cam_path in enumerate(self.cam_paths):
            with open(cam_path,'r') as f:
                cam_data = json.load(f)
            c2w = np.zeros([4,4])
            c2w[3, 3] = 1
            c2w[:3, :3] = np.array(cam_data['orientation']).T @ np.diag([1,-1,-1])
            c2w[:3, 3] = cam_data['position']

            # c2w = np.array([[ 0., -1.,  0.,  0.],
            #                 [ 0.,  0.,  1.,  0.],
            #                 [-1.,  0.,  0.,  0.],
            #                 [ 0.,  0.,  0.,  1.]]) @ c2w
            # c2w[1,3] = -c2w[1,3] # reverse y axis
            # c2w[2,3] = -c2w[2,3] # reverse z axis
            poses[i,:3,:3] = c2w[:3, :3]
            poses[i,:3,3] = c2w[:3, 3]


            # poses[i,:3,:3] = np.array(cam_data['orientation']).T
            # t = cam_data['position']
            # # poses[i,:3,3] = [t[0], -t[1], t[2]]
            # poses[i,:3,3] = t


        self.poses, self.pose_avg = center_poses(poses)

        # read split json 
        with (Path(self.root_dir) / 'split.json').open('r') as f:
            split_data = json.load(f)
        self.val_ids = split_data['val_ids']
            
        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = self.img_wh[0]/2
        self.K[1, 2] = self.img_wh[1]/2

        np.save(str(Path(self.root_dir) / 'centered_pose.npy'), self.poses)
        np.save(str(Path(self.root_dir) / 'pose.npy'), poses)

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.K) # (H, W, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i in self.val_ids: # exclude the val image
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = i * torch.ones(len(rays_o), 1)
                # img_name = Path(image_path).stem
                # # load time frame from metadata if exists
                # if img_name in metadata:
                #     rays_t = metadata[img_name] * torch.ones(len(rays_o), 1)

                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                self.focal, 1.0, rays_o, rays_d)
                                    # near plane is always at 1.0
                                    # near and far in NDC are always 0 and 1
                                    # See https://github.com/bmild/nerf/issues/34


                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 8)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split == 'val':
            print('val image is', [self.image_paths[val_idx] for val_idx in self.val_ids])
        elif self.split == 'all_val':
            print('val image is', self.image_paths)

        else: 
            raise NotImplementedError('No other split supported for Kubric dataset!')
    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return len(self.val_ids)
        if self.split == 'all_val':
            return len(self.image_paths)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx]}

        else:
            if self.split == 'val':
                idx = self.val_ids[idx]
            elif self.split == 'train_val':
                # render only training views
                all_ids = list(range(len(self.image_paths)))
                train_ids = list(set(all_ids) - set(self.val_ids))
                idx = train_ids[idx]
            c2w = torch.FloatTensor(self.poses[idx])
            # if self.split == 'val':
            # else:
            #     c2w = torch.FloatTensor(self.poses_test[idx])

            if self.split == 'all_val' and idx not in self.val_ids:
                # use original embedding for training input
                t = idx
            else:
                t = 0 

            rays_o, rays_d = get_rays(self.directions, c2w)
            near, far = 0, 1
            rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                            self.focal, 1.0, rays_o, rays_d)

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_paths[idx]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img
                sample['valid_mask'] = (img[-1]>0).flatten()

        return sample



class HyperNeRFDataset(Dataset):
    def __init__(self, root_dir, split='train', image_scale=2.):
        """
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        image_scale: scaled to 1/2
        """
        self.root_dir = root_dir
        self.split = split
        self.image_scale = image_scale
        self.define_transforms()

        self.read_meta()
        self.white_back = False

    def read_meta(self):        
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))
        self.cam_paths = list(sorted((Path(self.root_dir)/'cameras/').glob('*.json')))
                        # load full resolution image then resize
        if self.split in ['train', 'val', 'all_val', 'train_val']:
            assert len(self.cam_paths) == len(self.image_paths), \
                'Mismatch between number of images and number of poses! Please rerun COLMAP!'

        with self.cam_paths[0].open('r') as f:
            cam0 = json.load(f)
        W, H = cam0['image_size']
        self.img_wh = [int(W/self.image_scale), int(H/self.image_scale)]
        self.focal = cam0['focal_length'] / self.image_scale
        
        self.bounds = np.zeros((len(self.image_paths), 2))

        with open(os.path.join(self.root_dir, 'scene.json'),'r') as f:
            scene_data = json.load(f)

        self.bounds[:,0] = scene_data['near'] / scene_data['scale']
        self.bounds[:,1] = scene_data['far'] / scene_data['scale']

        poses = np.zeros((len(self.image_paths), 3, 4))
        # load camera poses
        for i, cam_path in enumerate(self.cam_paths):
            with open(cam_path,'r') as f:
                cam_data = json.load(f)
            c2w = np.zeros([4,4])
            c2w[3, 3] = 1
            c2w[:3, :3] = np.array(cam_data['orientation']).T @ np.diag([1,-1,-1])
            c2w[:3, 3] = cam_data['position']

            poses[i,:3,:3] = c2w[:3, :3]
            poses[i,:3,3] = c2w[:3, 3]

        self.poses, self.pose_avg = center_poses(poses)

        # read split json 
        with (Path(self.root_dir) / 'split.json').open('r') as f:
            split_data = json.load(f)
        self.val_ids = split_data['val_ids']
            
        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter
                                          # the nearest depth is at 1/0.75=1.33
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        self.K = np.eye(3)
        self.K[0, 0] = self.K[1, 1] = self.focal
        self.K[0, 2] = self.img_wh[0]/2
        self.K[1, 2] = self.img_wh[1]/2

        np.save(str(Path(self.root_dir) / 'centered_pose.npy'), self.poses)
        np.save(str(Path(self.root_dir) / 'pose.npy'), poses)

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.K) # (H, W, 3)
            
        if self.split == 'train': # create buffer of all rays and rgb data
                                  # use first N_images-1 to train, the LAST is val
            self.all_rays = []
            self.all_rgbs = []
            for i, image_path in enumerate(self.image_paths):
                if i in self.val_ids: # exclude the val image
                    continue
                c2w = torch.FloatTensor(self.poses[i])

                img = Image.open(image_path).convert('RGB')
                assert img.size[1]*self.img_wh[0] == img.size[0]*self.img_wh[1], \
                    f'''{image_path} has different aspect ratio than img_wh, 
                        please check your data!'''
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                self.all_rgbs += [img]
                
                rays_o, rays_d = get_rays(self.directions, c2w) # both (h*w, 3)
                rays_t = i * torch.ones(len(rays_o), 1)
                # img_name = Path(image_path).stem
                # # load time frame from metadata if exists
                # if img_name in metadata:
                #     rays_t = metadata[img_name] * torch.ones(len(rays_o), 1)

                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                self.focal, 1.0, rays_o, rays_d)
                                    # near plane is always at 1.0
                                    # near and far in NDC are always 0 and 1
                                    # See https://github.com/bmild/nerf/issues/34


                self.all_rays += [torch.cat([rays_o, rays_d, 
                                             near*torch.ones_like(rays_o[:, :1]),
                                             far*torch.ones_like(rays_o[:, :1]),
                                             rays_t],
                                             1)] # (h*w, 8)
                                 
            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
        
        elif self.split == 'val':
            print('val image is', [self.image_paths[val_idx] for val_idx in self.val_ids])
        elif self.split == 'all_val':
            print('val image is', self.image_paths)

        elif self.split == 'train_val':
            print('rendering only training views')

        else: 
            raise NotImplementedError('No other split supported for Kubric dataset!')
    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return len(self.val_ids)
        if self.split == 'all_val':
            return len(self.image_paths)
        if self.split == 'train_val':
            return len(self.image_paths) - len(self.val_ids)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            sample = {'rays': self.all_rays[idx, :8],
                      'ts': self.all_rays[idx, 8].long(),
                      'rgbs': self.all_rgbs[idx]}
            

        else:
            if self.split == 'val':
                idx = self.val_ids[idx]
            c2w = torch.FloatTensor(self.poses[idx])
            # if self.split == 'val':
            # else:
            #     c2w = torch.FloatTensor(self.poses_test[idx])

            if (self.split == 'all_val' or self.split == 'train_val') and idx not in self.val_ids:
                # use original embedding for training input
                t = idx
            else:
                t = 0 

            rays_o, rays_d = get_rays(self.directions, c2w)
            near, far = 0, 1
            rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                            self.focal, 1.0, rays_o, rays_d)

            rays = torch.cat([rays_o, rays_d, 
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)

            sample = {'rays': rays,
                      'ts': t * torch.ones(len(rays), dtype=torch.long),
                      'c2w': c2w}

            if self.split == 'val':
                img = Image.open(self.image_paths[idx]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).permute(1, 0) # (h*w, 3)
                sample['rgbs'] = img
                sample['valid_mask'] = (img[-1]>0).flatten()
        sample['img_wh'] = torch.LongTensor(self.img_wh)

        return sample
