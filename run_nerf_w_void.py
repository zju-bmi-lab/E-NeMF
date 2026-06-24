import os
import sys
import torch
from math import log
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
import time

import imageio
import numpy as np
import random
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from .render_utils import *
from .run_nerf_helpers import *
from tools.load_llff import *
from tools.flow_utils import flow_to_image
from tools.load_event import *
def produce_frame(c2w, mul, time_list=None):
    T = []
    time_list_norm = [i//mul + (time_list[i]-time_list[i//mul*mul])/(time_list[i//mul*mul+mul]-time_list[i//mul*mul]) for i in range(len(time_list)-1)]
    print(time_list_norm)
    key_rots = R.from_matrix(c2w[:, :, :-1])
    key_times = [i*1.0 for i in range(c2w.shape[0])]
    print(key_times)
    slerp = Slerp(key_times, key_rots)
    interp_times = [i*1.0/mul for i in range(c2w.shape[0]*mul-mul)]+[c2w.shape[0]*1.0-1.0]
    if time_list is not None:
        interp_times = time_list_norm+[c2w.shape[0]*1.0-1.0]

    for i in range(c2w.shape[0]-1):
        trans_T = c2w[i+1, :, -1] - c2w[i, :, -1]
        new_T = c2w[i, :,-1]
        T += [new_T]
        for j in range(mul-1):
            diff_T = trans_T * (interp_times[i*mul+j+1]-i)
            print(interp_times[i*mul+j+1]-i)
            new_T2 = new_T + diff_T
            T += [new_T2]
    T += [c2w[-1, :,-1]]
    interp_R = slerp(interp_times)
    R_new = interp_R.as_matrix()
    c2w = []
    for i in range(R_new.shape[0]):
        c2w += [torch.Tensor(np.concatenate([R_new[i], np.expand_dims(T[i],axis=1)], axis=1)).float()]
    c2w = torch.stack(c2w)
    return c2w
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='',
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=300000,
                        help='exponential learning rate decay')
    parser.add_argument("--chunk", type=int, default=1024*128,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*128,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--random_seed", type=int, default=1,
                        help='fix random seed for repeatability')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_viewdirsDyn", action='store_true',
                        help='use full 5D input instead of 3D for D-NeRF')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", type=bool, default=False,
                        help='do not optimize, reload weights and render out render_poses path')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=500,
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=100,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--N_iters", type=int, default=1000000,
                        help='number of training iterations')
    parser.add_argument("--N_iters_pre", type=int, default=201,
                        help='number of training iterations')
    # Dynamic NeRF lambdas
    parser.add_argument("--dynamic_loss_lambda", type=float, default=1.,
                        help='lambda of dynamic loss')
    parser.add_argument("--static_loss_lambda", type=float, default=1.,
                        help='lambda of static loss')
    parser.add_argument("--full_loss_lambda", type=float, default=3.,
                        help='lambda of full loss')
    parser.add_argument("--depth_loss_lambda", type=float, default=0.04,
                        help='lambda of depth loss')
    parser.add_argument("--order_loss_lambda", type=float, default=0.1,
                        help='lambda of order loss')
    parser.add_argument("--flow_loss_lambda", type=float, default=0.02,
                        help='lambda of optical flow loss')
    parser.add_argument("--slow_loss_lambda", type=float, default=0.1,
                        help='lambda of sf slow regularization')
    parser.add_argument("--smooth_loss_lambda", type=float, default=0.1,
                        help='lambda of sf smooth regularization')
    parser.add_argument("--consistency_loss_lambda", type=float, default=0.1,
                        help='lambda of sf cycle consistency regularization')
    parser.add_argument("--mask_loss_lambda", type=float, default=0.1,
                        help='lambda of the mask loss')
    parser.add_argument("--sparse_loss_lambda", type=float, default=0.1,
                        help='lambda of sparse loss')
    parser.add_argument("--DyNeRF_blending", action='store_true',
                        help='use Dynamic NeRF to predict blending weight')
    parser.add_argument("--pretrain", action='store_true',
                        help='Pretrain the StaticneRF')
    parser.add_argument("--ft_path_S", type=str, default=None,
                        help='specific weights npy file to reload for StaticNeRF')

    # For rendering teasers
    parser.add_argument("--frame2dolly", type=int, default=-1,
                        help='choose frame to perform dolly zoom')
    parser.add_argument("--x_trans_multiplier", type=float, default=1.,
                        help='x_trans_multiplier')
    parser.add_argument("--y_trans_multiplier", type=float, default=0.33,
                        help='y_trans_multiplier')
    parser.add_argument("--z_trans_multiplier", type=float, default=5.,
                        help='z_trans_multiplier')
    parser.add_argument("--num_novelviews", type=int, default=60,
                        help='num_novelviews')
    parser.add_argument("--focal_decrease", type=float, default=200,
                        help='focal_decrease')
    #diy option
    parser.add_argument("--eval_event", type=bool, default=False,
                        help='whether to load event window pose')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[856, 957],
                    help="Spatial dimension to crop training samples for training")
    parser.add_argument('--mean_pix', nargs='+', type=float,
                    default=[109.93, 109.167, 101.455],
                    help='mean pixel values carried over from superslomo')
    parser.add_argument("--num_interp", type=int, default=4, 
                        help='the number of interpolation')
    parser.add_argument("--warp_lrate", type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument("--warp_lrate_decay", type=float, default=3000,
                        help='learning rate')
    parser.add_argument("--N_iters_warp", type=int, default=20001,
                        help='number of training iterations')
    parser.add_argument("--event_pose", type=bool, default=True,
                        help='whether to load event window pose')
    parser.add_argument("--interp_pose", type=bool, default=False,
                        help='whether to load event window pose')
    parser.add_argument("--event_lambda", type=float, default=0.5,
                        help='lambda of event flow')
    parser.add_argument("--bezier_lambda", type=float, default=1.0,
                        help='lambda of bezier curve')
    parser.add_argument("--b_lambda", type=float, default=0.02,
                        help='lambda of bezier curve')
    parser.add_argument("--eve_type", type=str, default='log+lin', 
                        help='options: log+lin / log / gamma')
    parser.add_argument("--B", type=float, default=0.2,
                        help='lambda of bezier curve')
    parser.add_argument("--use_B_model", type=bool, default=False,
                        help='learn the threshold B')
    parser.add_argument("--is_color", type=bool, default=False,
                        help='color camera or gray camera')
    return parser

class B_model(nn.Module):
    def __init__(self, init_b=0.3, req_grad=True):
        super(B_model, self).__init__()
        self.b_pos = nn.Parameter(torch.tensor(init_b, dtype=torch.float32), requires_grad=req_grad)
        self.b_neg = nn.Parameter(torch.tensor(init_b, dtype=torch.float32), requires_grad=req_grad)
    def forward(self, cam_id):
        b_pos, b_neg = self.b_pos, self.b_neg
        return b_pos, b_neg

def train():
    parser = config_parser()
    args, remaining_args = parser.parse_known_args()
    print(remaining_args)

    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
    print(args.event_pose)
    args.event_pose = False
    args.interp_pose = True

    # Load data
    if args.dataset_type == 'llff':
        frame2dolly = args.frame2dolly
        images, invdepths, masks, poses, bds, \
        render_poses, render_focals, grids = load_llff_data(args, args.datadir,
                                                            args.factor,
                                                            frame2dolly=frame2dolly,
                                                            recenter=True, bd_factor=.9,
                                                            spherify=args.spherify,
                                                            event_pose=args.event_pose)
        
        hwf = poses[0, :3, -1]
        events, time_list = load_event_timereplayer(args.datadir, int(hwf[0]),int(hwf[1]), args.num_interp, int(images.shape[0]))
        print(events.shape)

        if args.event_pose:
            e_bds, e_poses = bds, poses
            bds, poses = bds[::args.num_interp, ...], poses[::args.num_interp, ...]
            e_poses = e_poses[:, :3, :4]
            e_poses = torch.Tensor(e_poses)
        elif args.interp_pose:
            e_poses = produce_frame(poses[:, :3, :4], 4, time_list)
            e_poses = torch.Tensor(e_poses)
            print(e_poses.shape)

        poses = poses[:, :3, :4]
        num_img = float(poses.shape[0])
        assert len(poses) == len(images)
        print('Loaded llff', images.shape,
            render_poses.shape, hwf, args.datadir)

        # Use all views to train
        i_train = np.array([i for i in np.arange(int(images.shape[0]))])
        i_noveltime = np.array([i for i in np.arange(4 * int(images.shape[0]))])
        print('DEFINING BOUNDS')
        if args.no_ndc:
            raise NotImplementedError
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)

    if not args.render_only:
        f = os.path.join(basedir, expname, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        if args.config is not None:
            f = os.path.join(basedir, expname, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = 0

    bds_dict = {
        'near': near,
        'far': far,
        'num_img': num_img,
        'num_interp': args.num_interp,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    N_rand = args.N_rand

    # Move training data to GPU
    images = torch.Tensor(images)
    invdepths = torch.Tensor(invdepths)
    masks = 1.0 - torch.Tensor(masks)
    poses = torch.Tensor(poses)
    grids = torch.Tensor(grids)
    events = torch.Tensor(events)

    print('Begin')
    print('TRAIN views are', i_train)

    decay_iteration = max(25, num_img)

    # Pre-train StaticNeRF
    if args.pretrain:
        render_kwargs_train.update({'pretrain': True})

        # Pre-train StaticNeRF first and use DynamicNeRF to blend
        assert args.DyNeRF_blending == True

        if args.ft_path_S is not None and args.ft_path_S != 'None':
            # Load Pre-trained StaticNeRF
            ckpt_path = args.ft_path_S
            print('Reloading StaticNeRF from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            render_kwargs_train['network_fn_s'].load_state_dict(ckpt['network_fn_s_state_dict'])
        else:
            ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname)))\
                      if 'tar' in f and 'Pretrained' in f]
            skip_pre = False
            if len(ckpts) > 0:
                skip_pre = True
                ckpt_path = ckpts[-1]
                print('Reloading StaticNeRF from', ckpt_path)
                ckpt = torch.load(ckpt_path)
                render_kwargs_train['network_fn_s'].load_state_dict(ckpt['network_fn_s_state_dict'])
                global_step = ckpt['global_step'] + 1
                if global_step < args.N_iters_pre:
                    skip_pre = False
                    pass
            if not skip_pre:
                # Train StaticNeRF from scratch
                for i in range(global_step, args.N_iters_pre):
                    time0 = time.time()

                    # No raybatching as we need to take random rays from one image at a time
                    img_i = np.random.choice(i_train)
                    t = img_i / num_img * 2. - 1.0 # time of the current frame
                    target = images[img_i]
                    pose = poses[img_i, :3, :4]
                    mask = masks[img_i] # Static region mask

                    rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose)) # (H, W, 3), (H, W, 3)
                    coords_s = torch.stack((torch.where(mask >= 0.5)), -1)
                    select_inds_s = np.random.choice(coords_s.shape[0], size=[N_rand], replace=False)
                    select_coords = coords_s[select_inds_s]

                    def select_batch(value, select_coords=select_coords):
                        return value[select_coords[:, 0], select_coords[:, 1]]

                    rays_o = select_batch(rays_o) # (N_rand, 3)
                    rays_d = select_batch(rays_d) # (N_rand, 3)
                    target_rgb = select_batch(target)
                    batch_mask = select_batch(mask[..., None])
                    batch_rays = torch.stack([rays_o, rays_d], 0)

                    #####  Core optimization loop  #####
                    ret = render(t,
                                False,
                                H, W, focal,
                                chunk=args.chunk,
                                rays=batch_rays,
                                **render_kwargs_train)

                    optimizer.zero_grad()

                    # Compute MSE loss between rgb_s and true RGB.
                    img_s_loss = img2mse(ret['rgb_map_s'], target_rgb)
                    psnr_s = mse2psnr(img_s_loss)
                    loss = args.static_loss_lambda * img_s_loss

                    loss.backward()
                    optimizer.step()

                    # Learning rate decay.
                    decay_rate = 0.1
                    decay_steps = args.lrate_decay
                    new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lrate

                    dt = time.time() - time0

                    if global_step%500==0:
                        print(f"Pretraining step: {global_step}, Loss: {loss}, Time: {dt}, expname: {expname}")
                    global_step += 1

                # Save the pretrained weight
                torch.save({
                    'global_step': global_step,
                    'network_fn_s_state_dict': render_kwargs_train['network_fn_s'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(basedir, expname, 'Pretrained_S.tar'))

    # Reset
    render_kwargs_train.update({'pretrain': False})
    global_step = start

    # Fix the StaticNeRF and only train the DynamicNeRF
    grad_vars_d = list(render_kwargs_train['network_fn_d'].parameters())
    if args.use_B_model:
        b_model = B_model(init_b = args.B)
        grad_vars_d += list(b_model.parameters())
    optimizer = torch.optim.Adam(params=grad_vars_d, lr=args.lrate, betas=(0.9, 0.999))
    torch.cuda.empty_cache()

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #torch.cuda.set_device(0)
    global_step = start
    N_depthguide = 1
    num_interp = args.num_interp

    for i in range(start, args.N_iters):
        """torch.cuda.empty_cache()
        torch.cuda.set_device(1) """
        time0 = time.time()

        # Use frames at t-2, t-1, t, t+1, t+2 (adapted from NSFF)
        if i < decay_iteration * 2000:
            chain_5frames = False
        else:
            chain_5frames = True

        # Lambda decay.
        Temp = 1. / (10 ** (i // (decay_iteration * 1000)))

        # No raybatching as we need to take random rays from one image at a time
        img_i = np.random.choice(i_train)
        t = img_i / num_img * 2. - 1.0 # time of the current frame
        target = images[img_i]
        pose = poses[img_i, :3, :4]
        mask = masks[img_i] # Static region mask
        invdepth = invdepths[img_i]
        grid = grids[img_i]

        rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose)) # (H, W, 3), (H, W, 3)
        coords_d = torch.stack((torch.where(mask < 0.5)), -1)
        coords_s = torch.stack((torch.where(mask >= 0.5)), -1)
        coords = torch.stack((torch.where(mask > -1)), -1)

        # Evenly sample dynamic region and static region
        """ if i%10000==0 and i!=0 and N_depthguide<=64:
            N_depthguide *= 2
            N_rand *= 2 """
        select_inds_d = np.random.choice(coords_d.shape[0], size=[min(len(coords_d), N_rand//2)], replace=False)
        select_inds_s = np.random.choice(coords_s.shape[0], size=[N_rand//2], replace=False)
        select_coords = torch.cat([coords_s[select_inds_s],
                                   coords_d[select_inds_d]], 0)

        def select_batch(value, select_coords=select_coords):
            return value[select_coords[:, 0], select_coords[:, 1]]

        rays_o = select_batch(rays_o) # (N_rand, 3)
        rays_d = select_batch(rays_d) # (N_rand, 3)
        target_rgb = select_batch(target)
        batch_grid = select_batch(grid) # (N_rand, 8)
        batch_mask = select_batch(mask[..., None])
        batch_invdepth = select_batch(invdepth)
        batch_rays = torch.stack([rays_o, rays_d], 0)

        if args.is_color:
            color_mask = torch.zeros_like(target)
            color_mask[0::2, 0::2, 0] = 1
            color_mask[1::2, 0::2, 1] = 1
            color_mask[0::2, 1::2, 1] = 1
            color_mask[1::2, 1::2, 2] = 1
            batch_color_mask = select_batch(color_mask)

        #####  Core optimization loop  #####
        rand = random.randint(0,num_interp-2)
        bi_time = {'rand_f':rand ,'rand_b':num_interp-2-rand}
        if args.use_B_model:
            B_pos, B_neg = b_model(0)
            if i<=20000:
                B_pos, B_neg = B_pos.detach(), B_neg.detach()
        else:
            B_pos, B_neg = args.B, args.B
        para_dict = {
            'bi_time': bi_time,'N_depthguide':N_depthguide,'e_render':False,
        }
        def rgb2g(frame):
            if args.is_color == True:
                return (frame * batch_color_mask).sum(-1)
            return frame[:,0]*0.3+frame[:,1]*0.59+frame[:,2]*0.11

        render_kwargs_train.update(para_dict)
        render_kwargs_test.update(para_dict)
        ret = render(t,
                     chain_5frames,
                     H, W, focal,
                     chunk=args.chunk,
                     rays=batch_rays,
                     **render_kwargs_train)

        delta_t = 1. / num_img * 2.
        frame_now = target_rgb
        gray_now = rgb2g(frame_now)

        # ----------------compute event image--------------------------------------------------
        inv_bi_time = {'rand_f':num_interp-2-rand,'rand_b': rand}
        render_kwargs_train.update({'e_render': True, 'bi_time': inv_bi_time})

        def img2event(img):
            if args.eve_type == 'log+lin':
                linmask = img >= 20./255.
                logval = torch.log((img+1e-8) * 255.)
                linval = log(20.)*img*255./20.
                eve = logval*linmask+(~linmask)*linval
            elif args.eve_type == 'log':
                eve = torch.log(img+1e-8)
            elif args.eve_type == 'gamma':
                eve = torch.log(img**2.2+1e-8)
            return eve

        if t>-1.:
            e_pose_b = e_poses[img_i*num_interp-num_interp+1+rand, :3, :4]
            rays_o_b, rays_d_b = get_rays(H, W, focal, torch.Tensor(e_pose_b))
            rays_o_b = select_batch(rays_o_b) 
            rays_d_b = select_batch(rays_d_b) 
            batch_rays_b = torch.stack([rays_o_b, rays_d_b], 0)
            ret_b = render(t - (bi_time['rand_b']+1)/num_interp*delta_t,
                           chain_5frames, H, W, focal,
                            chunk=args.chunk,
                            rays=batch_rays_b,
                            **render_kwargs_train)
            frame_half_b = ret_b['rgb_map_d']
            frame_hb_hf = ret_b['rgb_map_d_fmid']
            frame_hb_hb = ret_b['rgb_map_d_bmid']
            gray_b = rgb2g(frame_half_b)
            gray_bf = rgb2g(frame_hb_hf)
            gray_bb = rgb2g(frame_hb_hb)
            event_b = (img2event(gray_now) - img2event(gray_b))
            event_bf = (img2event(gray_now) - img2event(gray_bf))
            event_bb = (img2event(gray_now) - img2event(gray_bb))
        
        if t<((num_img - 1) / num_img * 2. - 1.0):
            e_pose_f = e_poses[img_i*num_interp+rand+1, :3, :4]
            rays_o_f, rays_d_f = get_rays(H, W, focal, torch.Tensor(e_pose_f))
            rays_o_f = select_batch(rays_o_f) 
            rays_d_f = select_batch(rays_d_f) 
            batch_rays_f = torch.stack([rays_o_f, rays_d_f], 0)
            ret_f = render(t + (bi_time['rand_f']+1)/num_interp*delta_t,
                            chain_5frames,H, W, focal,
                            chunk=args.chunk,
                            rays=batch_rays_f,
                            **render_kwargs_train)
            frame_half_f = ret_f['rgb_map_d']
            frame_hf_hb = ret_f['rgb_map_d_bmid']
            frame_hf_hf = ret_f['rgb_map_d_fmid']
            gray_f = rgb2g(frame_half_f)
            gray_fb = rgb2g(frame_hf_hb)
            gray_ff = rgb2g(frame_hf_hf)
            event_f = (img2event(gray_f) - img2event(gray_now))
            event_fb = (img2event(gray_fb) - img2event(gray_now))
            event_ff = (img2event(gray_ff) - img2event(gray_now))
        # ----------------end event part image--------------------------------------------------

        render_kwargs_train.update({'e_render': False, 'bi_time': bi_time})
        optimizer.zero_grad()
        loss = 0
        loss_dict = {}

        # ----------------compute event to image part--------------------------------------------------
        # Compute event loss in T - 0.5t and T + 0.5t
        # |-------*-----|-----------*----|
        # 0       1     2           3

        if t>-1.:                                 
            ev02_pos = events[img_i-1][-1][0]
            ev02_neg = events[img_i-1][-1][1]
            ev01_pos = events[img_i-1][bi_time['rand_f']][0]
            ev01_neg = events[img_i-1][bi_time['rand_f']][1]
            target_ev12_pos = ev02_pos - ev01_pos
            target_ev12_neg = ev02_neg - ev01_neg
            target_ev12_pos = select_batch(target_ev12_pos[..., None])
            target_ev12_neg = select_batch(target_ev12_neg[..., None])
            M = ((target_ev12_neg==0)*(target_ev12_pos==0))*1.0
            M = torch.bernoulli(M * 0.05) + (1 - M) #void sample
            M = M.detach()

            e_loss_b = event2loss(event_b[..., None], target_ev12_pos, target_ev12_neg, B_pos, B_neg, M)
            e_loss_b += event2loss(event_bf[..., None], target_ev12_pos, target_ev12_neg, B_pos, B_neg, M)
            e_loss_b += event2loss(event_bb[..., None], target_ev12_pos, target_ev12_neg, B_pos, B_neg, M)
            loss_dict['event_loss_b'] = e_loss_b

        if t<((num_img - 1) / num_img * 2. - 1.0): 
            target_ev23_pos = events[img_i][bi_time['rand_f']][0]
            target_ev23_neg = events[img_i][bi_time['rand_f']][1]
            target_ev23_pos = select_batch(target_ev23_pos[..., None])
            target_ev23_neg = select_batch(target_ev23_neg[..., None])
            M = ((target_ev23_neg==0)*(target_ev23_pos==0))*1.0
            M = torch.bernoulli(M * 0.05) + (1 - M)
            M = M.detach()

            e_loss_f = event2loss(event_f[..., None], target_ev23_pos, target_ev23_neg, B_pos, B_neg, M)
            e_loss_f += event2loss(event_fb[..., None], target_ev23_pos, target_ev23_neg, B_pos, B_neg, M)
            e_loss_f += event2loss(event_ff[..., None], target_ev23_pos, target_ev23_neg, B_pos, B_neg, M)
            loss_dict['event_loss_f'] = e_loss_f

        if 'event_loss_f' in loss_dict:
            loss += args.event_lambda * loss_dict['event_loss_f']
        if 'event_loss_b' in loss_dict:
            loss += args.event_lambda * loss_dict['event_loss_b']
        # -----------------end-------------------------------------------------------------------------

        # Compute MSE loss between rgb_full and true RGB.
        img_loss = img2mse(ret['rgb_map_full'], target_rgb)
        psnr = mse2psnr(img_loss)
        loss_dict['psnr'] = psnr
        loss_dict['img_loss'] = img_loss
        loss += args.full_loss_lambda * loss_dict['img_loss']

        # Compute MSE loss between rgb_s and true RGB.
        img_s_loss = img2mse(ret['rgb_map_s'], target_rgb, batch_mask)
        psnr_s = mse2psnr(img_s_loss)
        loss_dict['psnr_s'] = psnr_s
        loss_dict['img_s_loss'] = img_s_loss
        loss += args.static_loss_lambda * loss_dict['img_s_loss']

        # Compute MSE loss between rgb_d and true RGB.
        img_d_loss = img2mse(ret['rgb_map_d'], target_rgb)
        psnr_d = mse2psnr(img_d_loss)
        loss_dict['psnr_d'] = psnr_d
        loss_dict['img_d_loss'] = img_d_loss
        loss += args.dynamic_loss_lambda * loss_dict['img_d_loss']

        # Compute MSE loss between rgb_d_f and true RGB.
        img_d_f_loss = img2mse(ret['rgb_map_d_f'], target_rgb)
        psnr_d_f = mse2psnr(img_d_f_loss)
        loss_dict['psnr_d_f'] = psnr_d_f
        loss_dict['img_d_f_loss'] = img_d_f_loss
        loss += args.dynamic_loss_lambda * loss_dict['img_d_f_loss']

        # Compute MSE loss between rgb_d_b and true RGB.
        img_d_b_loss = img2mse(ret['rgb_map_d_b'], target_rgb)
        psnr_d_b = mse2psnr(img_d_b_loss)
        loss_dict['psnr_d_b'] = psnr_d_b
        loss_dict['img_d_b_loss'] = img_d_b_loss
        loss += args.dynamic_loss_lambda * loss_dict['img_d_b_loss']

        # ----------------compute image to event part------------------------------------------------
        # Compute MSE loss between rgb_d_f and true RGB.
        img_d_f_loss = img2mse(ret['rgb_map_d_fmid'], target_rgb)
        psnr_d_f = mse2psnr(img_d_f_loss)
        loss_dict['psnr_d_fmid'] = psnr_d_f
        loss_dict['img_d_fmid_loss'] = img_d_f_loss
        loss += args.dynamic_loss_lambda * loss_dict['img_d_f_loss']

        # Compute MSE loss between rgb_d_b and true RGB.
        img_d_b_loss = img2mse(ret['rgb_map_d_bmid'], target_rgb)
        psnr_d_b = mse2psnr(img_d_b_loss)
        loss_dict['psnr_d_bmid'] = psnr_d_b
        loss_dict['img_d_bmid_loss'] = img_d_b_loss
        loss += args.dynamic_loss_lambda * loss_dict['img_d_b_loss']
        # ----------------------------------end-----------------------------------------------------

        # Motion loss.
        # Compuate EPE between induced flow and true flow (forward flow).
        # The last frame does not have forward flow.
        if img_i < num_img - 1:
            pts_f = ret['raw_pts_f']
            weight = ret['weights_d']
            pose_f = poses[img_i + 1, :3, :4]
            induced_flow_f = induce_flow(H, W, focal, pose_f, weight, pts_f, batch_grid[..., :2])
            flow_f_loss = img2mae(induced_flow_f, batch_grid[:, 2:4], batch_grid[:, 4:5])
            loss_dict['flow_f_loss'] = flow_f_loss
            loss += args.flow_loss_lambda * Temp * loss_dict['flow_f_loss']

        # Compuate EPE between induced flow and true flow (backward flow).
        # The first frame does not have backward flow.
        if img_i > 0:
            pts_b = ret['raw_pts_b']
            weight = ret['weights_d']
            pose_b = poses[img_i - 1, :3, :4]
            induced_flow_b = induce_flow(H, W, focal, pose_b, weight, pts_b, batch_grid[..., :2])
            flow_b_loss = img2mae(induced_flow_b, batch_grid[:, 5:7], batch_grid[:, 7:8])
            loss_dict['flow_b_loss'] = flow_b_loss
            loss += args.flow_loss_lambda * Temp * loss_dict['flow_b_loss']

        # ------------------------the loss about event---------------------------------------
        # First, the Bezier curve consistency loss
        bezier_loss = L1(ret['sceneflow_fmid'] + ret['sf_hf_hb'])\
                    + L1(ret['sceneflow_bmid'] + ret['sf_hb_hf'])\
                    + L1(ret['sceneflow_fmid'] + ret['sf_hf_hf'] - ret['sceneflow_f'])\
                    + L1(ret['sceneflow_bmid'] + ret['sf_hb_hb'] - ret['sceneflow_b'])
        loss_dict['bezier_loss'] = bezier_loss
        loss += args.bezier_lambda * loss_dict['bezier_loss']
        #Second, B loss
        if args.use_B_model:
            b_loss = torch.abs(B_pos-args.B)+torch.abs(B_neg-args.B)+torch.abs(B_pos-B_neg)
            loss_dict['b_loss'] = b_loss
            loss += args.b_lambda * b_loss
        #-------------------------the end----------------------------------------------------
        # Slow scene flow. The forward and backward sceneflow should be small.
        slow_loss = L1(ret['sceneflow_b']) + L1(ret['sceneflow_f'])
        loss_dict['slow_loss'] = slow_loss
        loss += args.slow_loss_lambda * loss_dict['slow_loss']

        # Smooth scene flow. The summation of the forward and backward sceneflow should be small.
        smooth_loss = compute_sf_smooth_loss(ret['raw_pts'],
                                             ret['raw_pts_f'],
                                             ret['raw_pts_b'],
                                             H, W, focal)
        loss_dict['smooth_loss'] = smooth_loss
        loss += args.smooth_loss_lambda * loss_dict['smooth_loss']

        # Spatial smooth scene flow. (loss adapted from NSFF)
        sp_smooth_loss = compute_sf_smooth_s_loss(ret['raw_pts'], ret['raw_pts_f'], H, W, focal) \
                       + compute_sf_smooth_s_loss(ret['raw_pts'], ret['raw_pts_b'], H, W, focal)
        loss_dict['sp_smooth_loss'] = sp_smooth_loss
        loss += args.smooth_loss_lambda * loss_dict['sp_smooth_loss']

        # Consistency loss.
        consistency_loss = L1(ret['sceneflow_f'] + ret['sceneflow_f_b']) + \
                           L1(ret['sceneflow_b'] + ret['sceneflow_b_f'])
        loss_dict['consistency_loss'] = consistency_loss
        loss += args.consistency_loss_lambda * loss_dict['consistency_loss']

        # Mask loss.
        mask_loss = L1(ret['blending'][batch_mask[:, 0].type(torch.bool)]) + \
                    img2mae(ret['dynamicness_map'][..., None], 1 - batch_mask) + \
                    img2mae(ret['blending'], ret['blending_fmid']) + \
                    img2mae(ret['blending'], ret['blending_bmid'])#mask new time 
        loss_dict['mask_loss'] = mask_loss
        if i < decay_iteration * 1000:
            loss += args.mask_loss_lambda * loss_dict['mask_loss']

        # Sparsity loss.
        sparse_loss = entropy(ret['weights_d']) + entropy(ret['blending'])
        loss_dict['sparse_loss'] = sparse_loss
        loss += args.sparse_loss_lambda * loss_dict['sparse_loss']

        # Depth constraint
        # Depth in NDC space equals to negative disparity in Euclidean space.
        depth_loss = compute_depth_loss(ret['depth_map_d'], -batch_invdepth)
        loss_dict['depth_loss'] = depth_loss
        loss += args.depth_loss_lambda * Temp * loss_dict['depth_loss']

        # Order loss
        order_loss = torch.mean(torch.square(ret['depth_map_d'][batch_mask[:, 0].type(torch.bool)] - \
                                             ret['depth_map_s'].detach()[batch_mask[:, 0].type(torch.bool)]))
        loss_dict['order_loss'] = order_loss
        loss += args.order_loss_lambda * loss_dict['order_loss']

        sf_smooth_loss = compute_sf_smooth_loss(ret['raw_pts_b'],
                                                ret['raw_pts'],
                                                ret['raw_pts_b_b'],
                                                H, W, focal) + \
                         compute_sf_smooth_loss(ret['raw_pts_f'],
                                                ret['raw_pts_f_f'],
                                                ret['raw_pts'],
                                                H, W, focal)
        loss_dict['sf_smooth_loss'] = sf_smooth_loss
        loss += args.smooth_loss_lambda * loss_dict['sf_smooth_loss']

        if chain_5frames:
            img_d_b_b_loss = img2mse(ret['rgb_map_d_b_b'], target_rgb)
            loss_dict['img_d_b_b_loss'] = img_d_b_b_loss
            loss += args.dynamic_loss_lambda * loss_dict['img_d_b_b_loss']

            img_d_f_f_loss = img2mse(ret['rgb_map_d_f_f'], target_rgb)
            loss_dict['img_d_f_f_loss'] = img_d_f_f_loss
            loss += args.dynamic_loss_lambda * loss_dict['img_d_f_f_loss']

            
        loss.backward()
        optimizer.step()

        # Learning rate decay.
        decay_rate = 0.1
        decay_steps = args.lrate_decay
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time() - time0
        if global_step % 100==0:
            print(f"Step: {global_step}, Loss: {loss}, Time: {dt}, PSNR: {psnr.detach().cpu().numpy()[0]}, b_pos: {B_pos}, b_neg: {B_neg}")
        
        # Rest is logging
        if i % args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))

            if args.N_importance > 0:
                raise NotImplementedError
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_d_state_dict': render_kwargs_train['network_fn_d'].state_dict(),
                    'network_fn_s_state_dict': render_kwargs_train['network_fn_s'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)

            print('Saved weights at', path)

        if i % args.i_img == 0:
            # Log a rendered training view to Tensorboard.
            # img_i = np.random.choice(i_train[1:-1])
            target = images[img_i]
            pose = poses[img_i, :3, :4]
            mask = masks[img_i]
            grid = grids[img_i]
            invdepth = invdepths[img_i]

            with torch.no_grad():
                choice = random.randint(0,1)
                testdir = os.path.join(basedir, expname, 'tboard_val_imgs')
                testimgdir = os.path.join(testdir, 'rgb')
                testdepthdir = os.path.join(testdir, 'depth')
                testflow_f_dir = os.path.join(testdir, 'flow_f')
                testflow_b_dir = os.path.join(testdir, 'flow_b')
                if i == 0:
                    os.makedirs(testimgdir, exist_ok=True)
                    os.makedirs(testdepthdir, exist_ok=True)
                    os.makedirs(testflow_f_dir, exist_ok=True)
                    os.makedirs(testflow_b_dir, exist_ok=True)
                if choice == 0:
                    ret = render(t,
                                False,
                                H, W, focal,
                                chunk=1024*16,
                                c2w=pose,
                                **render_kwargs_test)
                    if img_i > 0:
                        pose_bmid = e_pose_b
                        pose_b = poses[img_i - 1, :3, :4]
                        inv_pose_bmid = pose_bmid[:3, :3].transpose(0, 1) # same as np.linalg.inv(c2w[:3, :3])
                        inv_pose_b = pose_b[:3, :3].transpose(0, 1) # same as np.linalg.inv(c2w[:3, :3])

                        # Rendered 3D position in NDC coordinate and NDC coordinate to world coordinate
                        pts_b_map_NDC, pts_bmid_map_NDC = ret['pts_b_map_NDC'], ret['pts_bmid_map_NDC']
                        pts_b_map_world, pts_bmid_map_world = NDC2world(pts_b_map_NDC, H, W, focal), NDC2world(pts_bmid_map_NDC, H, W, focal)

                        # World coordinate to camera coordinate
                        # Translate
                        pts_b_map_world,  pts_bmid_map_world = pts_b_map_world - pose_b[:, 3], pts_bmid_map_world - pose_bmid[:, 3]
                        # Rotate
                        pts_b_map_cam = torch.sum(pts_b_map_world[..., None, :] * inv_pose_b[:3, :3], -1)
                        pts_bmid_map_cam = torch.sum(pts_bmid_map_world[..., None, :] * inv_pose_bmid[:3, :3], -1)

                        # Camera coordinate to 2D image coordinate
                        pts_b_plane = torch.cat([pts_b_map_cam[..., 0:1] / (- pts_b_map_cam[..., 2:]) * focal + W * .5,
                                            - pts_b_map_cam[..., 1:2] / (- pts_b_map_cam[..., 2:]) * focal + H * .5],
                                            -1)
                        pts_bmid_plane = torch.cat([pts_bmid_map_cam[..., 0:1] / (- pts_bmid_map_cam[..., 2:]) * focal + W * .5,
                                            - pts_bmid_map_cam[..., 1:2] / (- pts_bmid_map_cam[..., 2:]) * focal + H * .5],
                                            -1)
                        pred_flow_b = pts_b_plane - grid[..., :2]
                        pred_flow_bmid = pts_bmid_plane - grid[..., :2]
                    if img_i < num_img - 1:
                        pose_fmid = e_pose_f
                        pose_f = poses[img_i + 1, :3, :4]
                        inv_pose_fmid = pose_fmid[:3, :3].transpose(0, 1) # same as np.linalg.inv(c2w[:3, :3])
                        inv_pose_f = pose_f[:3, :3].transpose(0, 1) # same as np.linalg.inv(c2w[:3, :3])

                        # Rendered 3D position in NDC coordinate and NDC coordinate to world coordinate
                        pts_f_map_NDC, pts_fmid_map_NDC = ret['pts_f_map_NDC'], ret['pts_fmid_map_NDC']
                        pts_f_map_world, pts_fmid_map_world = NDC2world(pts_f_map_NDC, H, W, focal), NDC2world(pts_fmid_map_NDC, H, W, focal)

                        # World coordinate to camera coordinate
                        # Translate
                        pts_f_map_world, pts_fmid_map_world = pts_f_map_world - pose_f[:, 3], pts_fmid_map_world - pose_fmid[:, 3]
                        # Rotate
                        pts_f_map_cam = torch.sum(pts_f_map_world[..., None, :] * inv_pose_f[:3, :3], -1)
                        pts_fmid_map_cam = torch.sum(pts_fmid_map_world[..., None, :] * inv_pose_fmid[:3, :3], -1)

                        # Camera coordinate to 2D image coordinate
                        pts_f_plane = torch.cat([pts_f_map_cam[..., 0:1] / (- pts_f_map_cam[..., 2:]) * focal + W * .5,
                                            - pts_f_map_cam[..., 1:2] / (- pts_f_map_cam[..., 2:]) * focal + H * .5],
                                            -1)
                        pts_fmid_plane = torch.cat([pts_fmid_map_cam[..., 0:1] / (- pts_fmid_map_cam[..., 2:]) * focal + W * .5,
                                            - pts_fmid_map_cam[..., 1:2] / (- pts_fmid_map_cam[..., 2:]) * focal + H * .5],
                                            -1)
                        pred_flow_f = pts_f_plane - grid[..., :2]
                        pred_flow_fmid = pts_fmid_plane - grid[..., :2]
                    imageio.imwrite(os.path.join(testimgdir, 'RGB_FULL{:06d}.png'.format(i)), to8b(ret['rgb_map_full'].cpu().numpy()))
                    imageio.imwrite(os.path.join(testimgdir, 'RGB_DY{:06d}.png'.format(i)), to8b(ret['rgb_map_d'].cpu().numpy()))
                    imageio.imwrite(os.path.join(testimgdir, 'RGB_S{:06d}.png'.format(i)), to8b(ret['rgb_map_s'].cpu().numpy()))

                    imageio.imwrite(os.path.join(testdepthdir, 'DEPTH_FULL{:06d}.png'.format(i)), to8b(ret['depth_map_full'].cpu().numpy()))
                    imageio.imwrite(os.path.join(testdepthdir, 'DEPTH_D{:06d}.png'.format(i)), to8b(ret['depth_map_d'].cpu().numpy()))
                    if img_i < num_img - 1:
                        imageio.imwrite(os.path.join(testflow_f_dir, 'FLOW_F{:06d}.png'.format(i)), flow_to_image(pred_flow_f.cpu().numpy()))
                        imageio.imwrite(os.path.join(testflow_f_dir, 'FLOW_FMID{:06d}.png'.format(i)), flow_to_image(pred_flow_fmid.cpu().numpy()))
                    if img_i > 0:
                        imageio.imwrite(os.path.join(testflow_b_dir, 'FLOW_B{:06d}.png'.format(i)), flow_to_image(pred_flow_b.cpu().numpy()))
                        imageio.imwrite(os.path.join(testflow_b_dir, 'FLOW_BMID{:06d}.png'.format(i)), flow_to_image(pred_flow_bmid.cpu().numpy()))
                    if i == 80000:
                        imageio.imwrite(os.path.join(testdepthdir, 'DEPTH_S{:06d}.png'.format(i)), to8b(ret['depth_map_s'].cpu().numpy()))
                        imageio.imwrite(os.path.join(testdepthdir, 'RGB_S{:06d}.png'.format(i)), to8b(ret['rgb_map_s'].cpu().numpy()))
                        
                else:
                    ret = render(t + (bi_time['rand_f']+1)/num_interp*delta_t,
                            False,H, W, focal,chunk=1024*16,
                            c2w=e_pose_f,
                            **render_kwargs_test)
                    imageio.imwrite(os.path.join(testimgdir, 'RGB_F{:06d}.png'.format(i)), to8b(ret['rgb_map_d'].cpu().numpy()))
                    imageio.imwrite(os.path.join(testimgdir, 'RGB_F_FULL{:06d}.png'.format(i)), to8b(ret['rgb_map_full'].cpu().numpy()))
                    imageio.imwrite(os.path.join(testdepthdir, 'DEPTH_FULL{:06d}.png'.format(i)), to8b(ret['depth_map_full'].cpu().numpy()))
                    imageio.imwrite(os.path.join(testdepthdir, 'DEPTH_D{:06d}.png'.format(i)), to8b(ret['depth_map_d'].cpu().numpy()))
                    ret = render(t - (bi_time['rand_b']+1)/num_interp*delta_t,
                            False, H, W, focal,chunk=1024*16,
                            c2w=e_pose_b,
                            **render_kwargs_test)
                    imageio.imwrite(os.path.join(testimgdir, 'RGB_B{:06d}.png'.format(i)), to8b(ret['rgb_map_d'].cpu().numpy()))
                    imageio.imwrite(os.path.join(testimgdir, 'RGB_B_FULL{:06d}.png'.format(i)), to8b(ret['rgb_map_full'].cpu().numpy()))
                    imageio.imwrite(os.path.join(testdepthdir, 'DEPTH_FULL{:06d}.png'.format(i)), to8b(ret['depth_map_full'].cpu().numpy()))
                    imageio.imwrite(os.path.join(testdepthdir, 'DEPTH_D{:06d}.png'.format(i)), to8b(ret['depth_map_d'].cpu().numpy()))

        global_step += 1


if __name__ == '__main__':
    torch.cuda.set_device(1)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train()
