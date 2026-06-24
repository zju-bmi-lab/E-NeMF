import os
import time
import torch
import imageio
import numpy as np
import torch.nn.functional as F
from math import comb

from .run_nerf_helpers import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batchify_rays(t, chain_5frames,
                rays_flat, chunk=1024*16, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(t, chain_5frames, rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(t, chain_5frames,
           H, W, focal, focal_render=None,
           chunk=1024*16, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    if c2w is not None:
        # special case to render full image
        if focal_render is not None:
            # Render full image using different focal length for dolly zoom. Inference only.
            rays_o, rays_d = get_rays(H, W, focal_render, c2w)
        else:
            rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            raise NotImplementedError
        # Make all directions unit magnitude.
        # shape: [batch_size, 3]
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()
    near, far = near * \
        torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])

    # (ray origin, ray direction, min dist, max dist) for each ray
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(t, chain_5frames,
                        rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    return all_ret


def render_path_batch(render_poses, time2render,
                    hwf, chunk, render_kwargs, savedir=None, focal2render=None):
    """Render frames using batch.

    Args:
      render_poses: array of shape [num_frame, 3, 4]. Camera-to-world transformation matrix of each frame.
      time2render: array of shape [num_frame]. Time of each frame.
      hwf: list. [Height of image in pixels, Width of image in pixels, Focal length of pinhole camera]
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      render_kwargs: dictionary. args for the render function.
      savedir: string. Directory to save results.
      focal2render: list. Only used to perform dolly-zoom.
    Returns:
      ret_dict: dictionary. Final and intermediate results.
    """
    H, W, focal = hwf

    ret_dict = {}
    rgbs = []
    rgbs_d = []
    rgbs_s = []
    dynamicnesses = []

    time_curr = time.time()
    for i, c2w in enumerate(render_poses):

        print(i, time.time() - time_curr)
        time_curr = time.time()

        t = time2render[i]

        if focal2render is not None:
            # Render full image using different focal length
            rays_o, rays_d = get_rays(H, W, focal2render[i], c2w)
        else:
            rays_o, rays_d = get_rays(H, W, focal, c2w)
        rays_o = torch.reshape(rays_o, (-1, 3))
        rays_d = torch.reshape(rays_d, (-1, 3))
        batch_rays = torch.stack([rays_o, rays_d], 0)
        rgb = []
        rgb_d = []
        rgb_s = []
        dynamicness = []
        for j in range(0, batch_rays.shape[1], chunk):
            # print(j, '/', batch_rays.shape[1])
            ret = render(t, False,
                         H, W, focal,
                         chunk=chunk, rays=batch_rays[:, j:j+chunk, :],
                         **render_kwargs)
            rgb.append(ret['rgb_map_full'].cpu())
            rgb_d.append(ret['rgb_map_d'].cpu())
            rgb_s.append(ret['rgb_map_s'].cpu())
            dynamicness.append(ret['dynamicness_map'].cpu())
        rgb = torch.reshape(torch.cat(rgb, 0), (H, W, 3)).numpy()
        rgb_d = torch.reshape(torch.cat(rgb_d, 0), (H, W, 3)).numpy()
        rgb_s = torch.reshape(torch.cat(rgb_s, 0), (H, W, 3)).numpy()
        dynamicness = torch.reshape(torch.cat(dynamicness, 0), (H, W)).numpy()

        # Not a good solution. Should take care of this when preparing the data.
        if W%2 == 1:
            # rgb = cv2.resize(rgb, (W - 1, H))
            rgb = rgb[:, :-1, :]
            rgb_d = rgb_d[:, :-1, :]
            rgb_s = rgb_s[:, :-1, :]
            dynamicness = dynamicness[:, :-1]
        rgbs.append(rgb)
        rgbs_d.append(rgb_d)
        rgbs_s.append(rgb_s)
        dynamicnesses.append(dynamicness)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    ret_dict['rgbs'] = np.stack(rgbs, 0)
    ret_dict['rgbs_d'] = np.stack(rgbs_d, 0)
    ret_dict['rgbs_s'] = np.stack(rgbs_s, 0)
    ret_dict['dynamicnesses'] = np.stack(dynamicnesses, 0)

    return ret_dict

def raw2outputs(raw_s,
                raw_d,
                blending,
                z_vals,
                rays_d,
                raw_noise_std):
    """Transforms model's predictions to semantically meaningful values.

    Args:
      raw_d: [num_rays, num_samples along ray, 4]. Prediction from Dynamic model.
      raw_s: [num_rays, num_samples along ray, 4]. Prediction from Static model.
      z_vals: [num_rays, num_samples along ray]. Integration time.
      rays_d: [num_rays, 3]. Direction of each ray.

    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
      disp_map: [num_rays]. Disparity map. Inverse of depth map.
      acc_map: [num_rays]. Sum of weights along each ray.
      weights: [num_rays, num_samples]. Weights assigned to each sampled color.
      depth_map: [num_rays]. Estimated distance to object.
    """
    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(raw, dists, act_fn=F.relu): return 1.0 - \
        torch.exp(-act_fn(raw) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)],
         -1) # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb_d = torch.sigmoid(raw_d[..., :3])  # [N_rays, N_samples, 3]
    rgb_s = torch.sigmoid(raw_s[..., :3])  # [N_rays, N_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_d[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_d = raw2alpha(raw_d[..., 3] + noise, dists) # [N_rays, N_samples]
    alpha_s = raw2alpha(raw_s[..., 3] + noise, dists) # [N_rays, N_samples]
    alphas  = 1. - (1. - alpha_s) * (1. - alpha_d) # [N_rays, N_samples]

    T_d    = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), 1. - alpha_d + 1e-10], -1), -1)[:, :-1]
    T_s    = torch.cumprod(torch.cat([torch.ones((alpha_s.shape[0], 1)), 1. - alpha_s + 1e-10], -1), -1)[:, :-1]
    T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), (1. - alpha_d * blending) * (1. - alpha_s * (1. - blending)) + 1e-10], -1), -1)[:, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), torch.pow(1. - alpha_d + 1e-10, blending) * torch.pow(1. - alpha_s + 1e-10, 1. - blending)], -1), -1)[:, :-1]
    # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), (1. - alpha_d) * (1. - alpha_s) + 1e-10], -1), -1)[:, :-1]

    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_d = alpha_d * T_d
    weights_s = alpha_s * T_s
    weights_full = (alpha_d * blending + alpha_s * (1. - blending)) * T_full
    # weights_full = alphas * T_full

    # Computed weighted color of each sample along each ray.
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)
    rgb_map_s = torch.sum(weights_s[..., None] * rgb_s, -2)
    rgb_map_full = torch.sum(
        (T_full * alpha_d * blending)[..., None] * rgb_d + \
        (T_full * alpha_s * (1. - blending))[..., None] * rgb_s, -2)

    # Estimated depth map is expected distance.
    depth_map_d = torch.sum(weights_d * z_vals, -1)
    depth_map_s = torch.sum(weights_s * z_vals, -1)
    depth_map_full = torch.sum(weights_full * z_vals, -1)

    # Sum of weights along each ray. This value is in [0, 1] up to numerical error.
    acc_map_d = torch.sum(weights_d, -1)
    acc_map_s = torch.sum(weights_s, -1)
    acc_map_full = torch.sum(weights_full, -1)

    # Computed dynamicness
    dynamicness_map = torch.sum(weights_full * blending, -1)
    # dynamicness_map = 1 - T_d[..., -1]

    return rgb_map_full, depth_map_full, acc_map_full, weights_full, \
           rgb_map_s, depth_map_s, acc_map_s, weights_s, \
           rgb_map_d, depth_map_d, acc_map_d, weights_d, dynamicness_map


def raw2outputs_d(raw_d,
                  z_vals,
                  rays_d,
                  raw_noise_std):

    # Function for computing density from model prediction. This value is
    # strictly between [0, 1].
    def raw2alpha(raw, dists, act_fn=F.relu): return 1.0 - \
        torch.exp(-act_fn(raw) * dists)

    # Compute 'distance' (in time) between each integration time along a ray.
    dists = z_vals[..., 1:] - z_vals[..., :-1]

    # The 'distance' from the last integration time is infinity.
    dists = torch.cat(
        [dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)],
        -1)  # [N_rays, N_samples]

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Extract RGB of each sample position along each ray.
    rgb_d = torch.sigmoid(raw_d[..., :3])  # [N_rays, N_samples, 3]

    # Add noise to model's predictions for density. Can be used to
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw_d[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point.
    alpha_d = raw2alpha(raw_d[..., 3] + noise, dists)  # [N_rays, N_samples]

    T_d = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), 1. - alpha_d + 1e-10], -1), -1)[:, :-1]
    # Compute weight for RGB of each sample along each ray.  A cumprod() is
    # used to express the idea of the ray not having reflected up to this
    # sample yet.
    weights_d = alpha_d * T_d

    # Computed weighted color of each sample along each ray.
    rgb_map_d = torch.sum(weights_d[..., None] * rgb_d, -2)

    return rgb_map_d, weights_d


def render_rays(t,
                chain_5frames,
                ray_batch,
                network_fn_d,
                network_fn_s,
                network_query_fn_d,
                network_query_fn_s,
                N_samples,
                num_img,
                DyNeRF_blending,
                pretrain=False,
                lindisp=False,
                perturb=0.,
                N_depthguide=1,
                raw_noise_std=0.,
                inference=False,
                bi_time=None,
                e_render=False,
                num_interp=4):

    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn_d: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn_d: function used for passing queries to network_fn_d.
      N_samples: int. Number of different times to sample along each ray.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """

    # batch size
    N_rays = ray_batch.shape[0]

    # ray_batch: [N_rays, 11]
    # rays_o:    [N_rays, 0:3]
    # rays_d:    [N_rays, 3:6]
    # near:      [N_rays, 6:7]
    # far:       [N_rays, 7:8]
    # viewdirs:  [N_rays, 8:11]

    # Extract ray origin, direction.
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6] # [N_rays, 3] each

    # Extract unit-normalized viewing direction.
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None

    # Extract lower, upper bound for ray distance.
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    # Decide where to sample along each ray. Under the logic, all rays will be sampled at
    # the same times.
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        # Space integration times linearly between 'near' and 'far'. Same
        # integration points will be used for all rays.
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # Sample linearly in inverse depth (disparity).
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])

    # Perturb sampling time along each ray.
    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand

    # Points in space to evaluate model at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None] # [N_rays, N_samples, 3]

    # Add the time dimension to xyz.
    pts_ref = torch.cat([pts, torch.ones_like(pts[..., 0:1]) * t], -1)

    # First pass: we have the staticNeRF results
    raw_s = network_query_fn_s(pts_ref[..., :3], viewdirs, network_fn_s)
    # raw_s:          [N_rays, N_samples, 5]
    # raw_s_rgb:      [N_rays, N_samples, 0:3]
    # raw_s_a:        [N_rays, N_samples, 3:4]
    # raw_s_blending: [N_rays, N_samples, 4:5]

    # Second pass: we have the DyanmicNeRF results and the blending weight
    raw_d = network_query_fn_d(pts_ref, viewdirs, network_fn_d)
    # raw_d:          [N_rays, N_samples, 26]
    # raw_d_rgb:      [N_rays, N_samples, 0:3]
    # raw_d_a:        [N_rays, N_samples, 3:4]
    # sceneflow_cp:    [N_rays, N_samples, 4:25]
    # raw_d_blending: [N_rays, N_samples, 25:26]
    if pretrain:
        rgb_map_s, _ = raw2outputs_d(raw_s[..., :4],
                                     z_vals,
                                     rays_d,
                                     raw_noise_std)
        ret = {'rgb_map_s': rgb_map_s}
        return ret

    raw_s_rgba = raw_s[..., :4]
    raw_d_rgba = raw_d[..., :4]

    # We need the sceneflow from the dynamicNeRF.
    sceneflow_cp = raw_d[..., 4:25]
    sceneflow_f, sceneflow_b, sceneflow_f_mid, sceneflow_b_mid = compute_biflow(sceneflow_cp, bi_time, num_interp)
    raw_s_rgba = raw_s[..., :4]
    raw_d_rgba = raw_d[..., :4]
    if DyNeRF_blending:
        blending = raw_d[..., 25] #blending=1为动态 blending=0为静态
    else:
        blending = raw_s[..., 4]

    # if sfmask:
    #     sceneflow_f = sceneflow_f * blending.detach()[..., None]
    #     sceneflow_b = sceneflow_b * blending.detach()[..., None]

    # Rerndering.
    rgb_map_full, depth_map_full, acc_map_full, weights_full, \
    rgb_map_s, depth_map_s, acc_map_s, weights_s, \
    rgb_map_d, depth_map_d, acc_map_d, weights_d, \
    dynamicness_map = raw2outputs(raw_s_rgba,
                                  raw_d_rgba,
                                  blending,
                                  z_vals,
                                  rays_d,
                                  raw_noise_std)

    ret = {'rgb_map_full': rgb_map_full, 'depth_map_full': depth_map_full, 'acc_map_full': acc_map_full, 'weights_full': weights_full,
           'rgb_map_s': rgb_map_s, 'depth_map_s': depth_map_s, 'acc_map_s': acc_map_s, 'weights_s': weights_s,
           'rgb_map_d': rgb_map_d, 'depth_map_d': depth_map_d, 'acc_map_d': acc_map_d, 'weights_d': weights_d,
           'dynamicness_map': dynamicness_map}

    t_interval = 1. / num_img * 2.
    rand_f = bi_time['rand_f']
    rand_b = bi_time['rand_b']
    inv_bi_time = {'rand_f':rand_b, 'rand_b': rand_f}
    pts_f = torch.cat([pts + sceneflow_f, torch.ones_like(pts[..., 0:1]) * (t + t_interval)], -1)
    pts_b = torch.cat([pts + sceneflow_b, torch.ones_like(pts[..., 0:1]) * (t - t_interval)], -1)
    pts_fmid = torch.cat([pts + sceneflow_f_mid, 
                          torch.ones_like(pts[..., 0:1]) * (t + t_interval * (rand_f+1)/num_interp)], -1)
    pts_bmid = torch.cat([pts + sceneflow_b_mid, 
                          torch.ones_like(pts[..., 0:1]) * (t - t_interval * (rand_b+1)/num_interp)], -1)
    
    if inference:
        ret['pts_f_map_NDC'] = torch.sum(weights_d[..., None] * pts_f[..., :3], -2)
        ret['pts_b_map_NDC'] = torch.sum(weights_d[..., None] * pts_b[..., :3], -2)
        ret['pts_fmid_map_NDC'] = torch.sum(weights_d[..., None] * pts_fmid[..., :3], -2)
        ret['pts_bmid_map_NDC'] = torch.sum(weights_d[..., None] * pts_bmid[..., :3], -2)
        return ret

    ret['sceneflow_b'] = sceneflow_b
    ret['sceneflow_f'] = sceneflow_f
    ret['sceneflow_bmid'] = sceneflow_b_mid
    ret['sceneflow_fmid'] = sceneflow_f_mid
    ret['raw_pts'] = pts_ref[..., :3]
    ret['raw_pts_f'] = pts_f[..., :3]
    ret['raw_pts_b'] = pts_b[..., :3]
    ret['blending'] = blending

    # Third pass: we have the DyanmicNeRF results at time t - 1
    raw_d_b = network_query_fn_d(pts_b, viewdirs, network_fn_d)
    raw_d_b_rgba = raw_d_b[..., :4]
    sceneflow_b_cp = raw_d_b[..., 4:25]
    sceneflow_b_f, sceneflow_b_b, _, _ = compute_biflow(sceneflow_b_cp, bi_time, num_interp)

    # Rerndering t - 1
    rgb_map_d_b, weights_d_b = raw2outputs_d(raw_d_b_rgba,
                                             z_vals,
                                             rays_d,
                                             raw_noise_std)

    ret['sceneflow_b_f'] = sceneflow_b_f
    ret['rgb_map_d_b'] = rgb_map_d_b
    ret['acc_map_d_b'] = torch.abs(torch.sum(weights_d_b - weights_d, -1))

    # Fourth pass: we have the DyanmicNeRF results at time t + 1
    raw_d_f = network_query_fn_d(pts_f, viewdirs, network_fn_d)
    raw_d_f_rgba = raw_d_f[..., :4]
    sceneflow_f_cp = raw_d_f[..., 4:25]
    sceneflow_f_f, sceneflow_f_b, _, _ = compute_biflow(sceneflow_f_cp, bi_time, num_interp)

    rgb_map_d_f, weights_d_f = raw2outputs_d(raw_d_f_rgba,
                                             z_vals,
                                             rays_d,
                                             raw_noise_std)

    ret['sceneflow_f_b'] = sceneflow_f_b
    ret['rgb_map_d_f'] = rgb_map_d_f
    ret['acc_map_d_f'] = torch.abs(torch.sum(weights_d_f - weights_d, -1))

    # --------------------------event to middle image part-----------------------------------
    # Third pass: we have the DynamicNeRF results at time t - 1
    if e_render:
        raw_d_bmid = network_query_fn_d(pts_bmid, viewdirs, network_fn_d)
        raw_d_bmid_rgba = raw_d_bmid[..., :4]
        #blending_bmid = raw_d_bmid[..., 25]
        #_, _, sf_hb_hf, sf_hb_hb = compute_biflow(sceneflow_bmid_cp, bi_time, num_interp)

        # Rerndering t - 1
        rgb_map_d_bmid, _ = raw2outputs_d(raw_d_bmid_rgba,
                                                z_vals,
                                                rays_d,
                                                raw_noise_std)

        # Fourth pass: we have the DyanmicNeRF results at time t + 1
        raw_d_fmid = network_query_fn_d(pts_fmid, viewdirs, network_fn_d)
        raw_d_fmid_rgba = raw_d_fmid[..., :4]
        #blending_fmid = raw_d_fmid[..., 25]
        #_, _, sf_hf_hf, sf_hf_hb = compute_biflow(sceneflow_fmid_cp, bi_time, num_interp)

        rgb_map_d_fmid, _ = raw2outputs_d(raw_d_fmid_rgba,
                                                z_vals,
                                                rays_d,
                                                raw_noise_std)
  
        ret = {'rgb_map_d': rgb_map_d, 'depth_map_d': depth_map_d, 'weights_d': weights_d,
               'dynamicness_map': dynamicness_map, 'sceneflow_bmid': sceneflow_b_mid,
               'sceneflow_fmid': sceneflow_f_mid, 'rgb_map_d_fmid': rgb_map_d_fmid, 
               'rgb_map_d_bmid': rgb_map_d_bmid, 'sceneflow_f': sceneflow_f, 
               'sceneflow_b': sceneflow_b, }
        return ret
    # -------------------------------end-----------------------------------------------------
    raw_d_bmid = network_query_fn_d(pts_bmid, viewdirs, network_fn_d)
    raw_d_bmid_rgba = raw_d_bmid[..., :4]
    sceneflow_bmid_cp = raw_d_bmid[..., 4:25]
    blending_bmid = raw_d_bmid[..., 25]
    _, _, sf_hb_hf, sf_hb_hb = compute_biflow(sceneflow_bmid_cp, inv_bi_time, num_interp)

    rgb_map_d_bmid, _ = raw2outputs_d(raw_d_bmid_rgba,
                                            z_vals,
                                            rays_d,
                                            raw_noise_std)

    raw_d_fmid = network_query_fn_d(pts_fmid, viewdirs, network_fn_d)
    raw_d_fmid_rgba = raw_d_fmid[..., :4]
    sceneflow_fmid_cp = raw_d_fmid[..., 4:25]
    blending_fmid = raw_d_fmid[..., 25]
    _, _, sf_hf_hf, sf_hf_hb = compute_biflow(sceneflow_fmid_cp, inv_bi_time, num_interp)

    rgb_map_d_fmid, _ = raw2outputs_d(raw_d_fmid_rgba,
                                            z_vals,
                                            rays_d,
                                            raw_noise_std)
    ret['rgb_map_d_bmid'] = rgb_map_d_bmid
    ret['rgb_map_d_fmid'] = rgb_map_d_fmid
    ret['sf_hf_hf'] = sf_hf_hf
    ret['sf_hf_hb'] = sf_hf_hb
    ret['sf_hb_hf'] = sf_hb_hf
    ret['sf_hb_hb'] = sf_hb_hb
    ret['blending_fmid'] = blending_fmid
    ret['blending_bmid'] = blending_bmid

    # Also consider time t - 2 and t + 2 (Learn from NSFF)

    # Fifth pass: we have the DyanmicNeRF results at time t - 2
    pts_b_b = torch.cat([pts_b[..., :3] + sceneflow_b_b, torch.ones_like(pts[..., 0:1]) * (t - t_interval * 2)], -1)
    ret['raw_pts_b_b'] = pts_b_b[..., :3]

    if chain_5frames:
        raw_d_b_b = network_query_fn_d(pts_b_b, viewdirs, network_fn_d)
        raw_d_b_b_rgba = raw_d_b_b[..., :4]
        rgb_map_d_b_b, _ = raw2outputs_d(raw_d_b_b_rgba,
                                      z_vals,
                                      rays_d,
                                      raw_noise_std)

        ret['rgb_map_d_b_b'] = rgb_map_d_b_b

    # Sixth pass: we have the DyanmicNeRF results at time t + 2
    pts_f_f = torch.cat([pts_f[..., :3] + sceneflow_f_f, torch.ones_like(pts[..., 0:1]) * (t + t_interval * 2)], -1)
    ret['raw_pts_f_f'] = pts_f_f[..., :3]

    if chain_5frames:
        raw_d_f_f = network_query_fn_d(pts_f_f, viewdirs, network_fn_d)
        raw_d_f_f_rgba = raw_d_f_f[..., :4]
        rgb_map_d_f_f, _ = raw2outputs_d(raw_d_f_f_rgba,
                                      z_vals,
                                      rays_d,
                                      raw_noise_std)

        ret['rgb_map_d_f_f'] = rgb_map_d_f_f

    for k in ret:
        if torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any():
            print(f"! [Numerical Error] {k} contains nan or inf.")
            import pdb; pdb.set_trace()

    return ret

#DIY function
def compute_biflow(sceneflow_cp, bi_time, num_interp):
    time_f = bi_time['rand_f']
    time_b = bi_time['rand_b']
    cp_backward = sceneflow_cp[..., :12]
    cp_backward = cp_backward.flip(dims=[-1]).reshape(sceneflow_cp.shape[0], sceneflow_cp.shape[1],4,3) #[N_rays, N_samples,4, 3] 倒置输出，第一行是原点
    cp_forward = sceneflow_cp[..., 9:].reshape(sceneflow_cp.shape[0], sceneflow_cp.shape[1],4,3) #[N_rays, N_samples,4, 3] 第一行是原点
    B_num = np.array([[comb(3, i)*(1-(t+1)/num_interp)**(3-i)*((t+1)/num_interp)**i for i in range(4)] for t in range(num_interp)])
    B_num = torch.Tensor(B_num) #[num_interp, 4]
    forward_query = [time_f, num_interp-1] #time_f = 0~num_interp-2
    backward_query = [time_b, num_interp-1]
    sceneflow_f_set = torch.matmul(B_num[forward_query], cp_forward)#[N_rays, N_samples,2,3]
    sceneflow_b_set = torch.matmul(B_num[backward_query], cp_backward)
    sceneflow_f = sceneflow_f_set[..., 1, :]
    sceneflow_b = sceneflow_b_set[..., 1, :]
    sceneflow_f_mid = sceneflow_f_set[..., 0, :]
    sceneflow_b_mid = sceneflow_b_set[..., 0, :]

    return sceneflow_f, sceneflow_b, sceneflow_f_mid, sceneflow_b_mid