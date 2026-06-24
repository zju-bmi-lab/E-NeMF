import numpy as np
import cv2
import os
import time

def combine(ev1, ev2):#same direction
    event2 = np.concatenate([ev1[..., :2]+ev2[..., :2], ev1[..., 2:4], ev2[..., 4:]], -1)
    return event2

def load_event_timereplayer(basedir, H, W, num_interp=4, num_img=12):
    e_dir = os.path.join(basedir, 'events')
    #timestamp = np.load(os.path.join(basedir, 'timestamp.txt'))
    pos_e = np.zeros([num_img - 1, num_interp, H, W], dtype=np.uint8)
    neg_e = np.zeros([num_img - 1, num_interp, H, W], dtype=np.uint8)
    pos_tbegin = np.ones([num_img - 1, num_interp, H, W], dtype=np.float32)
    neg_tbegin = np.ones([num_img - 1, num_interp, H, W], dtype=np.float32)
    pos_tend = np.ones([num_img - 1, num_interp, H, W], dtype=np.float32)
    neg_tend = np.ones([num_img - 1, num_interp, H, W], dtype=np.float32)
    timestamp = []
    t_dir = os.path.join(basedir, 'timestamp_eval.txt')
    with open(t_dir, 'r') as timef:
        for ts in timef.readlines():
            ts = ts.strip('\n')      
            timestamp.append(float(ts))
    for i in range(num_img - 1):
        tmin = timestamp[i*num_interp]
        pos_tbegin[i,:,:,:] *= timestamp[i*num_interp+num_interp]
        neg_tbegin[i,:,:,:] *= timestamp[i*num_interp+num_interp]
        pos_tend[i,:,:,:] *= tmin
        neg_tend[i,:,:,:] *= tmin

        e_path = os.path.join(e_dir, '%03d.npz'%(i+1))
        e_data = np.load(e_path)
        x = e_data['x']
        x = np.clip(x, 0, W-1)
        x = np.round(x).astype(np.uint16)
        y = e_data['y']
        y = np.round(y).astype(np.uint16)
        y = np.clip(y, 0, H-1)
        t = e_data['t']
        p = e_data['p']
        for j in range(num_interp):
            dt = timestamp[i*num_interp+j+1] - timestamp[i*num_interp+j]
            t_now = timestamp[i*num_interp+j]
            np.add.at(pos_e[i,j,:,:], (y[np.where((p==1)&(t<=t_now+dt)&(t>=tmin))], \
                                            x[np.where((p==1)&(t<=t_now+dt)&(t>=tmin))]), 1)
            np.add.at(neg_e[i,j,:,:], (y[np.where((p==0)&(t<=t_now+dt)&(t>=tmin))], \
                                            x[np.where((p==0)&(t<=t_now+dt)&(t>=tmin))]), 1)
            np.maximum.at(pos_tend[i,j,:,:], (y[np.where((p==1)&(t<=t_now+dt)&(t>=tmin))], \
                                                    x[np.where((p==1)&(t<=t_now+dt)&(t>=tmin))]), \
                                                    t[np.where((p==1)&(t<=t_now+dt)&(t>=tmin))])
            np.maximum.at(neg_tend[i,j,:,:], (y[np.where((p==0)&(t<=t_now+dt)&(t>=tmin))], \
                                                    x[np.where((p==0)&(t<=t_now+dt)&(t>=tmin))]), \
                                                    t[np.where((p==0)&(t<=t_now+dt)&(t>=tmin))])
            np.minimum.at(pos_tbegin[i,j,:,:], (y[np.where((p==1)&(t<=t_now+dt)&(t>=tmin))], \
                                                        x[np.where((p==1)&(t<=t_now+dt)&(t>=tmin))]), \
                                                        t[np.where((p==1)&(t<=t_now+dt)&(t>=tmin))])
            np.minimum.at(neg_tbegin[i,j,:,:], (y[np.where((p==0)&(t<=t_now+dt)&(t>=tmin))], \
                                                        x[np.where((p==0)&(t<=t_now+dt)&(t>=tmin))]), \
                                                        t[np.where((p==0)&(t<=t_now+dt)&(t>=tmin))])
            pos_tbegin[i,j][np.where((pos_e[i,j,:,:]==0))] = tmin
            neg_tbegin[i,j][np.where((neg_e[i,j,:,:]==0))] = tmin
            pos_tend[i,j][np.where((pos_e[i,j,:,:]==0))] = t_now + dt
            neg_tend[i,j][np.where((neg_e[i,j,:,:]==0))] = t_now + dt
            t_now = t_now + dt

    print(np.max(pos_e))
    event = np.concatenate([np.expand_dims(pos_e,2), np.expand_dims(neg_e,2), np.expand_dims(pos_tbegin,2), \
                            np.expand_dims(neg_tbegin,2), np.expand_dims(pos_tend,2), np.expand_dims(neg_tend,2)], 2)
    return event, np.array(timestamp)

if __name__ == '__main__':
    load_event_timereplayer('/home/ly/NSFF/nerf_data/balloon/dense', 856, 957)