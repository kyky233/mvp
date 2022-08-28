"""
use this script to check if mvhw dataset can load target data
by: ydq
"""
import copy
import os
import glob
import json
import numpy as np

import cv2
import matplotlib.pyplot as plt


dataset_root = '/home/yandanqi/0_data/MVHW'
sequence_list = os.listdir('/home/yandanqi/0_data/MVHW')
sequence_list = [d for d in sequence_list if '_o' in d]

num_views = 5
cam_list = ['c0{}'.format(i) for i in range(1, 9, 1)][:num_views]

_interval = 1
num_joints = 15


def _get_img_name(idx):
    return str(idx+1).zfill(6)+'.jpg'


def _get_cam(seq):
    cam_file = os.path.join(dataset_root, seq, 'cameras.json')
    with open(cam_file) as cfile:
        calib = json.load(cfile)

    M = np.array([[1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]])
    cameras = {}
    for cam in calib:
        if cam['name'] in cam_list:    # # 'c01'-'c05' cams if num_list=5
            sel_cam = dict()
            sel_cam['K'] = np.array(cam['matrix'])
            sel_cam['distCoef'] = np.array(cam['distortions'])
            # sel_cam['R'] = cv2.Rodrigues(np.array(cam['rotation']))[0].dot(M)
            sel_cam['R'] = cv2.Rodrigues(np.array(cam['rotation']))[0]  # rotation vector to rotation matrix
            sel_cam['t'] = np.array(cam['translation']).reshape((3, 1))    # cm to mm
            cameras[cam['name']] = sel_cam
    return cameras


def _get_db():
    width = 1920
    height = 1080
    db = []
    for seq in sequence_list:

        # get camera anno in this seq
        cameras = _get_cam(seq)

        # get image dir
        img_dir = os.path.join(dataset_root, seq, 'vframes')

        # get length of this seq
        seq_len = len(glob.glob(os.path.join(img_dir, 'c01', '*.jpg')))

        # curr_anno = osp.join(self.dataset_root, seq, 'hdPose3d_stage1_coco19')
        # anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))

        for idx in range(seq_len):
            if idx % _interval == 0:
                # with open(file) as dfile:
                #     bodies = json.load(dfile)['bodies']
                # if len(bodies) == 0:
                #     continue

                for k, v in cameras.items():
                    # get image
                    img_path = os.path.join(img_dir, k, _get_img_name(idx))

                    # get pose
                    all_poses_3d = []
                    all_poses_vis_3d = []
                    all_poses = []
                    all_poses_vis = []
                    # fake pose (assume one person in each image)
                    pose3d = np.zeros(shape=[num_joints, 3])
                    all_poses_3d.append(pose3d)
                    pose3d_vis = pose3d
                    all_poses_vis.append(pose3d_vis)
                    pose2d = np.zeros(shape=[pose3d.shape[0], 2])
                    all_poses.append(pose2d)
                    pose2d_vis = pose2d
                    all_poses_vis.append(pose2d_vis)

                    # get each cam
                    our_cam = dict()
                    our_cam['R'] = v['R']
                    our_cam['T'] = v['t']
                    our_cam['standard_T'] = -np.dot(v['R'], v['t'])
                    our_cam['fx'] = np.array(v['K'][0, 0])
                    our_cam['fy'] = np.array(v['K'][1, 1])
                    our_cam['cx'] = np.array(v['K'][0, 2])
                    our_cam['cy'] = np.array(v['K'][1, 2])
                    our_cam['k'] = v['distCoef'][[0, 1, 2]].reshape(3, 1)
                    our_cam['p'] = v['distCoef'][[3, 4]].reshape(2, 1)

                    db.append({
                        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
                        'image': img_path,
                        'joints_3d': all_poses_3d,
                        'joints_3d_vis': all_poses_vis_3d,
                        'joints_2d': all_poses,
                        'joints_2d_vis': all_poses_vis,
                        'camera': our_cam
                    })
    return db


def _get_group_item(db, idx):
    """
    db_rec:
        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
        'image': img_path,
        'joints_3d': all_poses_3d,
        'joints_3d_vis': all_poses_vis_3d,
        'joints_2d': all_poses,
        'joints_2d_vis': all_poses_vis,
        'camera': our_cam
    """
    keys = []
    cameras = []
    images = []

    # collect group data
    for k in range(num_views):
        cur_idx = num_views * idx + k

        db_rec = copy.deepcopy(db[cur_idx])
        keys.append(db_rec['key'])
        cameras.append(db_rec['camera'])
        images.append(db_rec['image'])

    group_rec = dict()
    group_rec['key'] = keys
    group_rec['camera'] = cameras
    group_rec['image'] = images

    return group_rec


def show_db_item(db_rec):
    """
    db_rec:
        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
        'image': img_path,
        'joints_3d': all_poses_3d,
        'joints_3d_vis': all_poses_vis_3d,
        'joints_2d': all_poses,
        'joints_2d_vis': all_poses_vis,
        'camera': our_cam
    """
    print(f"key of this rec = {db_rec['key']}")
    print(f"camera = {db_rec['camera']}")
    print(f"shape of joints_3d = {len(db_rec['joints_3d'])}")

    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_subplot(1, 1, 1)
    img = plt.imread(db_rec['image'])
    ax.imshow(img)

    plt.show()
    plt.close()


def show_db_single_group(db_group_rec):
    """
    db_group_rec:
        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
        'image': img_path,
        'camera': our_cam
    """
    keys = db_group_rec['key']
    cameras = db_group_rec['camera']
    images = db_group_rec['image']

    # show group data
    for i in range(num_views):
        print(f"key of this rec = {keys[i]}")
        print(f"camera = {cameras[i]}")
        print(f"image = {images[i]}")

    fig = plt.figure(figsize=(1, num_views))
    for i in range(num_views):
        ax = fig.add_subplot(1, num_views, i+1)
        img = plt.imread(images[i])
        ax.imshow(img)

    plt.show()
    plt.close()


def show_db_groups(db, num_groups, group_intervals, idx):
    """
    db_group_rec:
        'key': "{}_{}-{}".format(seq, k, _get_img_name(idx).split('.')[0]),
        'image': img_path,
        'camera': our_cam
    """
    # collect data
    keys = []
    cameras = []
    images = []

    for i in range(num_groups):
        db_group_rec = _get_group_item(db=db, idx=idx + i*group_intervals)
        keys.append(db_group_rec['key'])
        cameras.append(db_group_rec['camera'])
        images.append(db_group_rec['image'])

    # show data
    for i in range(num_groups):
        print(f"key of db[{idx + i*group_intervals}] = {keys[i]}")
        print(f"image of db[{idx + i*group_intervals}]= {images[i]}")

    fig = plt.figure()
    for i in range(num_groups):
        for j in range(num_views):
            ax = fig.add_subplot(num_groups, num_views, i*num_views + j+1)
            img = plt.imread(images[i][j])
            ax.imshow(img)

    plt.show()
    plt.close()


def main():
    db = _get_db()
    print(f"length of dataset = {len(db)}")

    idx = 0

    vis_db_item = False
    vis_single_group_db = True
    vis_several_groups_db = False

    '''-----------show db by idx--------------'''
    if vis_db_item:
        db_rec = copy.deepcopy(db[idx])
        show_db_item(db_rec=db_rec)
        print(f"{idx}th rec has been showed!")

    '''-----------show group db by idx-------------'''
    if vis_single_group_db:
        db_group_rec = _get_group_item(db=db, idx=idx)
        show_db_single_group(db_group_rec=db_group_rec)
        print(f"{idx}th group recs has been showed!")

    '''-------------------show several groups db rec-------------------'''
    if vis_several_groups_db:
        num_groups = 7
        group_intervals = 500
        show_db_groups(db=db, num_groups=num_groups, group_intervals=group_intervals, idx=idx)
        print(f"{idx}th~{idx+num_groups}th group recs have been showed!")

    print(f"this is the end!")


if __name__ == '__main__':
    main()
