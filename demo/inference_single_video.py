import sys
import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
from lib.yolo_pose import gen_video_kpts_yolo as yolo_pose
import os
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
import pandas as pd

sys.path.append(os.getcwd())
from demo.lib.utils import normalize_screen_coordinates, camera_to_world
from model.MotionAGFormer import MotionAGFormer

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    """
    CGTGait order (16 joints, 0-15):
    0: Pelvis
    1: Spine
    2: Neck
    3: Head
    4: Right Shoulder
    5: Right Elbow
    6: Right Wrist
    7: Left Shoulder
    8: Left Elbow
    9: Left Wrist
    10: Right Hip
    11: Right Knee
    12: Right Ankle
    13: Left Hip
    14: Left Knee
    15: Left Ankle
    """
    ax.view_init(elev=15., azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    # CGTGait order connections
    I = np.array([0, 1, 2, 2, 4, 5, 2, 7, 8,  0,  10, 11, 0,  13, 14])
    J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])

    # Left=1, Right=0
    LR = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)

    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)


def img2video(video_path, output_dir, video_name):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    if not names:
        print('No pose images found to compile into video.')
        return

    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + video_name + '.mp4', fourcc, fps, size)

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def get_pose2D(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    # Temporarily save and clear sys.argv to avoid conflicts with hrnet_pose's argument parser
    saved_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep only the script name
    try:
        keypoints, scores = yolo_pose(video_path, imgsz=640)
    finally:
        sys.argv = saved_argv  # Restore original argv
    
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    # Save H36M-order CSV (used by get_pose3D as model input — do NOT reorder)
    output_csv = output_dir + 'keypoints.csv'
    num_frames = keypoints.shape[1]
    num_joints = keypoints.shape[2]

    data_list = []
    for frame_idx in range(num_frames):
        for joint_idx in range(num_joints):
            row = [frame_idx, joint_idx] + keypoints[0, frame_idx, joint_idx, :].tolist()
            data_list.append(row)

    df = pd.DataFrame(data_list, columns=['frame', 'joint', 'x', 'y', 'confidence'])
    df.to_csv(output_csv, index=False)
    print(f'Saved 2D keypoints (H36M order, for 3D pipeline) to {output_csv}')

    # Also save CGTGait-order CSV so 2D output matches 3D output format
    keypoints_cgtgait_2d = h36m_to_cgtgait_order(keypoints)  # (1, T, 16, 3)
    output_csv_cgtgait = output_dir + 'keypoints_cgtgait_order.csv'

    joint_names_2d = ['Pelvis', 'Spine', 'Neck', 'Head',
                      'RShoulder', 'RElbow', 'RWrist',
                      'LShoulder', 'LElbow', 'LWrist',
                      'RHip', 'RKnee', 'RAnkle',
                      'LHip', 'LKnee', 'LAnkle']

    data_list_cgtgait = []
    num_joints_cg = keypoints_cgtgait_2d.shape[2]
    for frame_idx in range(num_frames):
        for joint_idx in range(num_joints_cg):
            row = ([frame_idx, joint_idx, joint_names_2d[joint_idx]]
                   + keypoints_cgtgait_2d[0, frame_idx, joint_idx, :].tolist())
            data_list_cgtgait.append(row)

    df_cgtgait = pd.DataFrame(data_list_cgtgait,
                              columns=['frame', 'joint_id', 'joint_name', 'x', 'y', 'confidence'])
    df_cgtgait.to_csv(output_csv_cgtgait, index=False)
    print(f'Saved 2D keypoints (CGTGait order) to {output_csv_cgtgait}')


def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]

    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)

    return clips, downsample

def h36m_to_cgtgait_order(keypoints_h36m):
    """
    Convert H36M order (17 joints) to CGTGait order (16 joints, dropping Thorax).

    H36M index reference:
      0=Pelvis, 1=R Hip, 2=R Knee, 3=R Ankle, 4=L Hip, 5=L Knee, 6=L Ankle,
      7=Spine, 8=Thorax (dropped), 9=Neck/Nose, 10=Head,
      11=L Shoulder, 12=L Elbow, 13=L Wrist,
      14=R Shoulder, 15=R Elbow, 16=R Wrist
    """
    new_shape = list(keypoints_h36m.shape)
    new_shape[-2] = 16
    keypoints_cgtgait = np.zeros(new_shape, dtype=keypoints_h36m.dtype)

    # Map H36M to CGTGait order
    keypoints_cgtgait[..., 0, :]  = keypoints_h36m[..., 0, :]   # Pelvis
    keypoints_cgtgait[..., 1, :]  = keypoints_h36m[..., 7, :]   # Spine
    keypoints_cgtgait[..., 2, :]  = keypoints_h36m[..., 9, :]   # Neck/Nose
    keypoints_cgtgait[..., 3, :]  = keypoints_h36m[..., 10, :]  # Head
    keypoints_cgtgait[..., 4, :]  = keypoints_h36m[..., 14, :]  # Right Shoulder
    keypoints_cgtgait[..., 5, :]  = keypoints_h36m[..., 15, :]  # Right Elbow
    keypoints_cgtgait[..., 6, :]  = keypoints_h36m[..., 16, :]  # Right Wrist
    keypoints_cgtgait[..., 7, :]  = keypoints_h36m[..., 11, :]  # Left Shoulder
    keypoints_cgtgait[..., 8, :]  = keypoints_h36m[..., 12, :]  # Left Elbow
    keypoints_cgtgait[..., 9, :]  = keypoints_h36m[..., 13, :]  # Left Wrist
    keypoints_cgtgait[..., 10, :] = keypoints_h36m[..., 1, :]   # Right Hip
    keypoints_cgtgait[..., 11, :] = keypoints_h36m[..., 2, :]   # Right Knee
    keypoints_cgtgait[..., 12, :] = keypoints_h36m[..., 3, :]   # Right Ankle
    keypoints_cgtgait[..., 13, :] = keypoints_h36m[..., 4, :]   # Left Hip
    keypoints_cgtgait[..., 14, :] = keypoints_h36m[..., 5, :]   # Left Knee
    keypoints_cgtgait[..., 15, :] = keypoints_h36m[..., 6, :]   # Left Ankle

    return keypoints_cgtgait


def flip_data(data, left_joints=[7, 8, 9, 13, 14, 15], right_joints=[4, 5, 6, 10, 11, 12]):
    """Flip data for CGTGait order (16 joints)"""
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]
    return flipped_data


@torch.no_grad()
def get_pose3D(video_path, output_dir, generate_images=True):
    # Create args namespace without parsing command-line (avoid conflicts with main parser)
    args = argparse.Namespace(
        n_layers=16, dim_in=3, dim_feat=128, dim_rep=512, dim_out=3,
        mlp_ratio=4, act_layer=nn.GELU,
        attn_drop=0.0, drop=0.0, drop_path=0.0,
        use_layer_scale=True, layer_scale_init_value=0.00001, use_adaptive_fusion=True,
        num_heads=8, qkv_bias=False, qkv_scale=None,
        hierarchical=False,
        use_temporal_similarity=True, neighbour_num=2, temporal_connection_len=1,
        use_tcn=False, graph_only=False,
        n_frames=243
    )
    args = vars(args)

    ## Reload model
    model = MotionAGFormer(**args) 
    model_path = sorted(glob.glob(os.path.join('checkpoint', 'motionagformer-b-h36m.pth.tr')))[0]
    pre_dict = torch.load(model_path, weights_only=False, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in pre_dict['model'].items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    ## Load input
    keypoints_csv = output_dir + 'input_2D/keypoints.csv'
    df = pd.read_csv(keypoints_csv)

    # Reconstruct keypoints array from CSV
    num_frames = df['frame'].max() + 1
    num_joints = df['joint'].max() + 1
    keypoints = np.zeros((1, num_frames, num_joints, 3))

    for _, row in df.iterrows():
        frame_idx = int(row['frame'])
        joint_idx = int(row['joint'])
        keypoints[0, frame_idx, joint_idx, 0] = row['x']
        keypoints[0, frame_idx, joint_idx, 1] = row['y']
        keypoints[0, frame_idx, joint_idx, 2] = row['confidence']

    clips, downsample = turn_into_clips(keypoints)

    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_size = None

    if generate_images:
        # -------------------------------------------------------
        # 2. Generate per-frame 2D pose images (pose2D/)
        # -------------------------------------------------------
        print('\nGenerating 2D pose images...')
        output_dir_2D = output_dir + 'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)

        for i in tqdm(range(video_length)):
            ret, img = cap.read()
            if img is None:
                continue
            img_size = img.shape

            input_2D = keypoints[0][i]
            image = show2Dpose(input_2D, copy.deepcopy(img))
            cv2.imwrite(output_dir_2D + str(('%04d' % i)) + '_2D.png', image)

        cap.release()
    else:
        # Just get image size from first frame
        ret, img = cap.read()
        if img is not None:
            img_size = img.shape
        cap.release()

    if img_size is None:
        raise ValueError("Could not read video frame to get dimensions")

    all_3d_keypoints = []

    # -------------------------------------------------------
    # 3. Generate 3D pose predictions and optionally images
    # -------------------------------------------------------
    print('\nGenerating 3D pose...')
    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
        input_2D_aug = flip_data(input_2D)
        input_2D = torch.from_numpy(input_2D.astype('float32'))
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32'))

        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()

        # Convert from H36M to CGTGait order
        post_out_all_cgtgait = h36m_to_cgtgait_order(post_out_all)

        for j, post_out in enumerate(post_out_all_cgtgait):
            all_3d_keypoints.append(post_out.copy())

            if generate_images:
                rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
                rot = np.array(rot, dtype='float32')
                post_out_world = camera_to_world(post_out, R=rot, t=0)
                post_out_world[:, 2] -= np.min(post_out_world[:, 2])
                max_value = np.max(post_out_world)
                post_out_world /= max_value

                fig = plt.figure(figsize=(9.6, 5.4))
                gs = gridspec.GridSpec(1, 1)
                gs.update(wspace=-0.00, hspace=0.05)
                ax = plt.subplot(gs[0], projection='3d')
                show3Dpose(post_out_world, ax)

                output_dir_3D = output_dir + 'pose3D/'
                os.makedirs(output_dir_3D, exist_ok=True)
                plt.savefig(output_dir_3D + str(('%04d' % (idx * 243 + j))) + '_3D.png',
                            dpi=200, format='png', bbox_inches='tight')
                plt.close(fig)

    # Save 3D keypoints as CSV in CGTGait order
    output_3d_csv_dir = output_dir + 'output_3D/'
    os.makedirs(output_3d_csv_dir, exist_ok=True)
    output_3d_csv = output_3d_csv_dir + 'keypoints_cgtgait_order.csv'

    data_list = []
    joint_names = ['Pelvis', 'Spine', 'Neck', 'Head',
                   'RShoulder', 'RElbow', 'RWrist',
                   'LShoulder', 'LElbow', 'LWrist',
                   'RHip', 'RKnee', 'RAnkle',
                   'LHip', 'LKnee', 'LAnkle']

    for frame_idx, keypoints_3d in enumerate(all_3d_keypoints):
        for joint_idx in range(keypoints_3d.shape[0]):
            row = [frame_idx, joint_idx, joint_names[joint_idx]] + keypoints_3d[joint_idx, :].tolist()
            data_list.append(row)

    df_out = pd.DataFrame(data_list, columns=['frame', 'joint_id', 'joint_name', 'x', 'y', 'z'])
    df_out.to_csv(output_3d_csv, index=False)
    print(f'Saved 3D keypoints (CGTGait order) to {output_3d_csv}')
    print('Generating 3D pose successful!')

    if generate_images:
        # -------------------------------------------------------
        # 4. Generate combined side-by-side images (pose/)
        # -------------------------------------------------------
        image_2d_dir = sorted(glob.glob(os.path.join(output_dir + 'pose2D/', '*.png')))
        image_3d_dir = sorted(glob.glob(os.path.join(output_dir + 'pose3D/', '*.png')))

        output_dir_pose = output_dir + 'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)

        print('\nGenerating combined pose images...')
        for i in tqdm(range(len(image_2d_dir))):
            image_2d = plt.imread(image_2d_dir[i])
            image_3d = plt.imread(image_3d_dir[i])

            edge = 130
            image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

            font_size = 12
            fig = plt.figure(figsize=(15.0, 5.4))
            ax = plt.subplot(121)
            showimage(ax, image_2d)
            ax.set_title("Input", fontsize=font_size)

            ax = plt.subplot(122)
            showimage(ax, image_3d)
            ax.set_title("Reconstruction", fontsize=font_size)

            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose.png',
                        dpi=200, bbox_inches='tight')
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--generate-images', action='store_true', default=False, 
                        help='Generate visualization images (2D, 3D, combined poses and MP4 video)')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'

    get_pose2D(video_path, output_dir)
    get_pose3D(video_path, output_dir, generate_images=args.generate_images)

    if args.generate_images:
        # 5. Stitch combined images into final MP4 video
        img2video(video_path, output_dir, video_name)

    print('Processing complete!')
