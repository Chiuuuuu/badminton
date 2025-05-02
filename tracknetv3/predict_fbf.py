import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torch.utils.data import DataLoader
from collections import deque

from test import predict_location, get_ensemble_weight, generate_inpaint_mask
from dataset import Shuttlecock_Trajectory_Dataset, Video_IterableDataset
from utils.general import write_pred_csv, draw_traj, get_model


def process_and_display_frame(frame, pred_dict, traj, x_pred, y_pred, vis_pred, traj_len=8):
    """ Process and display a single frame with predictions.

        Args:
            frame (numpy.ndarray): Current video frame.
            pred_dict (Dict): Dictionary to store predictions.
            traj (deque): Trajectory queue for drawing.
            x_pred (int): Predicted X coordinate.
            y_pred (int): Predicted Y coordinate.
            vis_pred (int): Visibility prediction (0 or 1).
            traj_len (int): Length of trajectory to draw.

        Returns:
            frame (numpy.ndarray): Processed frame with trajectory drawn.
    """
    if vis_pred:
        traj.appendleft([x_pred, y_pred])
    else:
        traj.appendleft(None)

    # Keep trajectory length within the limit
    if len(traj) > traj_len:
        traj.pop()

    # Draw trajectory on the frame
    frame = draw_traj(frame, traj, color='yellow')

    # Add prediction to the dictionary
    pred_dict['X'].append(x_pred)
    pred_dict['Y'].append(y_pred)
    pred_dict['Visibility'].append(vis_pred)

    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', type=str, help='file path of the video')
    parser.add_argument('--tracknet_file', type=str, help='file path of the TrackNet model checkpoint')
    parser.add_argument('--inpaintnet_file', type=str, default='', help='file path of the InpaintNet model checkpoint')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--eval_mode', type=str, default='weight', choices=['nonoverlap', 'average', 'weight'], help='evaluation mode')
    parser.add_argument('--max_sample_num', type=int, default=1800, help='maximum number of frames to sample for generating median image')
    parser.add_argument('--video_range', type=lambda splits: [int(s) for s in splits.split(',')], default=None, help='range of start second and end second of the video for generating median image')
    parser.add_argument('--save_dir', type=str, default='pred_result', help='directory to save the prediction result')
    parser.add_argument('--large_video', action='store_true', default=False, help='whether to process large video')
    parser.add_argument('--output_video', action='store_true', default=False, help='whether to output video with predicted trajectory')
    parser.add_argument('--traj_len', type=int, default=8, help='length of trajectory to draw on video')
    args = parser.parse_args()

    num_workers = args.batch_size if args.batch_size <= 16 else 16
    video_file = args.video_file
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    out_csv_file = os.path.join(args.save_dir, f'{video_name}_ball.csv')
    out_video_file = os.path.join(args.save_dir, f'{video_name}.mp4')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Load model
    tracknet_ckpt = torch.load(args.tracknet_file)
    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']
    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).cuda()
    tracknet.load_state_dict(tracknet_ckpt['model'])

    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_scaler, h_scaler = w / 512, h / 288
    img_scaler = (w_scaler, h_scaler)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_file, fourcc, fps, (w, h))

    # Initialize prediction dictionary and trajectory queue
    pred_dict = {'Frame': [], 'X': [], 'Y': [], 'Visibility': []}
    traj = deque(maxlen=args.traj_len)

    # Dataset and DataLoader
    if args.large_video:
        dataset = Video_IterableDataset(video_file, seq_len=tracknet_seq_len, sliding_step=1, bg_mode=bg_mode)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    else:
        frame_list = []
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame_list.append(frame)
        dataset = Shuttlecock_Trajectory_Dataset(frame_arr=np.array(frame_list), seq_len=tracknet_seq_len, bg_mode=bg_mode)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Predict frame by frame
    for step, (indices, frames) in enumerate(tqdm(data_loader)):
        frames = frames.float().cuda()
        with torch.no_grad():
            y_pred = tracknet(frames).detach().cpu()

        # Process each frame in the batch
        for i in range(indices.shape[0]):
            success, frame = cap.read() if args.large_video else (True, frame_list[indices[i][0][1]])
            if not success:
                break

            # Predict coordinates
            y_p = y_pred[i][0]
            bbox_pred = predict_location(y_p)
            x_pred, y_pred = int(bbox_pred[0] + bbox_pred[2] / 2), int(bbox_pred[1] + bbox_pred[3] / 2)
            x_pred, y_pred = int(x_pred * img_scaler[0]), int(y_pred * img_scaler[1])
            vis_pred = 0 if x_pred == 0 and y_pred == 0 else 1

            # Process and display the frame
            frame = process_and_display_frame(frame, pred_dict, traj, x_pred, y_pred, vis_pred, traj_len=args.traj_len)

            # Show the processed frame in real-time
            cv2.imshow('Prediction', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                exit()

            # Write the processed frame to the output video
            out.write(frame)

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Save predictions to CSV
    write_pred_csv(pred_dict, save_file=out_csv_file)

    print('Frame-by-frame video and CSV saved successfully.')