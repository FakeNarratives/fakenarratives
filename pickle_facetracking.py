import argparse
import logging
import numpy as np
import os
import uuid
import pickle
from scipy.interpolate import interp1d
import sys
import cv2
from collections import defaultdict
from tqdm import tqdm
from scipy import signal
import subprocess
from video_utils import read_video_and_get_info


def tracking_pkl(tracks: list, args) -> dict:
    """Converts tracking results to a pkl format"""
    args_dict = vars(args)
    if "videos" in args_dict:
        del args_dict["videos"]
    return {
        "y": tracks,
        "args": args_dict,
    }

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def normalize_to_pixel(bbox, frame_width, frame_height):
    x, y, w, h = bbox['x'], bbox['y'], bbox['w'], bbox['h']
    x1 = x * frame_width
    y1 = y * frame_height
    x2 = (x + w) * frame_width
    y2 = (y + h) * frame_height
    return [x1, y1, x2, y2]


def track_shot(args, shotFaces):
    iouThres = 0.5
    tracks = []
    while True:
        track = []
        enhanced_track = {
            'frame': [],
            'bbox': [],
            'frames': []
        }
        for frameFaces in shotFaces:
            for face in frameFaces:
                if not track:
                    track.append(face)
                    frameFaces.remove(face)
                    enhanced_track['frame'].append(face['frame'])
                    enhanced_track['bbox'].append(face['bbox'])
                    enhanced_track['frames'].append({
                        'frame_number': face['frame'],
                        'bbox': face['bbox'],
                        'det_score': face['conf'],
                        'embedding': face['embedding'],
                        'face_id': face['id']  # Use the original face ID
                    })
                elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
                    iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
                    if iou > iouThres:
                        track.append(face)
                        frameFaces.remove(face)
                        enhanced_track['frame'].append(face['frame'])
                        enhanced_track['bbox'].append(face['bbox'])
                        enhanced_track['frames'].append({
                            'frame_number': face['frame'],
                            'bbox': face['bbox'],
                            'det_score': face['conf'],
                            'embedding': face['embedding'],
                            'face_id': face['id']  # Use the original face ID
                        })
                        continue
                else:
                    break
        if not track:
            break
        elif len(track) > args.minTrack:
            frameNum = np.array(enhanced_track['frame'])
            bboxes = np.array(enhanced_track['bbox'])
            frameI = np.arange(frameNum[0], frameNum[-1] + 1)
            bboxesI = []
            for ij in range(0, 4):
                interpfn = interp1d(frameNum, bboxes[:, ij])
                bboxesI.append(interpfn(frameI))
            bboxesI = np.stack(bboxesI, axis=1)
            if max(np.mean(bboxesI[:, 2] - bboxesI[:, 0]), np.mean(bboxesI[:, 3] - bboxesI[:, 1])) > args.minFaceSize:
                track_id = str(uuid.uuid4())
                enhanced_track['track_id'] = track_id
                enhanced_track['frame'] = frameI
                enhanced_track['bbox'] = bboxesI
                
                tracks.append(enhanced_track)
    return tracks


def crop_video(args, track, vr, fps, cropFile, audioFilePath):
    vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (224, 224))
    
    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)
    
    dets['s'] = np.array(signal.medfilt(dets['s'], kernel_size=13))
    dets['x'] = np.array(signal.medfilt(dets['x'], kernel_size=13))
    dets['y'] = np.array(signal.medfilt(dets['y'], kernel_size=13))
    
    cs = args.cropScale
    frame_nums = np.array(track['frame'])
    
    for fidx, frame_num in enumerate(frame_nums):
        image = cv2.cvtColor(vr[frame_num].asnumpy(), cv2.COLOR_RGB2BGR)
        
        bs = dets['s'][fidx]
        my = dets['y'][fidx]
        mx = dets['x'][fidx]
        
        y1 = int(my - bs)
        y2 = int(my + bs * (1 + 2 * cs))
        x1 = int(mx - bs * (1 + cs))
        x2 = int(mx + bs * (1 + cs))
        
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - image.shape[0])
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - image.shape[1])
        
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[110, 110, 110])
        
        crop_y1 = max(0, y1 + pad_top)
        crop_y2 = min(image.shape[0], y2 + pad_top)
        crop_x1 = max(0, x1 + pad_left)
        crop_x2 = min(image.shape[1], x2 + pad_left)
        
        face = image[crop_y1:crop_y2, crop_x1:crop_x2]
        face_resized = cv2.resize(face, (224, 224))
        
        vOut.write(face_resized)
    
    vOut.release()
    
    subaudioFilePath = cropFile + '.wav'
    audioStart = frame_nums[0] / fps
    audioEnd = (frame_nums[-1] + 1) / fps
    
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" %
               (audioFilePath, 8, audioStart, audioEnd, subaudioFilePath))
    ret = subprocess.call(command, shell=True)

    command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" %
               (cropFile, subaudioFilePath, 8, cropFile))
    subprocess.call(command, shell=True)
    
    os.remove(cropFile + 't.avi')
    
    return {'track': track, 'proc_track': dets, 'video_file': cropFile + '.avi', 'audio_file': subaudioFilePath}


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument('--minTrack', type=int, default=25, help='Number of min frames for each shot')
    parser.add_argument('--numFailedDet', type=int, default=10, help='Number of missed detections allowed before tracking is stopped')
    parser.add_argument('--minFaceSize', type=int, default=1, help='Minimum face size in pixels')
    parser.add_argument('--cropScale', type=float, default=0.40, help='Scale bounding box')
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1280,
        help="max dimension of the video frames",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")
        vidname = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(args.pkl_dir, vidname)
        pycropPath = os.path.join(output_dir, 'facecrops')
        os.makedirs(pycropPath, exist_ok=True)

        audioFilePath = os.path.join(output_dir, 'audio.wav')
        if not os.path.exists(audioFilePath):
            logging.error(f"\tError: Audio file {audioFilePath} does not exist.")
            continue

        face_detection_path = os.path.join(output_dir, "face_detection_insightface.pkl")
        shot_detection_path = os.path.join(output_dir, "transnet_shotdetection.pkl")

        if not os.path.isfile(face_detection_path) or not os.path.isfile(shot_detection_path):
            logging.error(f"\tMissing shot or face detection file in {output_dir}")
            continue

        with open(face_detection_path, "rb") as pklfile:
            face_content = pickle.load(pklfile)
        with open(shot_detection_path, "rb") as pklfile:
            shot_content = pickle.load(pklfile)

        # get frames from video
        vd, frame_width, frame_height, fps = read_video_and_get_info(video_path, args, face_content["args"]["fps"])
        logging.info(f"\tVideo info: {len(vd)} frames, {fps} FPS, {frame_width} x {frame_height}")

        faces_by_frame = defaultdict(list)
        for face in face_content["y"]:
            if face['det_score'] >= 0.6 and face["bbox"]["h"] > 0.05:
                frame_index = face["frame"]
                pixel_bbox = normalize_to_pixel(face['bbox'], frame_width, frame_height)
                faces_by_frame[frame_index].append({
                    'id': face['id'],
                    'frame': frame_index,
                    'bbox': pixel_bbox,
                    'conf': face['det_score'],
                    'embedding': face['embedding']
                })

        max_frame = max(faces_by_frame.keys())
        faces_list = [[] for _ in range(max_frame + 1)]
        for frame_index, face_list in faces_by_frame.items():
            faces_list[frame_index] = face_list

        allTracks = []
        for shot in shot_content["output_data"]["shots"]:
            if shot["end_frame"] - shot["start_frame"] >= args.minTrack:
                allTracks.extend(track_shot(args, faces_list[shot["start_frame"]:shot["end_frame"]]))
        logging.info(f"\tDetected {len(allTracks)} tracks")

        vidTracks = []
        for ii, track in tqdm(enumerate(allTracks), total=len(allTracks)):
            vidTracks.append(crop_video(args, track, vd, fps, os.path.join(pycropPath, '%05d' % ii), audioFilePath))
        
        tracks_path = os.path.join(output_dir, 'tracks.pkl')
        with open(tracks_path, 'wb') as fil:
            pickle.dump(vidTracks, fil)
        logging.info(f"\tFace Crop and saved in {pycropPath} tracks")

        face_tracking_data = {"tracks": allTracks, "args": vars(args)}
        tracking_output_path = os.path.join(output_dir, 'face_tracking.pkl')
        with open(tracking_output_path, 'wb') as f:
            pickle.dump(face_tracking_data, f)
        logging.info(f"\tFace tracking results saved to {tracking_output_path}")

        print()

    return 0

if __name__ == "__main__":
    sys.exit(main())