import argparse
import os.path as osp
import os
import pickle
import sys

sys.path.append("3DGazeNet/")
from models import GazeModel
from handler import *
from video_decoder import VideoDecoder


def read_video_paths(file_path, base_input_dir, base_output_dir):
    with open(file_path, 'r') as f:
        video_paths = f.read().splitlines()
    return [os.path.join(base_input_dir, vp+".mp4") for vp in video_paths], [os.path.join(base_output_dir, vp) for vp in video_paths]

def headgaze_pkl(
    headgazes: list,
    times: list,
    args,
) -> dict:
    """Converts outputs from 6DRepNet for head pose estimation to a pkl

    Args:
        headgazes (list<dict>): list (length n) containing the id and headgaze (left and right eye)
        times (lismmocr): time values for each headgaze output (length n)
        args (Namespace): arguments the script has been executed with

    Returns:
        dict: dictionary ready to write in a .pkl
            y (list<dict>): list of dicts containing the headpose outputs per frame
                id (dict): face id according to the face_analysis.pkl input file
                headgaze (list<dict>): list (length n) of head gazes including
                    left_gaze_rad (list<float>): left eye gaze in radians [x,y]
                    right_gaze_rad (list<float>): right eye gaze in radians [x,y]
                    left_gaze_deg (list<float>): left eye gaze in degrees [x,y]
                    right_gaze_deg (list<float>): right eye gaze in degrees [x,y]
            time (list): t time values (length t)
            args (dict): arguments the script has been executed with
    """
    args_dict = vars(args)

    if "videos" in args_dict:
        del args_dict["videos"]

    return {
        "y": headgazes,
        "time": times,
        "args": args_dict,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="")

    # Required arguments
    parser.add_argument("-f", "--file", type=str, required=True, help="Text file containing paths to videos as <media>/<video_name>")
    parser.add_argument("-i", "--inp_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/videos", help="Base directory for input videos")
    parser.add_argument("-o", "--out_dir", type=str, default="/nfs/data/fakenarratives/202306_corpus/results_pkl", help="Base directory for output results")
    parser.add_argument(
        "--ckpt_path", type=str, default="3DGazeNet/assets/latest_a.ckpt", help="path to checkpoint file",
    )
    parser.add_argument("--cpu", action="store_true", help="process on cpu")
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1920,
        help="max dimension of the video frames",
    )
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.INFO
    if args.debug:
        level = logging.DEBUG

    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    # Create model and load fixed model parameters
    handler = GazeHandler(args.ckpt_path)

    # loop trough input videos
    ## File has lines with video names such as "Tagesschau/TV-20220101-2019-5100.webl.h264"
    input_paths, output_paths = read_video_paths(args.file, args.inp_dir, args.out_dir)

    for i, video_path in enumerate(input_paths):
        logging.info(f"Processing video {i+1}/{len(input_paths)}: {video_path}")

        output_path = output_paths[i]
        vidname = output_path.split("/")[-1]

        # setup output dir
        if not os.path.isfile(os.path.join(output_path, "face_analysis.pkl")):
            logging.error(
                f"Missing file: {os.path.join(output_path, 'face_analysis.pkl')}"
            )
            continue

        # read faceanalyis.pkl and store face bboxes
        with open(os.path.join(output_path, "face_analysis.pkl"), "rb") as pklfile:
            content = pickle.load(pklfile)

        fps = content["plugins"][0]["parameters"]["fps"]
        faces = content["faces"]

        faces_dict = {}
        for face in faces:
            if face["time"] not in faces_dict:
                faces_dict[face["time"]] = []

            faces_dict[face["time"]].append(face)

        # get frames from video
        vd = VideoDecoder(path=video_path, max_dimension=args.max_dimension, fps=fps)

        times = []
        headgazes = []
        for frame in vd:
            time = frame["time"]
            image = frame["frame"]

            faces_times = np.asarray(list(faces_dict.keys()))
            delta_time = np.abs(time - faces_times)
            closest_face_time = faces_times[np.argmin(delta_time)]
            eps = 1 / fps

            # check if frame contains faces
            if np.abs(closest_face_time - time) > eps / 2:
                continue

            logging.debug(f"video time: {time}, face time: {closest_face_time}")
            if closest_face_time != time:
                logging.warning(f"time difference: {np.abs(closest_face_time - time)}")
            logging.debug(f"{vidname}: Processing frame {frame['index']}")

            frame_faces = faces_dict[closest_face_time]
            ## Rescale normalized bounding boxes, kps to the original image size
            for face in frame_faces:
                face["bbox"]["w"] = face["bbox"]["w"] * image.shape[1]
                face["bbox"]["h"] = face["bbox"]["h"] * image.shape[0]
                face["bbox"]["x"] = face["bbox"]["x"] * image.shape[1]
                face["bbox"]["y"] = face["bbox"]["y"] * image.shape[0]
                face["kps"][:, 0] = face["kps"][:, 0] * image.shape[1] 
                face["kps"][:, 1] = face["kps"][:, 1] * image.shape[0]
            
            if len(frame_faces) > 0:
                results = handler.get(frame_faces, image)
                
                for face, result in zip(frame_faces, results):
                
                    gaze_pred_l_rad, gaze_pred_r_rad = handler.get_gaze(eye_kps=result[2])   ## In radians

                    ## Convert to degrees
                    gaze_pred_l_deg, gaze_pred_r_deg = np.degrees(gaze_pred_l_rad), np.degrees(gaze_pred_r_rad)
                    
                    # print(f"Left eye gaze: {gaze_pred_l_deg}, Right eye gaze: {gaze_pred_r_deg}")

                    headgazes.append(
                        {
                            "id": face["id"],
                            "headgaze": {
                                "left_gaze_rad": gaze_pred_l_rad.tolist(),
                                "right_gaze_rad": gaze_pred_r_rad.tolist(),
                                "left_gaze_deg": gaze_pred_l_deg.tolist(),
                                "right_gaze_deg": gaze_pred_r_deg.tolist(),
                            },
                        }
                    )
                    times.append(closest_face_time)
                    
                
                # # only for debugging
                # eimg = handler.draw_on(image, results)
                # oimg = np.concatenate((image, eimg), axis=1)
                # oimg = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
                # cv2.imwrite("3DGazeNet/output/test.jpg", oimg)

        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "headgaze_3DGazeNet.pkl"), "wb") as f:
            output_dict = headgaze_pkl(headgazes, times, args)
            pickle.dump(output_dict, f)

        logging.info(
            f"Output written to: {os.path.join(output_path, 'headgaze_3DGazeNet.pkl')}"
        )

        logging.debug(len(times), len(faces))

    return 0


if __name__ == "__main__":
    sys.exit(main())
