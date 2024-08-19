import sys, time, os, tqdm, torch, argparse, glob, subprocess, warnings, cv2, pickle, numpy, math, python_speech_features, logging

from scipy.io import wavfile
import numpy as np
from video_utils import read_video_and_get_info

sys.path.append("Light-ASD")
from ASD import ASD

warnings.filterwarnings("ignore")


def extract_MFCC(file, outPath):
    # CPU: extract mfcc
    sr, audio = wavfile.read(file)
    mfcc = python_speech_features.mfcc(audio, sr)  # (N_frames, 13)   [1s = 100 frames]
    featuresPath = os.path.join(outPath, file.split("/")[-1].replace(".wav", ".npy"))
    numpy.save(featuresPath, mfcc)


def evaluate_network(files, args, fps):
    # GPU: active speaker detection by pretrained model
    s = ASD()
    s.loadParameters(args.pretrainModel)
    sys.stderr.write("Model %s loaded from previous state! \r\n" % args.pretrainModel)
    s.eval()
    allScores = []
    # durationSet = {1,2,4,6} # To make the result more reliable
    durationSet = {
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        3,
        4,
        5,
        6,
    }  # Use this line can get more reliable result
    for file in tqdm.tqdm(files, total=len(files)):
        fileName = os.path.splitext(file.split("/")[-1])[0]  # Load audio and video
        _, audio = wavfile.read(os.path.join(args.pycropPath, fileName + ".wav"))
        if len(audio) == 0:  ## Added by me
            continue
        audioFeature = python_speech_features.mfcc(
            audio, 16000, numcep=13, winlen=0.025, winstep=0.010
        )
        video = cv2.VideoCapture(os.path.join(args.pycropPath, fileName + ".avi"))
        videoFeature = []
        while video.isOpened():
            ret, frames = video.read()
            if ret == True:
                face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = face[
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                    int(112 - (112 / 2)) : int(112 + (112 / 2)),
                ]
                videoFeature.append(face)
            else:
                break
        video.release()
        videoFeature = numpy.array(videoFeature)
        length = min(
            (audioFeature.shape[0] - audioFeature.shape[0] % 4) / 100,
            videoFeature.shape[0],
        )

       # Check if either audioFeature or videoFeature is empty
        if audioFeature.size == 0 or videoFeature.size == 0:
            print(f"Skipping file {fileName} due to empty audio or video feature")
            continue
        
        audioFeature = audioFeature[: int(round(length * 100)), :]
        videoFeature = videoFeature[: int(round(length * fps)), :, :]

        allScore = []  # Evaluation use model
        for duration in durationSet:
            batchSize = int(math.ceil(length / duration))
            scores = []
            with torch.no_grad():
                for i in range(batchSize):
                    inputA = (
                        torch.FloatTensor(
                            audioFeature[
                                i * duration * 100 : (i + 1) * duration * 100, :
                            ]
                        )
                        .unsqueeze(0)
                        .cuda()
                    )
                    inputV = (
                        torch.FloatTensor(
                            videoFeature[
                                i * duration * fps : (i + 1) * duration * fps, :, :
                            ]
                        )
                        .unsqueeze(0)
                        .cuda()
                    )


                    ## --- Conditions for either audio or visual empty batches
                    # If audio is empty but video is not, fill audio with zeros
                    if inputA.size(1) == 0 and inputV.size(1) != 0:
                        inputA = torch.zeros((1, inputV.size(1) * 4, inputA.size(2))).cuda()
                    
                    # If video is empty but audio is not, fill video with zeros
                    elif inputV.size(1) == 0 and inputA.size(1) != 0:
                        inputV = torch.zeros((1, inputA.size(1) // 4, 112, 112)).cuda()
                    
                    # If both are empty, skip this batch
                    elif inputA.size(1) == 0 and inputV.size(1) == 0:
                        print(f"Skipping batch {i} due to both inputs being empty")
                        continue
                    

                    embedA = s.model.forward_audio_frontend(inputA)
                    embedV = s.model.forward_visual_frontend(inputV)
                    
                    # Get the maximum length between audio and video embeddings
                    max_length = max(embedA.size(1), embedV.size(1))

                    # Pad audio embedding if necessary
                    if embedA.size(1) < max_length:
                        padding = torch.zeros(embedA.size(0), max_length - embedA.size(1), embedA.size(2)).cuda()
                        embedA = torch.cat([embedA, padding], dim=1)

                    # Pad video embedding if necessary
                    if embedV.size(1) < max_length:
                        padding = torch.zeros(embedV.size(0), max_length - embedV.size(1), embedV.size(2)).cuda()
                        embedV = torch.cat([embedV, padding], dim=1)

                    out = s.model.forward_audio_visual_backend(embedA, embedV)
                    score = s.lossAV.forward(out, labels=None)
                    scores.extend(score)
            allScore.append(scores)
        # Handle variable-length scores
        max_length = max(len(scores) for scores in allScore)
        padded_scores = [scores + [-5] * (max_length - len(scores)) for scores in allScore]
        # allScore = numpy.round((numpy.mean(numpy.array(allScore), axis=0)), 1).astype(float)
        allScore = numpy.round((numpy.mean(numpy.array(padded_scores), axis=0)), 1).astype(float)
        allScores.append(allScore)

    return allScores


def visualization(tracks, scores, args, vr, fps):
    # CPU: visulize the result for video format
    faces = [[] for i in range(len(vr))]
    for tidx, track in enumerate(tracks):
        if tidx < len(scores):  ## Added by me
            score = scores[tidx]
            for fidx, frame in enumerate(track["track"]["frame"].tolist()):
                s = score[
                    max(fidx - 2, 0) : min(fidx + 3, len(score) - 1)
                ]  # average smoothing
                s = numpy.mean(s)
                faces[frame].append(
                    {
                        "track": tidx,
                        "score": float(s),
                        "s": track["proc_track"]["s"][fidx],
                        "x": track["proc_track"]["x"][fidx],
                        "y": track["proc_track"]["y"][fidx],
                    }
                )
    firstImage = vr[0]
    fw = firstImage.shape[1]
    fh = firstImage.shape[0]
    vOut = cv2.VideoWriter(
        os.path.join(args.output_dir, "video_only.avi"),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (fw, fh),
    )
    colorDict = {0: 0, 1: 255}
    for fidx, image in tqdm.tqdm(enumerate(vr), total=len(vr)):
        image = image
        for face in faces[fidx]:
            clr = colorDict[int((face["score"] >= 0))]
            txt = round(face["score"], 1)
            cv2.rectangle(
                image,
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                (int(face["x"] + face["s"]), int(face["y"] + face["s"])),
                (0, clr, 255 - clr),
                10,
            )
            cv2.putText(
                image,
                "%s" % (txt),
                (int(face["x"] - face["s"]), int(face["y"] - face["s"])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, clr, 255 - clr),
                5,
            )
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vOut.write(image_rgb)
    vOut.release()
    command = (
        "ffmpeg -y -i %s -i %s -threads %d -c:v copy -c:a copy %s -loglevel panic"
        % (
            os.path.join(args.output_dir, "video_only.avi"),
            os.path.join(args.output_dir, "audio.wav"),
            args.nDataLoaderThread,
            os.path.join(args.output_dir, "video_out_%s.avi"%(args.model)),
        )
    )
    output = subprocess.call(command, shell=True, stdout=None)


def combine_tracks_and_scores(tracks, scores):
    assert len(tracks) == len(scores), "Number of tracks and scores must match"
    combined_results = []
    for track, score in zip(tracks, scores):
        smoothed_scores = []
        for i in range(len(score)):
            s = score[max(i-2, 0):min(i+3, len(score))]  # average smoothing
            smoothed_scores.append(float(np.mean(s)))
        
        speaking_frames = sum(s >= 0 for s in smoothed_scores)
        speaking_ratio = speaking_frames / len(smoothed_scores)

        combined_track = {
            "track_id": track['track']['track_id'],
            "frames": track['track']['frame'].tolist(),
            "bbox": track['track']['bbox'].tolist(),
            "is_speaking": speaking_ratio > 0.6,  # Threshold can be adjusted when directly using face_analysis.pkl
            "speaking_ratio": speaking_ratio,
            "speaking_frames": speaking_frames,
            "mean_score": float(np.mean(smoothed_scores)),
            "original_scores": score,
            "smoothed_scores": smoothed_scores
        }

        combined_results.append(combined_track)
        
    return combined_results


def parse_args():
    parser = argparse.ArgumentParser(description="Light-ASD For Active Speaker Detection")
    parser.add_argument("--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("--pkl_dir", type=str, required=True, help="path to the output folder")
    parser.add_argument(
        "--model",
        type=str,
        default="talkset",
        help="talkset or ava",
    )
    parser.add_argument(
        "--nDataLoaderThread", type=int, default=10, help="Number of workers"
    )
    parser.add_argument(
        "--minTrack", type=int, default=10, help="Number of min frames for each shot"
    )
    parser.add_argument("--start", type=int, default=0, help="The start time of the video")
    parser.add_argument(
        "--duration",
        type=int,
        default=0,
        help="The duration of the video, when set as 0, will extract the whole video",
    )
    parser.add_argument("--debug", action="store_true", help="debug output")
    parser.add_argument("--visualize", action="store_true", help="Visualize the result")
    parser.add_argument(
        "--max_dimension",
        type=int,
        required=False,
        default=1280,
        help="max dimension of the video frames",
    )

    return parser.parse_args()


# Main function
def main():

    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=level)

    args.pretrainModel = "Light-ASD/weight/finetuning_TalkSet.model" if args.model == "talkset" else "Light-ASD/weight/pretrain_AVA_CVPR.model"

    videos = args.videos
    for vi, video_path in enumerate(videos):
        logging.info(f"\tProcessing video [{vi+1}/{len(videos)}]: {video_path}")

        vidname = os.path.splitext(os.path.basename(video_path))[0]
        args.videoFilePath = video_path
        args.output_dir = os.path.join(args.pkl_dir, vidname)
        args.audioFilePath = os.path.join(args.output_dir, 'audio.wav')
        args.pycropPath = os.path.join(args.output_dir, 'facecrops')


        face_detection_path = os.path.join(args.output_dir, "face_detection_insightface.pkl")

        if not os.path.isfile(face_detection_path):
            logging.error(f"\tMissing face detection file in {face_detection_path}")
            continue

        with open(face_detection_path, "rb") as pklfile:
            face_content = pickle.load(pklfile)

        # get frames from video
        vd, frame_width, frame_height, fps, real_fps = read_video_and_get_info(video_path, args, face_content["args"]["fps"])
        logging.info(f"\tVideo info: {len(vd)} frames, New FPS {fps}, Original FPS {real_fps}, Size: {frame_width} x {frame_height}")
        assert frame_width == face_content["args"]["frame_width"]
        assert frame_height == face_content["args"]["frame_height"]

        vidTracks = pickle.load(open(os.path.join(args.output_dir, 'tracks.pkl'), 'rb'))

        # Active Speaker Detection
        savePath = os.path.join(args.output_dir, "scores.pkl")
        files = glob.glob("%s/*.avi" % args.pycropPath)
        files.sort()
        scores = evaluate_network(files, args, int(fps))

        with open(savePath, "wb") as fil:
            pickle.dump(scores, fil)
        logging.info("\t"+time.strftime("%Y-%m-%d %H:%M:%S") + " Scores extracted and saved in %s \r" % savePath)

        # Combine tracks and scores
        combined_results = combine_tracks_and_scores(vidTracks, scores)

        # Save combined results
        output_path = os.path.join(args.output_dir, 'asd_light-asd.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump({"tracks": combined_results, "args": args}, f)
        logging.info(f"\tASD results saved to {output_path}")

        # Visualization, save the result as the new video
        if args.visualize:
            visualization(vidTracks, scores, args, vd, int(fps))
        
        print()
        

if __name__ == "__main__":
    main()
