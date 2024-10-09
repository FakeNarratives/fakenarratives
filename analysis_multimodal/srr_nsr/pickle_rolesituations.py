import os
import pickle
import random
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_pickle(path):
    with open(path, 'rb') as pkl:
        return pickle.load(pkl)

def ensemble_predict(seg_models, sw_models, X_seg, X_sw, confidence_threshold=0.6):
    seg_predictions = np.array([model.predict(X_seg) for model in seg_models])
    sw_predictions = np.array([model.predict(X_sw) for model in sw_models])
    all_predictions = np.concatenate([seg_predictions, sw_predictions], axis=0)
    
    n_samples = all_predictions.shape[1]
    final_predictions = []
    
    for i in range(n_samples):
        votes = Counter(all_predictions[:, i])
        label, count = votes.most_common(1)[0]
        confidence = count / (len(seg_models) + len(sw_models))
        final_predictions.append(label if confidence >= confidence_threshold else -1)

    return np.array(final_predictions)

def load_models(model_dir, model_type, task, hierarchy_level=None):
    seg_models, sw_models = [], []
    for aggregation in ['segmentbased', 'windowbased']:
        if task == "srr":
            model_name = os.path.join(model_dir, f"{model_type}_{aggregation}_l{hierarchy_level}.pkl")
        else:
            model_name = os.path.join(model_dir, f"{model_type}_{aggregation}.pkl")

        if not os.path.exists(model_name):
            print(f"\tModel not found: {model_name}, skipping...")
            continue

        with open(model_name, 'rb') as file:
            model_dict = pickle.load(file)
        
        if aggregation == 'segmentbased':
            seg_models.extend(model_dict["models"])
        else:
            sw_models.extend(model_dict["models"])
    
    return seg_models, sw_models

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Classify speaker roles and situations in videos')
    parser.add_argument("-v", "--videos", nargs="+", type=str, required=True, help="path to the video files")
    parser.add_argument("-o", '--pkl_dir', type=str, required=True, help='path to the output folder')
    parser.add_argument("-m", '--model_dir', type=str, required=True, help='path to the model folder')
    parser.add_argument('--rf', action='store_true', help="Random Forest")
    parser.add_argument('--xgb', action='store_true', help="XGBoost")
    parser.add_argument('--gb', action='store_true', help="XGBoost")
    parser.add_argument('--srr', action='store_true', help="Speaker Role Recognition")
    parser.add_argument('--nsr', action='store_true', help="News Situation Recognition")
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    task = "srr" if args.srr else "nsr"
    model_type = "rf" if args.rf else "xgb" if args.xgb else "gb"

    if task == "srr":
        label_maps = [
            {0: "anchor", 1: "reporter", 2: "other"},
            {0: "anchor", 1: "reporter", 2: "other", 3: "expert", 4: "layperson", 5: "politician"}
        ]
    else:
        label_maps = [{0: "talking-head", 1: "voiceover", 2: "interview", 3: "commenting", 4: "speech"}]

    model_dir = f"{args.model_dir}/{task}"

    for vi, video_path in enumerate(args.videos):
        print(f"Processing video {vi+1}/{len(args.videos)}: {video_path}", flush=True)

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        turn_features = load_pickle(f"{args.pkl_dir}/{video_name}/srr_nsr_features.pkl")["speaker_turns"]
        
        results = []
        predictions = {}
        
        for hierarchy_level in range(len(label_maps)):
            seg_models, sw_models = load_models(model_dir, model_type, task, hierarchy_level)
            
            if not seg_models or not sw_models:
                print(f"Missing models for {'hierarchy level ' + str(hierarchy_level) if task == 'srr' else 'NSR'}")
                continue

            X_seg = np.array([turn["seg_feature"] for turn in turn_features])
            X_sw = np.array([turn["sw_feature"] for turn in turn_features])

            predictions[hierarchy_level] = ensemble_predict(seg_models, sw_models, X_seg, X_sw, 0.7)

            # print(f"Predictions for {'hierarchy level ' + str(hierarchy_level) if task == 'srr' else 'NSR'}:")
            # print(predictions[hierarchy_level])

        for i, turn in enumerate(turn_features):
            result = {
                "id": turn["id"],
                "start_time": turn["start_time"],
                "end_time": turn["end_time"]
            }
            if task == "srr":
                for level in range(len(label_maps)):
                    result[f"role_l{level}"] = label_maps[level].get(predictions[level][i], "unsure")
            else:
                result["situation"] = label_maps[0].get(predictions[0][i], "unsure")
            results.append(result)


        name_task = "speaker_roles" if task == "srr" else "situations"
        output_file = f"{args.pkl_dir}/{video_name}/icmr_{name_task}.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"Saved predictions to {vi+1}/{len(args.videos)}: {output_file}")
        print()

if __name__ == "__main__":
    main()