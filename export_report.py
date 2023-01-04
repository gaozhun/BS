import json
import os

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    args = parser.parse_args()
    all_best_total_score = []
    result="obj_name,eval_AUROC-image,eval_AUROC-pixel,eval_AP-image,eval_AP-pixel,best_epoch,\n"
    for obj_name in os.listdir(args.checkpoint_path):
        obj_path = os.path.join(args.checkpoint_path, obj_name)
        ob_name = obj_name.split("_")[-2]
        result+=f"{ob_name},"
        with open(os.path.join(obj_path, "best_total_socre.json"), "r") as f:
            best_total_score = json.load(f)
            for score in best_total_score.values():
                result+=f"{score:.4f},"
            result+="\n"
    print(result)
    with open("result.csv", "w") as f:
        f.write(result)