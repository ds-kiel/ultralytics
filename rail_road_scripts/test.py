from ultralytics import YOLO
import os
import wandb
from dotenv import load_dotenv
import os
from ultralytics.utils import SETTINGS
from ultralytics.utils.callbacks.wb import log_yaml
from pathlib import Path
import re
import yaml


from pathlib import Path

import random
import cv2

def log_sample_predictions_to_wandb(
    model,
    images_dirs,
    max_samples=10,
    conf=0.25
):
    sample_table = wandb.Table(
        columns=["image_path", "predicted_class", "prediction"]
    )

    # Collect all images
    all_images = []
    for img_dir, _ in images_dirs:
        all_images += list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))

    if not all_images:
        print("[WARN] No images found for prediction logging")
        return

    sampled_images = random.sample(
        all_images, min(max_samples, len(all_images))
    )

    for img_path in sampled_images:
        preds = model.predict(
            source=str(img_path),
            conf=conf,
            save=False,
            verbose=False
        )

        result = preds[0]

        # -------------------------------
        # Extract predicted class(es)
        # -------------------------------
        if result.boxes is None or len(result.boxes) == 0:
            pred_class = "background"
        else:
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = result.names
            unique_classes = sorted(set(class_names[c] for c in cls_ids))
            pred_class = ", ".join(unique_classes)

        # Render predictions
        rendered = result.plot()  # BGR
        rendered = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)

        sample_table.add_data(
            str(img_path),
            pred_class,
            wandb.Image(rendered)
        )

    wandb.log({"samples/predictions": sample_table})


def discover_image_label_pairs_from_yaml(data_yaml_path):
    """
    Returns:
    {
      "train": [(images_dir, labels_dir), ...],
      "val":   [...],
      "test":  [...]
    }
    """
    data_yaml_path = Path(data_yaml_path)

    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    root = Path(data.get("path", "."))
    pairs = {}

    for split in ["train", "val", "test"]:
        pairs[split] = []

        for img_path in normalize_to_list(data.get(split)):
            if not img_path:
                continue

            # Resolve path relative to YAML 'path'
            images_dir = (root / img_path).resolve()

            if not images_dir.exists():
                print(f"[WARN] Missing images dir: {images_dir}")
                continue

            # Infer labels directory
            if "images" in images_dir.parts:
                labels_dir = Path(
                    str(images_dir).replace("/images", "/labels")
                )
            else:
                labels_dir = images_dir.parent / "labels"

            pairs[split].append((images_dir, labels_dir))

    return pairs

def empty_image_stats(images_dir, labels_dir):
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    total_images = len(image_files)

    empty_images = []
    for img in image_files:
        label_file = labels_dir / f"{img.stem}.txt"

        if not label_file.exists():
            empty_images.append(img)
            continue

        content = label_file.read_text().strip()
        if not content or not re.search(r"\d", content):
            empty_images.append(img)

    return total_images, empty_images

def normalize_to_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def load_dataset_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


# Load environment variables from .env
load_dotenv()

wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key  # for wandb
print("WANDB_API_KEY loaded:", wandb_api_key)
#Initialize your Weights & Biases environment
wandb.login(key=wandb_api_key)
SETTINGS["wandb"] = True
run_name = "pre-trained_yolov8s_zollner_test_small_bbox"
run = wandb.init(
    project="zollner project",
    name=run_name,
    job_type="test",
)
run.tags = run.name.split("_")

# wandb.init(project="ultralytics", name="coco_8 forked_repo")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# pretrained YOLOV8s "/home/mal/ultralytics/rail_road_scripts/yolov8s.pt"
# TRAY dataset only "/home/mal/ultralytics/rail_road_scripts/zollner project/single_class_train/weights/best.pt"
# TRAY + COCO dataset "/home/mal/ultralytics/rail_road_scripts/BEST_coco_zollner_single_cls_train5.pt"
model = YOLO('/home/mal/ultralytics/rail_road_scripts/zollner project/single_class_train/weights/best.pt')


# ============================== UPDATE HERE ==============================


# data_yaml = '/home/mal/coco_projects/ultralytics/yolov8/ultralytics/ultralytics/cfg/datasets/zollner_train_single_cls.yaml'
data_yaml = '/home/mal/ultralytics/ultralytics/cfg/datasets/zollner_test.yaml'
results = model.val(
    data=data_yaml,
    split="test",
    cache=False,     
    imgsz=640,
    patience=300,
    project="zollner project",
    name=run_name,
    )

if results:
  log_yaml(data_yaml, f"{wandb.run.id}_data_yaml")

pairs = discover_image_label_pairs_from_yaml(data_yaml)

# Log prediction samples from TEST split
if "test" in pairs:
    log_sample_predictions_to_wandb(
        model,
        pairs["test"],
        max_samples=12,
        conf=0.25
    )

print("Discovered dataset splits and directories:", pairs)

empty_table = wandb.Table(columns=["split", "source", "image_path"])

for split, dir_pairs in pairs.items():
    split_total = 0
    split_empty = 0

    for images_dir, labels_dir in dir_pairs:
        total, empty_files = empty_image_stats(images_dir, labels_dir)

        split_total += total
        split_empty += len(empty_files)

        source = images_dir.name  # e.g. "test", "backgrounds"

        for img in empty_files:
            empty_table.add_data(
                split,
                source,
                str(img)
            )

    print(f"[{split}] total={split_total}, empty={split_empty}")

    wandb.run.summary[f"dataset/{split}_total_images"] = split_total
    wandb.run.summary[f"dataset/{split}_empty_images"] = split_empty
    wandb.run.summary[f"dataset/{split}_empty_ratio"] = split_empty / max(split_total, 1)

wandb.log({"dataset/empty_images": empty_table})


metrics = results.results_dict  # <-- key line
res_keys = results.keys
class_result = results.class_result
maps = results.maps
summary = results.summary

# print("res_keys:", res_keys)
# print("metrics:", metrics)
print("ap_class_index:", results.ap_class_index)
print("ap_class_index train:", results.ap_class_index[0])
P, R, AP50, AP5095 = results.class_result(0)

print("Train class (id=6):")
print("  Precision :", P)
print("  Recall    :", R)
print("  AP@50     :", AP50)
print("  AP@50-95  :", AP5095)

# AP_train = class_result[''].get('train', None)
# print("AP for 'train' class:", AP_train)
# print("maps:", maps)
# print("summary:", summary)


wandb.log({
    "test/mAP50": metrics["metrics/mAP50(B)"],
    "test/mAP50-95": metrics["metrics/mAP50-95(B)"],
    "test/precision": metrics["metrics/precision(B)"],
    "test/recall": metrics["metrics/recall(B)"],
    "test/AP_train": AP50,
    "test/AP50-95_train": AP5095,
    "test/P_train": P,
    "test/R_train": R,
})
wandb.log({
    "Test Metrics": wandb.Html(f"""
    <h2>🚆 Test Set Performance</h2>

    <table style="
        border-collapse: collapse;
        font-size: 16px;
        width: 60%;
    ">
      <thead>
        <tr>
          <th style="border:1px solid #ccc; padding:8px;">Metric</th>
          <th style="border:1px solid #ccc; padding:8px;">Mean (All Classes)</th>
          <th style="border:1px solid #ccc; padding:8px;">Train Class (ID 6)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="border:1px solid #ccc; padding:8px;">mAP@50</td>
          <td style="border:1px solid #ccc; padding:8px;">{metrics["metrics/mAP50(B)"]:.4f}</td>
          <td style="border:1px solid #ccc; padding:8px;">{AP50:.4f}</td>
        </tr>
        <tr>
          <td style="border:1px solid #ccc; padding:8px;">mAP@50–95</td>
          <td style="border:1px solid #ccc; padding:8px;">{metrics["metrics/mAP50-95(B)"]:.4f}</td>
          <td style="border:1px solid #ccc; padding:8px;">{AP5095:.4f}</td>
        </tr>
        <tr>
          <td style="border:1px solid #ccc; padding:8px;">Precision</td>
          <td style="border:1px solid #ccc; padding:8px;">{metrics["metrics/precision(B)"]:.4f}</td>
          <td style="border:1px solid #ccc; padding:8px;">{P:.4f}</td>
        </tr>
        <tr>
          <td style="border:1px solid #ccc; padding:8px;">Recall</td>
          <td style="border:1px solid #ccc; padding:8px;">{metrics["metrics/recall(B)"]:.4f}</td>
          <td style="border:1px solid #ccc; padding:8px;">{R:.4f}</td>
        </tr>
      </tbody>
    </table>
    """)
})


run_tags = list(wandb.run.tags) 

comparison_table = wandb.Table(
    columns=[
        "run_name",
        "tags",
        "mean_mAP50",
        "mean_mAP50_95",
        "mean_precision",
        "mean_recall",
        "train_mAP50",
        "train_mAP50_95",
        "train_precision",
        "train_recall",
    ]
)

comparison_table.add_data(
    wandb.run.name,
    run_tags,  # directly from initialized run
    metrics["metrics/mAP50(B)"],
    metrics["metrics/mAP50-95(B)"],
    metrics["metrics/precision(B)"],
    metrics["metrics/recall(B)"],
    AP50,
    AP5095,
    P,
    R,
)

wandb.log({"mAP_vs_trainAP": comparison_table})

# ============================
# Per-class TP / FP / FN / TN
# ============================

cm = results.confusion_matrix.matrix  # shape: (C+1, C+1)
class_names = results.names
num_classes = len(class_names)
total = cm.sum()

confusion_table = wandb.Table(
    columns=["class", "TP", "FP", "FN", "TN"]
)

for class_id in range(num_classes):
    TP = int(cm[class_id, class_id])
    FP = int(cm[:, class_id].sum() - TP)
    FN = int(cm[class_id, :].sum() - TP)
    TN = int(total - TP - FP - FN)

    confusion_table.add_data(
        class_names[class_id],
        TP,
        FP,
        FN,
        TN,
    )

wandb.log({"per_class_confusion": confusion_table})


wandb.finish()