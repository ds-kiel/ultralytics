from ultralytics import YOLO
import os
import wandb
from dotenv import load_dotenv
import os
from ultralytics.utils import SETTINGS
from ultralytics.utils.callbacks.wb import log_yaml

# Load environment variables from .env
load_dotenv()


wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key  # for wandb
print("WANDB_API_KEY loaded:", wandb_api_key)
#Initialize your Weights & Biases environment
wandb.login(key=wandb_api_key)
SETTINGS["wandb"] = True
run_name = "coco_zollner_single_cls_test"

run = wandb.init(
    project="zollner project",
    name=run_name,
    job_type="test",
)
run.tags = run.name.split("_")

# wandb.init(project="ultralytics", name="coco_8 forked_repo")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = YOLO('/home/mal/ultralytics/rail_road_scripts/BEST_coco_zollner_single_cls_train5.pt')

#UPDATE HERE
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