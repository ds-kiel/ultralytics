from ultralytics import YOLO
import os
import wandb
# from wandb.integration.ultralytics import add_wandb_callback

# Initialize your Weights & Biases environment
wandb.login(key=os.getenv("WANDB_API_KEY"))
# wandb.init(project="ultralytics", name="coco_8 forked_repo")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = YOLO('/home/mal/coco_projects/ultralytics/yolov8/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml')
# model = YOLO("/home/mal/coco_projects/ultralytics/yolov8/ultralytics/train_scratch_coco_OS_Zol_BG_cargo_3004/weights/best.pt") 
# model = YOLO('yolov8s.pt')
# add_wandb_callback(model, enable_model_checkpointing=True)

results = model.train(
    data='/home/mal/ultralytics/ultralytics/cfg/datasets/zollner_train.yaml',
    epochs=100, 
    imgsz=640,
    patience=100,
    project="zollner project",
    name="scratch_OS_OI_Zollner_train",
    )
wandb.finish()