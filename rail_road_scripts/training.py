from ultralytics import YOLO
import os
import wandb
# from wandb.integration.ultralytics import add_wandb_callback
from dotenv import load_dotenv
import os
from ultralytics.utils import SETTINGS


# Load environment variables from .env
load_dotenv()

# Now you can use them
wandb_api_key = os.getenv("WANDB_API_KEY")
os.environ["WANDB_API_KEY"] = wandb_api_key  # for wandb
print("WANDB_API_KEY loaded:", wandb_api_key)
# Initialize your Weights & Biases environment
wandb.login(key=wandb_api_key)
SETTINGS["wandb"] = True


# wandb.init(project="ultralytics", name="coco_8 forked_repo")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = YOLO('/home/mal/coco_projects/ultralytics/yolov8/ultralytics/ultralytics/cfg/models/v8/yolov8.yaml')
model.model.args["scale"] = "s"
# model= YOLO("/home/mal/ultralytics/rail_road_scripts/zollner project/scratch_COCO_OS_OI_Zollner_train3/weights/last.pt")
# model = YOLO("/home/mal/coco_projects/ultralytics/yolov8/ultralytics/train_scratch_coco_OS_Zol_BG_cargo_3004/weights/best.pt") 
# model = YOLO('yolov8s.pt')
# add_wandb_callback(model, enable_model_checkpointing=True)

results = model.train(
    data='/home/mal/ultralytics/ultralytics/cfg/datasets/zollner_train_single_cls.yaml',
    epochs=300, 
    imgsz=640,
    patience=100,
    project="zollner project",
    single_cls=False,
    name="single_class_train",
    )

wandb.finish()