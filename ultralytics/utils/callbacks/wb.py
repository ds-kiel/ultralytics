# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import SETTINGS, TESTS_RUNNING
from ultralytics.utils.torch_utils import model_info_for_loggers
from pathlib import Path
from collections import defaultdict



from pathlib import Path
from collections import defaultdict
import yaml

def load_dataset_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)

def normalize_to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]

def count_instances_in_dir(images_dir, class_names):
    """
    Count YOLO instances in one images directory.
    """
    counts = defaultdict(int)

    labels_dir = Path(str(images_dir).replace("/images", "/labels"))
    if not labels_dir.exists():
        return counts

    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                try:
                    cls_id = int(line.split()[0])
                    counts[class_names[cls_id]] += 1
                except KeyError:
                    print(
                        f"[CLASS ID NOT IN DATASET] "
                        f"file={label_file}, "
                        # f"line={line_num}, "
                        f"class_id={cls_id}"
                    )
                except (ValueError, IndexError):
                    print(f"error processing line in {label_file}: {line}")

    return counts

def empty_image_stats(images_dir):
    images_dir = Path(images_dir)
    labels_dir = Path(str(images_dir).replace("/images", "/labels"))

    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    total_images = len(image_files)

    empty_images = 0

    for img in image_files:
        label_file = labels_dir / f"{img.stem}.txt"
        if not label_file.exists():
            empty_images += 1
        else:
            if label_file.stat().st_size == 0:
                empty_images += 1

    return total_images, empty_images

def infer_source(path_str):
    path_str = str(path_str).lower()
    if "coco" in path_str:
        return "COCO"
    if "open_images" in path_str or "oid" in path_str:
        return "OpenImages"
    if "zollner" in path_str:
        return "Zollner"
    if "open_sensor" in path_str:
        return "OpenSensorRail"
    return "Other"


try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["wandb"] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, "__version__")  # verify package is not directory
    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


def log_yaml(path, name):
    path = Path(path)
    if path.exists():
        art = wb.Artifact(name=name, type="config")
        art.add_file(str(path))
        wb.run.log_artifact(art)

def _custom_table(x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of the default wandb precision-recall
    curve while allowing for enhanced customization. The visual metric is useful for monitoring model performance across
    different classes.

    Args:
        x (list): Values for the x-axis; expected to have length N.
        y (list): Corresponding values for the y-axis; also expected to have length N.
        classes (list): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot.
        x_title (str, optional): Label for the x-axis.
        y_title (str, optional): Label for the y-axis.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    import polars as pl  # scope for faster 'import ultralytics'
    import polars.selectors as cs

    df = pl.DataFrame({"class": classes, "y": y, "x": x}).with_columns(cs.numeric().round(3))
    data = df.select(["class", "y", "x"]).rows()

    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wb.plot_table(
        "wandb/area-under-curve/v0",
        wb.Table(data=data, columns=["class", "y", "x"]),
        fields=fields,
        string_fields=string_fields,
    )


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb. The curve can
    represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.

    Args:
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape (C, N), where C is the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C.
        id (str, optional): Unique identifier for the logged data in wandb.
        title (str, optional): Title for the visualization plot.
        x_title (str, optional): Label for the x-axis.
        y_title (str, optional): Label for the y-axis.
        num_x (int, optional): Number of interpolated data points for visualization.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted.

    Notes:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    import numpy as np

    # Create new x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # Create arrays for logging
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # add new x
            y_log.extend(np.interp(x_new, x, yi))  # interpolate y to new x
            classes.extend([names[i]] * len(x_new))  # add class names
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(plots, step):
    """Log plots to WandB at a specific step if they haven't been logged already.

    This function checks each plot in the input dictionary against previously processed plots and logs new or updated
    plots to WandB at the specified step.

    Args:
        plots (dict): Dictionary of plots to log, where keys are plot names and values are dictionaries containing plot
            metadata including timestamps.
        step (int): The step/epoch at which to log the plots in the WandB run.

    Notes:
        The function uses a shallow copy of the plots dictionary to prevent modification during iteration.
        Plots are identified by their stem name (filename without extension).
        Each plot is logged as a WandB Image object.
    """
    for name, params in plots.copy().items():  # shallow copy to prevent plots dict changing during iteration
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    """Initialize and start wandb project if module is present."""
    if not wb.run:
        wb.init(
            project=str(trainer.args.project).replace("/", "-") if trainer.args.project else "Ultralytics",
            name=str(trainer.args.name).replace("/", "-"),
            tags=str(trainer.args.name).split("_"),
            config=vars(trainer.args),
        )

        # -----------------------
        # LOG YAML CONFIG FILES
        # -----------------------

        # Dataset YAML (data.yaml)
        if hasattr(trainer.args, "data"):
            log_yaml(trainer.args.data, f"{wb.run.id}_data_yaml")

        # Model YAML (e.g. yolov8n.yaml)
        if hasattr(trainer.model, "yaml_file"):
            log_yaml(trainer.model.yaml_file, f"{wb.run.id}_model_yaml")
        
        data = load_dataset_yaml(trainer.args.data)
        root = Path(data["path"])
        class_names = data["names"]

        splits = ["train", "val", "test"]

        # Final structure:
        # counts[split][class] = total_instances
        counts = {s: defaultdict(int) for s in splits}

        for split in splits:
            split_entries = normalize_to_list(data.get(split))

            for rel_path in split_entries:
                images_dir = root / rel_path
                if not images_dir.exists():
                    print(f"[WARN] Missing dir: {images_dir}")
                    continue

                dir_counts = count_instances_in_dir(images_dir, class_names)

                for clss, n in dir_counts.items():
                    counts[split][clss] += n

        instance_table = wb.Table(
            columns=["split", "class", "num_instances"]
        )

        for split, cls_counts in counts.items():
            for clss in class_names.values():
                instance_table.add_data(
                    split,
                    clss,
                    cls_counts.get(clss, 0)
                )

        wb.log({"dataset/class_instance_counts": instance_table})

        for split, cls_counts in counts.items():
            total = sum(cls_counts.values())
            wb.run.summary[f"dataset/{split}_total_instances"] = total

            for clss, n in cls_counts.items():
                wb.log({f"dataset/{split}/class_instances/{clss}": n})
                wb.run.summary[f"dataset/{split}_{clss}_instances"] = n


        overall_counts = defaultdict(int)

        for split in counts:
            train_count = counts[split].get("train", 0)
            total = sum(counts[split].values())
            ratio = train_count / max(total, 1)

            wb.run.summary[f"dataset/{split}_train_ratio"] = ratio

            for clss, n in counts[split].items():
                overall_counts[clss] += n

        for clss, n in overall_counts.items():
            wb.log({f"dataset/all/class_instances/{clss}": n})
            
        empty_stats = {}

        for split in splits:
            total_imgs = 0
            empty_imgs = 0

            for rel_path in normalize_to_list(data.get(split)):
                images_dir = root / rel_path
                if not images_dir.exists():
                    continue

                t, e = empty_image_stats(images_dir)
                total_imgs += t
                empty_imgs += e

            ratio = empty_imgs / max(total_imgs, 1)
            empty_stats[split] = (total_imgs, empty_imgs, ratio)

            wb.run.summary[f"dataset/{split}_empty_images"] = empty_imgs
            wb.run.summary[f"dataset/{split}_total_images"] = total_imgs
            wb.run.summary[f"dataset/{split}_empty_ratio"] = ratio

        for split in splits:
            total_objects = sum(counts[split].values())
            train_objects = counts[split].get("train", 0)

            ratio = train_objects / max(total_objects, 1)

            wb.run.summary[f"dataset/{split}_train_objects"] = train_objects
            wb.run.summary[f"dataset/{split}_total_objects"] = total_objects
            wb.run.summary[f"dataset/{split}_train_object_ratio"] = ratio

        source_stats = {
            split: defaultdict(lambda: defaultdict(int)) for split in splits
        }

        for split in splits:
            for rel_path in normalize_to_list(data.get(split)):
                images_dir = root / rel_path
                if not images_dir.exists():
                    continue

                source = infer_source(rel_path)
                labels_dir = Path(str(images_dir).replace("/images", "/labels"))

                if not labels_dir.exists():
                    continue

                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file) as f:
                        for line in f:
                            cls_id = int(line.split()[0])
                            cls_name = class_names[cls_id]
                            source_stats[split][source][cls_name] += 1

        source_table = wb.Table(
            columns=["split", "source", "class", "num_instances"]
        )

        for split, src_data in source_stats.items():
            for source, cls_data in src_data.items():
                for clss, n in cls_data.items():
                    source_table.add_data(split, source, clss, n)

        wb.log({"dataset/source_class_distribution": source_table})

        for split, src_data in source_stats.items():
            for source, cls_data in src_data.items():
                total = sum(cls_data.values())
                wb.run.summary[f"dataset/{split}_{source}_objects"] = total




def on_fit_epoch_end(trainer):
    """Log training metrics and model information at the end of an epoch."""
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    if trainer.epoch == 0:
        wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)
    # wb.run.log(trainer.metrics, step=trainer.epoch + 1, commit=True)  # commit forces sync
    val_metrics = {}

    for k, v in trainer.metrics.items():
        if k.startswith("metrics/"):
            # metrics/mAP50(B) → val/mAP50
            new_key = k.replace("metrics/", "val/").replace("(B)", "")
            val_metrics[new_key] = v
        else:
            val_metrics[k] = v  # keep fitness etc.

    wb.run.log(val_metrics, step=trainer.epoch + 1, commit=True)


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    """Save the best model as an artifact and log final plots at the end of training."""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=["best"])
    # Check if we actually have plots to save
    if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
        for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(trainer.validator.metrics.names.values()),
                id=f"curves/{curve_name}",
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )
    
    wb.run.summary["final/train_loss"] = float(trainer.tloss.mean())
    wb.run.summary["best/fitness"] = trainer.best_fitness

    best = trainer.validator.metrics

    wb.run.summary.update({
        "best/mAP50": best.box.map50,
        "best/mAP50-95": best.box.map,
        "best/precision": best.box.mp,
        "best/recall": best.box.mr,
    })


    # --------------------
    # TEST SET EVALUATION
    # --------------------
    if hasattr(trainer.args, "data"):
        from ultralytics import YOLO

        model = YOLO(trainer.best)
        test_metrics = model.val(
            data=trainer.args.data,
            split="test",
            verbose=False
        )

        wb.run.log({
            "test/mAP50": test_metrics.box.map50,
            "test/mAP50-95": test_metrics.box.map,
            "test/precision": test_metrics.box.mp,
            "test/recall": test_metrics.box.mr,
        })
    
    wb.run.finish()  # required or run continues on dashboard


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
