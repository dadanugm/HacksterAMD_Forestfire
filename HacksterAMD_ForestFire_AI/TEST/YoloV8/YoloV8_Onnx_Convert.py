### Convert YoloV8 to ONNX Model ###
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
import onnx
import shutil


def get_yolo_model(version: int, onnx_model_name: str):
    # install yolov8
    from pip._internal import main as pipmain
    try:
        import ultralytics
    except ImportError:
        pipmain(['install', 'ultralytics'])
        import ultralytics
    pt_model = Path(f"model/forestfire_model.pt")
    model = ultralytics.YOLO(str(pt_model))  # load a pretrained model
    exported_filename = model.export(format="onnx",imgsz=[640,640], opset=11)  # export the model to ONNX format
    assert exported_filename, f"Failed to export yolov{version}n.pt to onnx"
    #shutil.move(exported_filename, onnx_model_name)
    # check/validate the model
    print("check/validate the model")
    model_onnx = onnx.load('model/forestfire_model.onnx')
    onnx.checker.check_model(model_onnx)


def add_pre_post_processing_to_yolo(input_model_file: Path, output_model_file: Path):
    """Construct the pipeline for an end2end model with pre and post processing.
    The final model can take raw image binary as inputs and output the result in raw image file.

    Args:
        input_model_file (Path): The onnx yolo model.
        output_model_file (Path): where to save the final onnx model.
    """
    from onnxruntime_extensions.tools import add_pre_post_processing_to_model as add_ppp
    add_ppp.yolo_detection(input_model_file, output_model_file, "jpg", onnx_opset=18)
    #shutil.copy("model/forestfire_model.onnx", "model/forestfire_model.pre_post_process.onnx")


if __name__ == '__main__':
    # YOLO version. Tested with 5 and 8.
    version = 8
    onnx_model_name = Path(f"model/forestfire_model.onnx")
    if not onnx_model_name.exists():
        print("Fetching original model...")
        get_yolo_model(version, str(onnx_model_name))

    #onnx_e2e_model_name = onnx_model_name.with_suffix(suffix=".pre_post_process.onnx")
    #print("Adding pre/post processing...")
    #add_pre_post_processing_to_yolo(onnx_model_name, onnx_e2e_model_name)