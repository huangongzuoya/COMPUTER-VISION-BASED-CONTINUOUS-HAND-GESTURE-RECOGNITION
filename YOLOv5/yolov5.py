import argparse
import os
import warnings
from YOLOv5_Detection.yolov5 import detect
from YOLOv5_Detection.yolov5.utils.general import check_requirements

from YOLOv5.setting import get_file_path, get_output_dir


def get_output_from_video(video_filenames: list, model_files: list, output_dir: str = None) -> None:
    if (not model_files) or (not video_filenames):
        raise ValueError(f"No model or video file provided")

    for model_file in model_files:
        if not os.path.exists(model_file):
            warnings.warn(f"model '{model_file}' does not exist!", UserWarning)
            print(f"Warning: model '{model_file}' does not exist!")
            continue

        for video_filename in video_filenames:
            video_file_path = get_file_path(video_filename)
            if not video_file_path.lower().endswith('.mp4'):
                warnings.warn(f"File '{video_file_path}' is not a video file. "
                              f"(Currently only mp4 type video files are supported.)", UserWarning)
                print(f"File '{video_file_path}' is not a video file. "
                      f"(Currently only mp4 type video files are supported.)")
                continue

            output_dir = get_output_dir(video_filename, model_file)
            try:
                run_yolov5_detection(video_file_path, model_file, output_dir)
            except Exception as e:
                print(f"Run YOLOv5 detection failed. \nModel: {model_file} \nFile: {video_file_path}\nError: {e}")
    return


def run_yolov5_detection(video_file: str, model_file: str, output_dir: str) -> None:
    check_requirements(exclude=('tensorboard', 'thop'))
    opt = argparse.Namespace(
        weights=model_file,
        source=video_file,
        save_txt=True,
        save_conf=True,
        name="",
        exist_ok=True,
        view_img=False,
        project=output_dir,
    )

    detect.run(**vars(opt))
    return


# def get_lstm_single_input_from_yolov5_output(txt_file_path: str) -> None:
#     pass

if __name__ == '__main__':
    # get_output_from_video(video_filenames=[f"video{i}.mp4" for i in range(13, 14)],
    #                       model_files=[f"../YOLOv5_Detection/yolov5/runs/train/exp{'' if i == 0 else i}/weights/best.pt"
    #                                    for i in range(4)])
    get_output_from_video(video_filenames=[f"video{i}.mp4" for i in range(278, 302)],
                          model_files=["../YOLOv5_Detection/yolov5/runs/train/exp/weights/best.pt"])
