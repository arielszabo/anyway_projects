import os
import yaml
import pydarknet
from detection import blur_the_video, find_all


CLASS_LABELS = ['car', 'person', 'motorbike', 'truck', 'bus']
VIDEOS_CONFIG_YAML = "videos_config.yaml"
AMOUNT_OF_NEAR_FRAMES_INCLUDE_BLUR = 1


def main():
    with open(VIDEOS_CONFIG_YAML, "r") as yamlfile:
        videos_config = yaml.load(yamlfile)

    input_videos_folder_path = videos_config["input_videos_folder_path"]
    output_folder_path = videos_config["output_folder_path"]
    os.makedirs(output_folder_path, exist_ok=True)

    for video in videos_config["videos"]:
        video_full_path = os.path.join(input_videos_folder_path, video["name"])
        output_video_full_path = get_video_output_full_path(video_name=video["name"],
                                                            video_start_frame=video["start_frame"],
                                                            video_end_frame=video["end_frame"],
                                                            output_folder_path=output_folder_path)

        run(input_video_path=video_full_path,
            start_frame=video["start_frame"],
            end_frame=video["end_frame"],
            output_video_path=output_video_full_path)


def run(input_video_path, start_frame, end_frame, output_video_path):
    coco_net = pydarknet.Detector(config=bytes("cfg/yolov3.cfg", encoding="utf-8"),
                                  weights=bytes("weights/yolov3.weights", encoding="utf-8"),
                                  p=0,
                                  meta=bytes("cfg/coco.data", encoding="utf-8"))
    all_frames_bounds = find_all(video_path=input_video_path,
                                 darknet_model=coco_net,
                                 class_labels=CLASS_LABELS,
                                 start_frame=start_frame,
                                 end_frame=end_frame)

    blur_the_video(video_path=input_video_path,
                   output_path=output_video_path,
                   frames_bounds=all_frames_bounds,
                   start_frame=start_frame,
                   end_frame=end_frame)


def get_video_name_and_extension(video_name):
    """
    Split the video name  todo: explain this better
    """
    reverse_video_name = video_name[::-1]

    reversed_ending, reversed_beginning = reverse_video_name.split(".", maxsplit=1)
    return reversed_beginning[::-1], reversed_ending[::-1]


def get_video_output_full_path(video_name, video_start_frame, video_end_frame, output_folder_path):
    core_video_name, video_extension = get_video_name_and_extension(video_name)
    if video_start_frame is None:
        video_start_frame = 0

    if video_end_frame is None:
        frames_signature = str(video_start_frame)
    else:
        frames_signature = f"{video_start_frame}_{video_end_frame}"
    output_video_name = f"{core_video_name}_{frames_signature}.{video_extension}"
    output_full_path = os.path.join(output_folder_path, output_video_name)
    return output_full_path


if __name__ == '__main__':
   main()