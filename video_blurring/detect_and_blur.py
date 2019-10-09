import os
import yaml
import cv2
import pydarknet
from tqdm import tqdm
from detection_utils import get_video_output_full_path, find_bounds_in_image, get_video_params, add_blur


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


def find_all(video_path, darknet_model,
             class_labels, start_frame=0,
             end_frame=None):
    video_capture = cv2.VideoCapture(video_path)

    if end_frame is None:
        end_frame = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    with tqdm(total=end_frame-start_frame, desc="Detect and find") as progress_bar:
        frame_counter = 0
        frames_bounds = []
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                if end_frame > frame_counter > start_frame:

                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # todo: is this a must ?
                    frame_results_bounds = find_bounds_in_image(image_frame=frame,
                                                                darknet_model=darknet_model,
                                                                class_label=class_labels)
                    frames_bounds.append(frame_results_bounds)
                    progress_bar.update(1)
                frame_counter += 1


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    # Release everything if job is finished
    video_capture.release()
    cv2.destroyAllWindows()
    return frames_bounds


def blur_the_video(video_path, output_path, frames_bounds, start_frame=0,
                   end_frame=None):
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        width, height, fps, fourcc = get_video_params(video_capture=cap)
        print(width, height, fps, fourcc)

        if end_frame is None:
            end_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        output = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        with tqdm(total=end_frame - start_frame, desc="Blur found boundaries") as progress_bar:
            frame_counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    if end_frame > frame_counter > start_frame:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        for i in range(-AMOUNT_OF_NEAR_FRAMES_INCLUDE_BLUR, AMOUNT_OF_NEAR_FRAMES_INCLUDE_BLUR+1):
                            real_index = frame_counter - start_frame
                            if 0 <= real_index + i < len(frames_bounds):
                                bounds = frames_bounds[real_index + i]

                                frame = add_blur(frame,
                                                  bounds,
                                                  expand=False)

                        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        output.write(new_frame)
                        progress_bar.update(1)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    frame_counter += 1


                else:
                    break

        # Release everything if job is finished
        output.release()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
   main()