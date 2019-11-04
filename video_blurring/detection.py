import os
import cv2
import pydarknet

DETECTION_THRESHOLD = 0.1
BOUND_RATIO_THRESHOLD = 0.5


def get_video_name_and_extension(video_name):
    """
    Split the video name  todo: explain this better
    """
    reverse_video_name = video_name[::-1]

    reversed_ending, reversed_beginning = reverse_video_name.split(".", maxsplit=1)
    return reversed_beginning[::-1], reversed_ending[::-1]


def get_video_output_full_path(video_name, video_start_frame, video_end_frame, output_folder_path):
    core_video_name, video_extension = get_video_name_and_extension(video_name)
    if video_end_frame is None:
        frames_signature = str(video_start_frame)
    else:
        frames_signature = f"{video_start_frame}_{video_end_frame}"
    output_video_name = f"{core_video_name}_{frames_signature}.{video_extension}"
    output_full_path = os.path.join(output_folder_path, output_video_name)
    return output_full_path


def get_box_size_ratio(y1, y2, x1, x2, image_shape):
    return (abs(y2 - y1) * abs(x2 - x1)) / (image_shape[0] * image_shape[1])


def expand_mask(image_shape, x_left, x_right, y_buttom, y_top, expand_amount=5):
    max_x = image_shape[0]
    max_y = image_shape[1]

    expanded_x_right = x_right + expand_amount
    expanded_y_top = y_top + expand_amount
    expanded_x_left = x_left - expand_amount
    expanded_y_buttom = y_buttom - expand_amount

    if expanded_x_left < 0:  # min_x
        expanded_x_left = 0

    if expanded_y_buttom < 0:  # min_y
        expanded_y_buttom = 0

    if expanded_x_right > max_x:
        expanded_x_right = max_x

    if expanded_y_top > max_y:
        expanded_y_top = max_y

    return expanded_x_left, expanded_x_right, expanded_y_buttom, expanded_y_top


def get_video_params(video_capture):
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # # Define the codec and create VideoWriter object
    # fourcc = int(video_capture.get(cv2.CAP_PROP_FOURCC))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # print(video_capture.get(cv2.CAP_PROP_FORMAT))

    return width, height, fps, fourcc


def add_mask(image_frame, results_bounds):
    for (y1, y2, x1, x2) in results_bounds:
        cv2.rectangle(image_frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        # cv2.putText(image_frame, category, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

    return image_frame


def add_blur(image_frame, results_bounds, expand=False):
    for (y1, y2, x1, x2) in results_bounds:
        if expand:
            x1, x2, y1, y2 = expand_mask(image_shape=image_frame.shape,
                                         x_left=x1,
                                         x_right=x2,
                                         y_buttom=y1,
                                         y_top=y2,
                                         expand_amount=20)


        y1 = max(y1, 0)
        y2 = max(y2, 0)
        x1 = max(x1, 0)
        x2 = max(x2, 0)
        image_frame[y1:y2, x1:x2] = cv2.GaussianBlur(image_frame[y1:y2, x1:x2], (11, 11), cv2.BORDER_DEFAULT)

    return image_frame


def find_bounds_in_image(image_frame, darknet_model, class_label):
    darknet_image_frame = pydarknet.Image(image_frame)

    results = darknet_model.detect(darknet_image_frame,
                                   thresh=DETECTION_THRESHOLD,
                                   hier_thresh=.5, nms=.45)  # todo: change this thresh-params thresh=0.01 #0.00051,
    results_bounds = []
    for category, score, bounds in results:
        category = str(category.decode("utf-8"))
        if category.lower() in class_label:
            x, y, w, h = bounds
            y1, y2 = int(y - h / 2), int(y + h / 2)
            x1, x2 = int(x - w / 2), int(x + w / 2)

            image_box_size_ratio = get_box_size_ratio(y1, y2, x1, x2, image_shape=image_frame.shape)
            if image_box_size_ratio < BOUND_RATIO_THRESHOLD:
                results_bounds.append((y1, y2, x1, x2))

    return results_bounds
