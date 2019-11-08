import os
import cv2
import pydarknet

DETECTION_THRESHOLD = 0.1
BOUND_RATIO_THRESHOLD = 0.5


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
