import cv2
import numpy as np

from pse import pse


def draw_result(origin_image, boxes, save_fn="result.jpg"):
    """Draw text boxes to original image.
    Args:
        origin_image: the original image.
        boxes: boxes detected.
        save_fn: save file name.
    Returns:
        None
    """
    for i in range(len(boxes)):
        box = boxes[i]
        cv2.polylines(
            origin_image,
            [box.astype(np.int32).reshape((-1, 1, 2))],
            True,
            color=(255, 255, 0),
            thickness=2,
        )
    cv2.imwrite(save_fn, origin_image)


def resize_image(image, max_side_len=1024):
    """Resize image so that both height and width are multiples of 32.
    Args:
        image: image to be resized.
        max_side_len: max length for both height and width.
    Returns:
        image: resized image.
        ratio_h: ratio of height.
        ratio_w: ratio of width.
    """
    h, w, _ = image.shape
    resize_w = w
    resize_h = h
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.0

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    image = cv2.resize(image, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return image, (ratio_h, ratio_w)


def preprocess(image):
    """Do preporcess as TensorFlow PSENet.
    Args:
        image: image to be processed.
    Returns:
        image: processed image.
        ratio_h: ratio of height.
        ratio_w: ratio of width.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, (ratio_h, ratio_w) = resize_image(image)
    image = image - (123.68, 116.78, 103.94)
    return image, (ratio_h, ratio_w)


def detect(seg_maps, image_h, image_w, min_area_thresh=10, seg_map_thresh=0.9, ratio=1):
    """Detect text boxes from score map and geo map
    Args:
        seg_maps: 6 segmentation maps from network.
        image_h: height of original image.
        image_w: width of original image.
        min_area_thresh: min area to be detected.
        seg_map_thresh: segmentation threshlod.
        ratio: segmentation ratio.
    Returns:
        boxes: detected text boxes.
    """
    # get kernals, sequence: 0->n, max -> min
    kernals = []
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1] - 1, -1, -1):  # 5(big),4,3,2,1,0
        # 0.8ms
        kernal = np.where(seg_maps[..., i] > thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh * ratio
    mask_res, label_values = pse(kernals, min_area_thresh)  # 2.8ms
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    for label_value in label_values:
        # (y,x)
        points = np.argwhere(mask_res_resized == label_value)
        points = points[:, (1, 0)]
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        boxes.append(box)

    return np.array(boxes)


def postprocess(origin_image, seg_maps, ratio_h, ratio_w):
    """Post process segmentation maps to get text boxes.
    Args:
        origin_image: original image.
        seg_maps: 6 segmentation maps from network.
        ratio_h: ratio of height.
        ratio_w: ratio of width.
    Returns:
        boxes: detected text boxes.
    """
    h, w, _ = origin_image.shape
    boxes = detect(
        seg_maps=seg_maps,
        image_h=h,
        image_w=w,
        min_area_thresh=5,
        seg_map_thresh=0.9,
    )
    if len(boxes) > 0:
        boxes = boxes.reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h
        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w)
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h)
    return boxes
