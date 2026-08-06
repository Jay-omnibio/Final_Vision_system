"""Core background-subtraction functions."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .types import BackgroundSubtractionConfig, Detection, Rect


def crop_roi(image: np.ndarray, roi: Rect | None) -> np.ndarray:
    if roi is None:
        return image
    x, y, width, height = roi
    return image[y : y + height, x : x + width]


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference_height, reference_width = reference.shape[:2]
    return cv2.resize(image, (reference_width, reference_height))


def create_standard_mask(
    frame: np.ndarray,
    background: np.ndarray,
    config: BackgroundSubtractionConfig,
) -> np.ndarray:
    if frame.shape != background.shape:
        if not config.resize_frame_to_background:
            raise ValueError("Frame and background shapes differ.")
        frame = resize_like(frame, background)

    frame_roi = crop_roi(frame, config.roi)
    background_roi = crop_roi(background, config.roi)

    diff = cv2.absdiff(frame_roi, background_roi)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY)
    return clean_mask(mask, config.kernel_size)


def create_improved_mask(
    frame: np.ndarray,
    background: np.ndarray,
    config: BackgroundSubtractionConfig,
) -> np.ndarray:
    if frame.shape != background.shape:
        if not config.resize_frame_to_background:
            raise ValueError("Frame and background shapes differ.")
        frame = resize_like(frame, background)

    frame_roi = crop_roi(frame, config.roi)
    background_roi = crop_roi(background, config.roi)

    diff = cv2.absdiff(frame_roi, background_roi)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask_bgr = cv2.threshold(gray, config.threshold, 255, cv2.THRESH_BINARY)

    frame_hsv = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2HSV)
    background_hsv = cv2.cvtColor(background_roi, cv2.COLOR_BGR2HSV)
    value_diff = cv2.absdiff(frame_hsv[:, :, 2], background_hsv[:, :, 2])
    _, mask_value = cv2.threshold(
        value_diff,
        config.threshold + config.hsv_threshold_boost,
        255,
        cv2.THRESH_BINARY,
    )

    return clean_mask(cv2.bitwise_and(mask_bgr, mask_value), config.kernel_size)


def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_detections(mask: np.ndarray, min_area: int = 500) -> list[Detection]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        detections.append(Detection(int(x), int(y), int(width), int(height), float(area)))
    return detections


def box_iou(first: Detection, second: Detection) -> float:
    first_x2 = first.x + first.width
    first_y2 = first.y + first.height
    second_x2 = second.x + second.width
    second_y2 = second.y + second.height
    intersection_x1 = max(first.x, second.x)
    intersection_y1 = max(first.y, second.y)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    if intersection_area == 0:
        return 0.0
    union_area = first.width * first.height + second.width * second.height - intersection_area
    return intersection_area / union_area


def horizontal_gap(first: Detection, second: Detection) -> int:
    first_right = first.x + first.width
    second_right = second.x + second.width
    if first.x <= second.x:
        return max(0, second.x - first_right)
    return max(0, first.x - second_right)


def vertical_overlap_ratio(first: Detection, second: Detection) -> float:
    overlap = max(
        0,
        min(first.y + first.height, second.y + second.height) - max(first.y, second.y),
    )
    smaller_height = min(first.height, second.height)
    return 0.0 if smaller_height <= 0 else overlap / smaller_height


def vertical_gap(first: Detection, second: Detection) -> int:
    first_bottom = first.y + first.height
    second_bottom = second.y + second.height
    if first.y <= second.y:
        return max(0, second.y - first_bottom)
    return max(0, first.y - second_bottom)


def horizontal_overlap_ratio(first: Detection, second: Detection) -> float:
    overlap = max(
        0,
        min(first.x + first.width, second.x + second.width) - max(first.x, second.x),
    )
    smaller_width = min(first.width, second.width)
    return 0.0 if smaller_width <= 0 else overlap / smaller_width


def expanded_box(detection: Detection, gap: int) -> Rect:
    return (
        detection.x - gap,
        detection.y - gap,
        detection.width + gap * 2,
        detection.height + gap * 2,
    )


def boxes_intersect(first: Rect, second: Rect) -> bool:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    return (
        first_x < second_x + second_width
        and first_x + first_width > second_x
        and first_y < second_y + second_height
        and first_y + first_height > second_y
    )


def merge_pair(first: Detection, second: Detection) -> Detection:
    x1 = min(first.x, second.x)
    y1 = min(first.y, second.y)
    x2 = max(first.x + first.width, second.x + second.width)
    y2 = max(first.y + first.height, second.y + second.height)
    return Detection(int(x1), int(y1), int(x2 - x1), int(y2 - y1), float(first.area + second.area))


def should_merge_boxes(
    first: Detection,
    second: Detection,
    config: BackgroundSubtractionConfig,
) -> bool:
    if box_iou(first, second) > config.merge_iou_threshold:
        return True
    if (
        horizontal_gap(first, second) <= config.merge_horizontal_gap
        and vertical_overlap_ratio(first, second) >= config.merge_vertical_overlap_ratio
    ):
        return True
    if (
        vertical_gap(first, second) <= config.merge_vertical_gap
        and horizontal_overlap_ratio(first, second) >= config.merge_horizontal_overlap_ratio
    ):
        return True
    if config.merge_gap <= 0:
        return False
    return boxes_intersect(expanded_box(first, config.merge_gap), expanded_box(second, config.merge_gap))


def merge_detections(
    detections: Sequence[Detection],
    config: BackgroundSubtractionConfig,
) -> list[Detection]:
    merged = list(detections)
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            for second_index in range(first_index + 1, len(merged)):
                if should_merge_boxes(merged[first_index], merged[second_index], config):
                    merged[first_index] = merge_pair(merged[first_index], merged[second_index])
                    del merged[second_index]
                    changed = True
                    break
            if changed:
                break
    return merged

