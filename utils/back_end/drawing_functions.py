import numpy as np
import cv2
from utils.data_classes import Bbox2D, Bbox3D


def draw_bbox2d(img, box, color=(255, 0, 0)):
    """ Draw a Bbox2D on image

    Args:
        img (np.ndarray): BGR image, shape (h, w, 3)
        box (Bbox2D): a 2D bounding box
        color (tuple): (B, G, R)
    """
    # clamp box size with image size
    box = box.clamp(img.shape[1], img.shape[0])
    draw_rect(img, box.corners(), color)


def draw_bbox3d(img, projection, label=None):
    """ Draw a Bbox3D on image

    Args:
        img (np.ndarray): BGR image, shape (h, w, 3)
        projection (np.ndarray): pixel-coordinate of projection of 8 corners
        label (str or int): box's label
    """
    draw_box3d_projection(img, projection, label)



def draw_rect(img, selected_corners, color, linewidth=2):
    """ Draw a rectangle (more like a polygon) by connecting 4 points in the image

    Args:
        img (np.ndarray): BGR image, shape (h, w, 3)
        selected_corners (np.ndarray): vertices of rectangle (almost), shape (4, 2)
        color (tuple): B, G, R color
        linewidth (float): width of rectangle edge
    """
    prev = selected_corners[-1]
    for corner in selected_corners:
        cv2.line(img,
                 (int(prev[0]), int(prev[1])),
                 (int(corner[0]), int(corner[1])),
                 color, linewidth)
        prev = corner


def draw_rect_label(img, label, top_left_corner):
    """ Draw label over an axis-aligned box in image

    Args:
        img (np.ndarray): BGR image, shape (h, w, 3)
        label (str or int): label to draw
        top_left_corner (list): [xmin, ymin] pixel-coordinate of the top_left_corner of the rect
    """
    if not isinstance(label, str):
        label = str(label)
    text_thickness, font_scale = 1, 1.0
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    # get label position
    text_x, text_y = top_left_corner[0], max(top_left_corner[1] - text_thickness, 0)
    # get textbox position
    textbox_x = min(top_left_corner[0] + text_size[0][0], img.shape[1])
    textbox_y = max(top_left_corner[1] - 2 * text_thickness - text_size[0][1], 0)
    # draw text box & label
    cv2.rectangle(img,
                  (int(top_left_corner[0]), int(top_left_corner[1])),
                  (int(textbox_x), int(textbox_y)),
                  (255, 0, 0), -1)
    cv2.putText(img, label, (int(text_x), int(text_y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), text_thickness)


def draw_box3d_projection(img, vertices, label=None, colors=((0, 0, 255), (255, 0, 0), (155, 155, 155)), linewidth=2):
    """ Draw a 3D Bbox on an image

    Args:
        img (np.ndarray): BGR image, shape (h, w, 3)
        vertices (np.ndarray): pixel-coordinate of 8 vertices of the projection on the image, shape (8, 2) \
            forward face is 0-1-2-3, backward face is 4-5-6-7, top face is 0-1-5-4, bottom face is 3-2-6-7
        label (int or str): label of the box
        colors (tuple[tuple]): ((B, G, R), (B, G, R), (B, G, R)). Colors for front, side & rear.
        linewidth (float): width of box's edges
    """
    # Draw the sides
    for i in range(4):
        cv2.line(img,
                 (int(vertices[i, 0]), int(vertices[i, 1])),
                 (int(vertices[i + 4, 0]), int(vertices[i + 4, 1])),
                 colors[2][::-1], linewidth)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(img, vertices[:4], colors[0][::-1])
    draw_rect(img, vertices[4:], colors[1][::-1])

    # Draw line indicating the front
    center_bottom_forward = np.mean(vertices[2:4], axis=0)
    center_bottom = np.mean(vertices[[2, 3, 7, 6]], axis=0)
    cv2.line(img,
             (int(center_bottom[0]), int(center_bottom[1])),
             (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
             colors[0][::-1], linewidth)

    if label is not None:
        draw_rect_label(img, label, np.amin(vertices, axis=0).tolist())

