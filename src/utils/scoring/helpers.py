import math

import mediapipe as mp
import numpy as np


def calculate_angle(a: list, b: list, c: list) -> float:
    """Calculate angles for three points.

    Parameters
    ----------
    a : list
        List of coordinates of first point
    b : list
        List of coordinates of second point
    c : list
        List of coordinates of third point

    Returns
    -------
    float
        Calculated angle in float
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(
        a[1] - b[1], a[0] - b[0]
    )
    angle = np.abs(radians * 180 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


def calculate_flatfoot(a: list, b: list, c: list) -> float:
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    vector1 = a - b
    vector2 = c - b

    angle = math.atan2(vector2[1], vector2[0]) - math.atan2(vector1[1], vector1[0])
    angle = abs(angle * (180.0 / np.pi))  # Convert to degrees

    # Adjust angle to be within 0-180 degrees
    if angle > 180:
        angle = 360 - angle
    return angle


def calculate_midpoint(x1, x2, y1, y2):
    """Calculates midpoint between to coordinates"""
    return (((x1 + x2) / 2), ((y1 + y2) / 2))


def calculate_distance(a, b):
    """Calculate the Euclidean distance between two points"""
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def calculate_angle_with_threshold(heel, foot_index, threshold_y):
    """Calculate the angle between the line from heel to foot index and the horizontal threshold line.

    Parameters
    ----------
    heel : tuple
        Coordinates of the heel (x, y).
    foot_index : tuple
        Coordinates of the foot index (x, y).
    threshold_y : float
        Y-coordinate of the threshold line.

    Returns
    -------
    float
        The angle in degrees between the heel-foot line and the threshold line.
    """
    heel_index_vector = (foot_index[0] - heel[0], foot_index[1] - heel[1])
    threshold_vector = (0, threshold_y - heel[1])

    angle = math.degrees(
        math.atan2(heel_index_vector[1], heel_index_vector[0])
        - math.atan2(threshold_vector[1], threshold_vector[0])
    )

    # Normalize angle to be between 0 and 180
    angle = abs(angle) % 360
    if angle > 180:
        angle = 360 - angle

    return angle


def which_foot_landed(
    landmarks,
    height: int,
    threshold: int,
) -> tuple[bool, bool]:
    """Determines what foot landed by checking which foot is below
    the threshold line (estimated ground)

    Parameters
    ----------
    landmarks : List
        List of landmarks returned from Mediapipe's Pose Estimation
    height : int
        Video's height in pixels
    threshold : int
        Estimated ground threshold

    Returns
    -------
    tuple[bool, bool]
        Booleans indicating if the left or right foot is landed in tuple
    """

    left_foot_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX.value].y
    right_foot_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y

    left_foot_near_ground = left_foot_y > (threshold / height)
    right_foot_near_ground = right_foot_y > (threshold / height)

    left_foot_landed = left_foot_near_ground
    right_foot_landed = right_foot_near_ground

    return left_foot_landed, right_foot_landed
