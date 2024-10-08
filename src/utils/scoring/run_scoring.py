"""
NOTE: Print statements are for debugging purposes.


Script for calculating Run skill based on
TGMD-2 Performance Criteria Checklist.
"""

from collections import namedtuple

import mediapipe as mp

from utils.scoring import helpers

# Create a namedtuple for a 2D point (x, y)
Point = namedtuple("Point", ["x", "y"])


def get_landmark_point(landmarks, landmark_index):
    """Extract a specific landmark's coordinates."""
    landmark = landmarks[landmark_index]
    return Point(landmark.x, landmark.y)


def criteria_1(landmarks, mp_pose: mp.solutions.pose) -> bool:
    """Checks if elbows are bent and move in opposition to legs for running form.

    Parameters
    ----------
    landmarks : list
        List of landmarks returned from Mediapipe's Pose Estimation
    mp_pose : mp.solutions.pose
        Index for each landmark

    Returns
    -------
    bool
        Returns True if the criteria is met
    """

    # Extract landmarks using helper function
    LEFT_HIP = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
    LEFT_KNEE = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value)
    LEFT_ANKLE = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value)

    RIGHT_HIP = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
    RIGHT_KNEE = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value)
    RIGHT_ANKLE = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    LEFT_SHOULDER = get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value
    )
    LEFT_ELBOW = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
    LEFT_WRIST = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)

    RIGHT_SHOULDER = get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    )
    RIGHT_ELBOW = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    RIGHT_WRIST = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)

    # Calculate angles for elbows and legs
    LEFT_LEG_ANGLE = helpers.calculate_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
    RIGHT_LEG_ANGLE = helpers.calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
    LEFT_ELBOW_ANGLE = helpers.calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST)
    RIGHT_ELBOW_ANGLE = helpers.calculate_angle(
        RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
    )

    # Criteria logic: Elbows bent (less than 90 degrees) and arms moving in opposition to legs
    elbows_bent = LEFT_ELBOW_ANGLE < 90.0 or RIGHT_ELBOW_ANGLE < 90.0
    arms_leg_opposition = (
        RIGHT_ELBOW_ANGLE < LEFT_ELBOW_ANGLE and RIGHT_LEG_ANGLE > LEFT_LEG_ANGLE
    )

    return elbows_bent and arms_leg_opposition


def criteria_2(landmarks, mp_pose: mp.solutions.pose, height: int):
    """Checks if both feet are off the ground briefly. It determines
    the threshold line by setting the maximum y-value of a flat-footed
    detected foot. If both y-axis for both feet are off the ground, it
    returns True.

    Parameters
    ----------
    landmarks : list
        List of landmarks returned from Mediapipe's Pose Estimation
    mp_pose : mp.solutions.pose
        Index for each landmark
    height: int
        Height of the video in pixels

    Returns
    -------
    bool
        Returns True if both feet are above the threshold line.
    int or None
        Returns the threshold pixel value if calculated, otherwise None.
    """

    # Initialize threshold pixel y as None
    threshold_pixel_y = None

    # LEFT FOOT COORDINATES
    LEFT_HEEL = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_HEEL.value)
    LEFT_FOOT_INDEX = get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    )
    LEFT_FOOT_MIDPOINT = helpers.calculate_midpoint(
        LEFT_HEEL.x, LEFT_FOOT_INDEX.x, LEFT_HEEL.y, LEFT_FOOT_INDEX.y
    )

    # RIGHT FOOT COORDINATES
    RIGHT_HEEL = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_HEEL.value)
    RIGHT_FOOT_INDEX = get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
    )
    RIGHT_FOOT_MIDPOINT = helpers.calculate_midpoint(
        RIGHT_HEEL.x, RIGHT_FOOT_INDEX.x, RIGHT_HEEL.y, RIGHT_FOOT_INDEX.y
    )

    # Calculate angle for both feet
    LEFT_FOOT_ANGLE = helpers.calculate_flatfoot(
        LEFT_HEEL, LEFT_FOOT_MIDPOINT, LEFT_FOOT_INDEX
    )
    RIGHT_FOOT_ANGLE = helpers.calculate_flatfoot(
        RIGHT_HEEL, RIGHT_FOOT_MIDPOINT, RIGHT_FOOT_INDEX
    )

    # Check if either foot is flat, if so, update the threshold_pixel_y
    left_foot_flat = 176 <= LEFT_FOOT_ANGLE <= 184
    right_foot_flat = 176 <= RIGHT_FOOT_ANGLE <= 184

    # If both feet are flat at the same time, check which foot is forward
    if left_foot_flat and right_foot_flat:
        # Compare y-values (heights) of both feet
        left_y = int(LEFT_FOOT_INDEX[1] * height)
        right_y = int(RIGHT_FOOT_INDEX[1] * height)

        # Check if the feet are roughly at the same y-coordinate, considering precision
        if abs(left_y - right_y) <= 5:  # Precision threshold for pixel difference
            # Choose the forward foot (larger x-value)
            if LEFT_FOOT_INDEX[0] > RIGHT_FOOT_INDEX[0]:
                threshold_pixel_y = left_y
                # print(f"Forward foot: Left, Threshold set to: {threshold_pixel_y}")
            else:
                threshold_pixel_y = right_y
                # print(f"Forward foot: Right, Threshold set to: {threshold_pixel_y}")
        else:
            # If feet are not at the same y, choose the lower foot's y-value as the threshold
            threshold_pixel_y = max(left_y, right_y)
            # print(f"Feet not aligned, threshold set to: {threshold_pixel_y}")

    # If only one foot is flat, use that foot's y-value
    elif left_foot_flat:
        threshold_pixel_y = int(LEFT_FOOT_INDEX[1] * height)
        # print(f"Left foot flat, threshold: {threshold_pixel_y}")
    elif right_foot_flat:
        threshold_pixel_y = int(RIGHT_FOOT_INDEX[1] * height)
        # print(f"Right foot flat, threshold: {threshold_pixel_y}")

    # Check if both feet are off the ground based on the calculated threshold
    both_feet_off_ground = False
    if threshold_pixel_y is not None:
        both_feet_off_ground = (LEFT_HEEL[1] * height < threshold_pixel_y) and (
            RIGHT_HEEL[1] * height < threshold_pixel_y
        )

    # print(
    #     f"Both feet off ground: {both_feet_off_ground}, Threshold: {threshold_pixel_y}"
    # )

    return both_feet_off_ground, threshold_pixel_y


def criteria_3(
    landmarks, mp_pose: mp.solutions.pose, height, threshold_pixel_y
) -> bool:
    """Calculates distance between heel of a foot and toe of another toe,
    must be narrow. Also checks if landing foot is parallel to the threshold (ground)

    Parameters
    ----------
    landmarks : list
        List of landmarks returned from Mediapipe's Pose Estimation
    mp_pose : mp.solutions.pose
        Index for each landmark

    Returns
    -------
    bool
        Returns True if criteria 3 is met
    """

    left_foot_landed = False
    right_foot_landed = False

    # LEFT FOOT COORDINATES
    LEFT_HEEL = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_HEEL.value)
    LEFT_FOOT_INDEX = get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    )
    LEFT_FOOT_MIDPOINT = helpers.calculate_midpoint(
        LEFT_HEEL.x, LEFT_FOOT_INDEX.x, LEFT_HEEL.y, LEFT_FOOT_INDEX.y
    )

    # RIGHT FOOT COORDINATES
    RIGHT_HEEL = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_HEEL.value)
    RIGHT_FOOT_INDEX = get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
    )
    RIGHT_FOOT_MIDPOINT = helpers.calculate_midpoint(
        RIGHT_HEEL.x, RIGHT_FOOT_INDEX.x, RIGHT_HEEL.y, RIGHT_FOOT_INDEX.y
    )

    # Calculate angle for both feet
    LEFT_FOOT_ANGLE = helpers.calculate_flatfoot(
        LEFT_HEEL, LEFT_FOOT_MIDPOINT, LEFT_FOOT_INDEX
    )
    RIGHT_FOOT_ANGLE = helpers.calculate_flatfoot(
        RIGHT_HEEL, RIGHT_FOOT_MIDPOINT, RIGHT_FOOT_INDEX
    )

    try:
        left_foot_landed, right_foot_landed = helpers.which_foot_landed(
            landmarks, height, threshold_pixel_y
        )
    except Exception as e:
        left_foot_landed = False
        right_foot_landed = False

    if left_foot_landed:
        distance = helpers.calculate_distance(
            [LEFT_HEEL.x, LEFT_HEEL.y], [RIGHT_FOOT_INDEX.x, RIGHT_FOOT_INDEX.y]
        )

        angle = helpers.calculate_angle_with_threshold(
            (LEFT_HEEL.x, LEFT_HEEL.y),
            (LEFT_FOOT_INDEX.x, LEFT_FOOT_INDEX.y),
            threshold_pixel_y,
        )

        angle = 180 - angle

        narrow_foot = distance <= 0.01
        notflatfooted = angle <= 178

        # print(f"Angle between heel and foot index with threshold: {angle:.2f}")

    if right_foot_landed:
        distance = helpers.calculate_distance(
            [RIGHT_HEEL.x, RIGHT_HEEL.y], [LEFT_FOOT_INDEX.x, LEFT_FOOT_INDEX.y]
        )

        angle = helpers.calculate_angle_with_threshold(
            (LEFT_HEEL.x, LEFT_HEEL.y),
            (LEFT_FOOT_INDEX.x, LEFT_FOOT_INDEX.y),
            threshold_pixel_y,
        )

        angle = 180 - angle

        narrow_foot = distance <= 0.01
        notflatfooted = angle <= 178

        # print(f"Angle between heel and foot index with threshold: {angle:.2f}")

    # print(f"angle: {angle}")
    # print(f"narrow foot: {narrow_foot}")
    # print(f"flat footed: {notflatfooted}")

    # return narrow_foot and notflatfooted
    return narrow_foot


def criteria_4(
    landmarks,
    mp_pose: mp.solutions.pose,
    height,
    threshold_pixel_y,
) -> bool:
    """Check if which leg has its foot landed. The other leg is used
    to calculate if its degree is less than approximately 90 degrees
    indicating nonsupport leg bent.

    Parameters
    ----------
    landmarks : list
        List of landmarks returned from Mediapipe's Pose Estimation
    mp_pose : mp.solutions.pose
        Index for each landmark

    Returns
    -------
    bool
        tuple[bool, bool, bool]
        Booleans indicating if the left or right foot is landed
        and if nonsupport leg is met
    """

    left_foot_landed = False
    right_foot_landed = False
    nonsupport_bent = False
    LEFT_LEG_ANGLE = 0.0
    RIGHT_LEG_ANGLE = 0.0

    LEFT_HIP = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
    LEFT_KNEE = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value)
    LEFT_ANKLE = get_landmark_point(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value)

    RIGHT_HIP = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
    RIGHT_KNEE = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value)
    RIGHT_ANKLE = get_landmark_point(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value)

    try:
        left_foot_landed, right_foot_landed = helpers.which_foot_landed(
            landmarks, height, threshold_pixel_y
        )
    except Exception as e:
        nonsupport_bent = False
        left_foot_landed = False
        right_foot_landed = False
        # print(f"Error: {e}")

    if left_foot_landed:
        RIGHT_LEG_ANGLE = helpers.calculate_angle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE)
        # print(f"RIGHT LEG ANGLE: {RIGHT_LEG_ANGLE}")
        nonsupport_bent = RIGHT_LEG_ANGLE <= 96
    elif right_foot_landed:
        LEFT_LEG_ANGLE = helpers.calculate_angle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE)
        # print(f"LEFT LEG ANGLE: {LEFT_LEG_ANGLE}")
        nonsupport_bent = int(LEFT_LEG_ANGLE) <= 96
    else:
        nonsupport_bent = False

    return (
        nonsupport_bent,
        left_foot_landed,
        right_foot_landed,
        LEFT_LEG_ANGLE,
        RIGHT_LEG_ANGLE,
    )
