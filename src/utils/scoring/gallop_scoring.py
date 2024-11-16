"""
NOTE: Print statements are for debugging purposes.


Script for calculating GALLOP skill based on
TGMD-2 Performance Criteria Checklist.
"""

import mediapipe as mp

from utils.scoring import helpers


def criteria_1(landmarks, mp_pose: mp.solutions.pose) -> bool:
    """Checks whether arms are bent by determining elbow angle
    approx. to 90 degrees. Also checks if elbow is above hips

    Parameters
    ----------
    landmarks : List
        List of landmarks returned from Mediapipe's Pose Estimation
    mp_pose : mp.solutions.pose
        Index for each landmark

    Returns
    -------
    bool
        Return True if arms are bent approx. to 90 degrees and are
        above the hip.
    """
    # GET LEFT AND RIGHT HIP LANDMARKS
    LEFT_HIP = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_HIP.value
    )
    RIGHT_HIP = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value
    )

    # EXTRACT LEFT ARM COORDINATES
    LEFT_SHOULDER = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value
    )
    LEFT_ELBOW = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value
    )
    LEFT_WRIST = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value
    )

    # EXTRACT RIGHT ARM COORDINATES
    RIGHT_SHOULDER = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    )
    RIGHT_ELBOW = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value
    )
    RIGHT_WRIST = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value
    )

    # Calculate angles left and right elbows
    LEFT_ELBOW_ANGLE = helpers.calculate_angle(
        LEFT_SHOULDER,
        LEFT_ELBOW,
        LEFT_WRIST,
    )

    RIGHT_ELBOW_ANGLE = helpers.calculate_angle(
        RIGHT_SHOULDER,
        RIGHT_ELBOW,
        RIGHT_WRIST,
    )

    # Checks whether elbows are above the hips
    LEFT_ELBOW_ABOVE = LEFT_HIP > LEFT_ELBOW
    RIGHT_ELBOW_ABOVE = RIGHT_HIP > RIGHT_HIP

    # Checks whether arms are less than or equal to 90 degrees
    LEFT_ELBOW_BENT = LEFT_ELBOW_ANGLE <= 90.0
    RIGHT_ELBOW_BENT = RIGHT_ELBOW_ANGLE <= 90.0

    # print(f"left elbow angle: {LEFT_ELBOW_ANGLE}")
    # print(f"right elbow angle: {RIGHT_ELBOW_ANGLE}")

    # print(f"left elbow above hip: {LEFT_ELBOW_ABOVE}")
    # print(f"right elbow above hip: {RIGHT_ELBOW_ABOVE}")

    if helpers.detect_facing_direction(landmarks, mp_pose) == "left":
        if LEFT_ELBOW_ABOVE and LEFT_ELBOW_BENT:
            return True
    elif helpers.detect_facing_direction(landmarks, mp_pose) == "right":
        if RIGHT_ELBOW_ABOVE and RIGHT_ELBOW_BENT:
            return True


def criteria_2(
    landmarks,
    mp_pose: mp.solutions.pose,
    height: int,
    width: int,
    direction: str,
    threshold_pixel_y,
) -> bool:
    """First checks what direction the subject is facing, if it is facing left,
    then the foot with the lower x-value pixel-wise is consider the leading foot,
    otherwise the foot with the higher x-value is considered. Determine if the leading
    foot landed depending on the threshold line followed by the trailing foot.

    Parameters
    ----------
    landmarks : List
        List of landmarks returned from Mediapipe's Pose Estimation
    mp_pose : mp.solutions.pose
        Index for each landmark
    direction: str
        Direction at which the subject is facing

    Returns
    -------
    bool
        True if the following criteria is met
    """

    # LEFT FOOT COORDINATES
    LEFT_HEEL = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_HEEL.value
    )
    LEFT_FOOT_INDEX = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    )

    # RIGHT FOOT COORDINATES
    RIGHT_HEEL = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_HEEL.value
    )
    RIGHT_FOOT_INDEX = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value
    )

    # left_heel_pos = (int(LEFT_HEEL.x * width), int(LEFT_HEEL.y * height))
    # left_foot_index_pos = (
    #     int(LEFT_FOOT_INDEX.x * width),
    #     int(LEFT_FOOT_INDEX.y * height),
    # )

    # right_heel_pos = (int(RIGHT_HEEL.x * width), int(RIGHT_HEEL.y * height))
    # right_foot_index_pos = (
    #     int(RIGHT_FOOT_INDEX.x * width),
    #     int(RIGHT_FOOT_INDEX.y * height),
    # )

    leading, trailing = helpers.determine_lead_foot(
        LEFT_FOOT_INDEX,
        RIGHT_FOOT_INDEX,
        direction,
    )

    leading_foot_landed = None
    trailing_foot_landed = None

    if int(leading.y * height) > threshold_pixel_y:
        leading_foot_landed = True
    else:
        leading_foot_landed = False

    if int(trailing.y * height) > threshold_pixel_y:
        trailing_foot_landed = True
    else:
        trailing_foot_landed = False

    return leading_foot_landed and trailing_foot_landed


def criteria_3(landmarks, mp_pose: mp.solutions.pose, height) -> bool:
    """Checks if both feet are off the ground by determining the
    threshold (ground) line using the maximum y-value of detected
    flat-footed. If both y-axis for both feet are below the threshold
    y-pixel value, it returns True

    Parameters
    ----------
    landmarks : List
        List of landmarks returned from Mediapipe's Pose Estimation
    mp_pose : mp.solutions.pose
        Index for each landmark

    Returns
    -------
    bool
        Returns True if both feet are above the threshold line.
    """

    threshold_pixel_y = None

    # LEFT FOOT COORDINATES
    LEFT_HEEL = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_HEEL.value
    )
    LEFT_FOOT_INDEX = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value
    )
    LEFT_FOOT_MIDPOINT = helpers.calculate_midpoint(
        LEFT_HEEL.x, LEFT_FOOT_INDEX.x, LEFT_HEEL.y, LEFT_FOOT_INDEX.y
    )

    # RIGHT FOOT COORDINATES
    RIGHT_HEEL = helpers.get_landmark_point(
        landmarks, mp_pose.PoseLandmark.RIGHT_HEEL.value
    )
    RIGHT_FOOT_INDEX = helpers.get_landmark_point(
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

    # Determines the threshold (ground) line
    if left_foot_flat and right_foot_flat:
        # Compare y-values
        left_y = int(LEFT_FOOT_INDEX[1] * height)
        right_y = int(RIGHT_FOOT_INDEX[1] * height)

        # Check if feet are roughly at the same y-coordinate (consider precision)
        if abs(left_y - right_y) <= 5:  # Precision threshold for pixel difference
            # Choose the forward foot (larger x-value)
            if LEFT_FOOT_INDEX[0] > RIGHT_FOOT_INDEX[0]:
                threshold_pixel_y = left_y
            else:
                threshold_pixel_y = right_y
        else:
            # If feet are not the same y-value, choose the maximum value as the threshold
            threshold_pixel_y = max(left_y, right_y)

    elif left_foot_flat:
        threshold_pixel_y = int(LEFT_FOOT_INDEX[1] * height)

    elif right_foot_flat:
        threshold_pixel_y = int(RIGHT_FOOT_INDEX[1] * height)

    both_feet_off_ground = False

    # Checks whether both feet are off the ground
    if threshold_pixel_y is not None:
        both_feet_off_ground = (LEFT_HEEL[1] * height < threshold_pixel_y) and (
            RIGHT_HEEL[1] * height < threshold_pixel_y
        )

    return both_feet_off_ground, threshold_pixel_y


def criteria_4(landmarks, mp_pose: mp.solutions.pose) -> bool:
    raise NotImplementedError
