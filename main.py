import cv2 as cv
import mediapipe as mp
import time
import utils
import math
import numpy as np

# variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0
# constants
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE_CENTER_LINE = [10, 151, 9, 8, 168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
LEFT_PUPIL_POINT = 468
LEFT_IRIS = [469,470,471,472]
LEFT_KEY_POINTS = [362, 263, 9, 8] #lewo, prawo, góra, dół

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_PUPIL_POINT = 473
RIGHT_IRIS = [474, 475, 476, 477]
RIGHT_KEY_POINTS = [33, 133, 9, 8] #lewo, prawo, góra, dół

map_face_mesh = mp.solutions.face_mesh
# camera object
camera = cv.VideoCapture(0)


# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord


# Euclaidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

# Distance ratio of x1->x2 length to x3->x4 length
def distanceRatio(x1, x2, x3, x4):
    return (np.abs(x1 - x2) * 1.0) / np.abs(x3 - x4)

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio


# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # getting the dimension of image
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask
    # cv.imshow('mask', mask)

    # draw eyes image on mask, where white shape is
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys
    # cv.imshow('eyes draw', eyes)
    eyes[mask == 0] = 155

    # getting minium and maximum x and y  for right and left eyes
    # For Right Eye
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes
    return cropped_right, cropped_left


# Eyes Postion Estimator
def positionEstimator(cropped_eye):
    # getting height and width of eye
    h, w = cropped_eye.shape

    # remove the noise from images
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # create fixd part for eye with
    piece = int(w / 3)

    # slicing the eyes into three parts
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    # calling pixel counter function
    eye_position, color = pixelCounter(left_piece, center_piece, right_piece)   #jako ze zrobilem na kamerze lustrzane odbicie, w tym miejscu zamienilem kolejnoscia left z right i jest git

    return eye_position, color


# creating pixel counter function
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part] #strzelam ze tutaj bedzie trzeba dodac logike gora/dol dodatkowo

    # getting the index of max values in the list
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''
    if max_index == 0:                               #i potem pewnie tutaj dodoac
        pos_eye = "RIGHT"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color


with (map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True) as face_mesh):
    # starting time here
    start_time = time.time()
    # starting Video loop here.
    calibration_cnt = 0
    is_calibrated = False
    left_eye_calibration = []
    right_eye_calibration = []
    quit_condition = False
    while not quit_condition:
        frame_counter += 1  # frame counter
        ret, frameBeforeFlip = camera.read()  # getting frame from camera
        if not ret:
            break  # no more frames break
        #  resizing frame
        frame = cv.flip(frameBeforeFlip, 1)   #odwrocilem kamere bo jest przejrzysciej i spowodowalem komplikacje, juz opanowane
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # mp_face_detection = mp.solutions.face_detection
        # face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        # results = face_detection.process(rgb_frame)
        # if results.detections:
        #     frame_height, frame_width = frame.shape[:2]
        #     test = results.detections
        #     bounding_box = results.detections[0].location_data.relative_bounding_box
        #     frame = frame[int(bounding_box.xmin * frame_width):int((bounding_box.xmin+bounding_box.width) * frame_width),
        #             int(bounding_box.ymin * frame_height):int((bounding_box.ymin + bounding_box.height) * frame_height)]
        #     frame = cv.resize(frame, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)

        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            test = results.multi_face_landmarks
            mesh_coords = landmarksDetection(frame, results, False)
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
            utils.colorBackgroundText(frame, f'Ratio : {round(ratio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                      utils.YELLOW)

            if ratio > 5.5:
                CEF_COUNTER += 1
                # cv.putText(frame, 'Blink', (200, 50), FONTS, 1.3, utils.PINK, 2)
                utils.colorBackgroundText(frame, f'Blink', FONTS, 1.7, (int(frame_height / 2), 100), 2, utils.YELLOW,
                                          pad_x=6, pad_y=6, )

            else:
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    CEF_COUNTER = 0
            # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)
            utils.colorBackgroundText(frame, f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30, 150), 2)

            right_eye_coords = [mesh_coords[p] for p in RIGHT_EYE]
            left_eye_coords = [mesh_coords[p] for p in LEFT_EYE]
            cv.polylines(frame, [np.array(left_eye_coords, dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array(right_eye_coords, dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)

            right_iris_coords = [mesh_coords[p] for p in LEFT_IRIS]
            left_iris_coords = [mesh_coords[p] for p in RIGHT_IRIS]
            right_pupil_coords = mesh_coords[LEFT_PUPIL_POINT]
            left_pupil_coords = mesh_coords[RIGHT_PUPIL_POINT]
            cv.polylines(frame, [np.array(right_iris_coords,dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.polylines(frame, [np.array(left_iris_coords, dtype=np.int32)], True, utils.GREEN, 1,
                         cv.LINE_AA)
            cv.circle(frame, right_pupil_coords, 1, utils.GREEN)
            cv.circle(frame, left_pupil_coords, 1, utils.GREEN)

            # face_oval_coords = [mesh_coords[p] for p in FACE_OVAL]
            # cv.polylines(frame, [np.array(face_oval_coords, dtype=np.int32)], True, utils.GREEN, 1,
            #              cv.LINE_AA)

            # nose_center_line_coords = [mesh_coords[p] for p in NOSE_CENTER_LINE]
            # for i in nose_center_line_coords:
            #     cv.circle(frame, i, 5, utils.GREEN)

            # mean_coords = np.asarray([np.mean(nose_center_line_coords[0:7], axis=0),
            #                           np.mean(nose_center_line_coords[7:14], axis=0)], dtype=np.int16)
            # nose_center_line_mean_coords = [tuple(p) for p in nose_center_line_mean_coords]
            # for i in mean_coords:
            #     cv.circle(frame, i, 5, utils.RED)

            # right_mean_coords = np.asarray([np.mean([mesh_coords[p] for p in [223,27]], axis=0),
            #                           np.mean([mesh_coords[p] for p in [23,230]], axis=0)], dtype=np.int16)
            # left_mean_coords = np.asarray([np.mean([mesh_coords[p] for p in [443, 257]], axis=0),
            #                               np.mean([mesh_coords[p] for p in [253, 450]], axis=0)], dtype=np.int16)
            right_mean_coords = np.asarray([np.mean([mesh_coords[p] for p in [2,94]], axis=0),
                                           np.mean([mesh_coords[p] for p in [168, 6]], axis=0)], dtype=np.int16)
            left_mean_coords = right_mean_coords

            for i in left_mean_coords:
                cv.circle(frame, i, 5, utils.RED)
            for i in right_mean_coords:
                cv.circle(frame, i, 5, utils.RED)

            right_key_points_coords = [mesh_coords[p] for p in RIGHT_KEY_POINTS]
            left_key_points_coords = [mesh_coords[p] for p in LEFT_KEY_POINTS]

            # RIGHT_TEST = [189, 221, 222, 223, 224, 225, 113, 226, 31, 228, 229, 230, 231, 232, 233, 244]
            # right_test_coords = [mesh_coords[p] for p in RIGHT_TEST]
            # LEFT_TEST = [413, 441, 442, 443, 444, 445, 342, 446, 261, 448, 449, 450, 451, 452, 453, 464]
            # left_test_coords = [mesh_coords[p] for p in LEFT_TEST]
            # cv.polylines(frame, [np.array(right_test_coords, dtype=np.int32)], True, utils.GREEN, 1,
            #              cv.LINE_AA)
            # cv.polylines(frame, [np.array(left_test_coords, dtype=np.int32)], True, utils.GREEN, 1,
            #              cv.LINE_AA)

            # Blink Detector Counter Completed
            # crop_right, crop_left = eyesExtractor(frame, right_eye_coords, left_eye_coords)
            # cv.imshow('right', cv.resize(crop_right, None, fx=5, fy=5, interpolation=cv.INTER_CUBIC))
            # cv.imshow('left', cv.resize(crop_left, None, fx=5, fy=5, interpolation=cv.INTER_CUBIC))

            # Detecting eye position if calibrated
            if is_calibrated:
                right_distance = distanceRatio(right_key_points_coords[1][0], right_pupil_coords[0],
                                               right_key_points_coords[1][0],
                                               right_key_points_coords[0][0])
                if right_distance >= right_eye_calibration[0]:
                    eye_position = 'LEFT'
                    color = [utils.GRAY, utils.YELLOW]
                elif right_distance <= right_eye_calibration[1]:
                    eye_position = "RIGHT"
                    color = [utils.BLACK, utils.GREEN]
                else:
                    eye_position = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                utils.colorBackgroundText(frame, f'R: {round(right_distance,2)}, {eye_position}',
                                          FONTS, 1.0, (40, 220), 2, color[0], color[1], 8, 8)

                left_distance = distanceRatio(left_key_points_coords[0][0], left_pupil_coords[0],
                                              left_key_points_coords[0][0],
                                              left_key_points_coords[1][0])
                if left_distance <= left_eye_calibration[0]:
                    eye_position = 'LEFT'
                    color = [utils.GRAY, utils.YELLOW]
                elif left_distance >= left_eye_calibration[1]:
                    eye_position = "RIGHT"
                    color = [utils.BLACK, utils.GREEN]
                else:
                    eye_position = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                utils.colorBackgroundText(frame, f'L: {round(left_distance,2)}, {eye_position}',
                                          FONTS, 1.0, (40, 280), 2, color[0], color[1], 8, 8)

                right_distance = distanceRatio(right_mean_coords[0][1], right_pupil_coords[1],
                                               right_mean_coords[0][1],
                                               right_mean_coords[1][1])
                if right_distance >= right_eye_calibration[2]:
                    eye_position = 'UP'
                    color = [utils.GRAY, utils.YELLOW]
                elif right_distance <= right_eye_calibration[3]:
                    eye_position = "DOWN"
                    color = [utils.BLACK, utils.GREEN]
                else:
                    eye_position = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                utils.colorBackgroundText(frame, f'R: {round(right_distance,2)}, {eye_position}',
                                          FONTS, 1.0, (40, 380), 2, color[0], color[1], 8, 8)

                left_distance = distanceRatio(left_mean_coords[0][1], left_pupil_coords[1],
                                              left_mean_coords[0][1],
                                              left_mean_coords[1][1])
                if left_distance <= left_eye_calibration[2]:
                    eye_position = 'UP'
                    color = [utils.GRAY, utils.YELLOW]
                elif left_distance >= left_eye_calibration[3]:
                    eye_position = "DOWN"
                    color = [utils.BLACK, utils.GREEN]
                else:
                    eye_position = 'CENTER'
                    color = [utils.YELLOW, utils.PINK]
                utils.colorBackgroundText(frame, f'L: {round(left_distance,2)}, {eye_position}',
                                          FONTS, 1.0, (40, 440), 2, color[0], color[1], 8, 8)

        # calculating  frame per seconds FPS
        end_time = time.time() - start_time
        fps = frame_counter / end_time
        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                         textThickness=2)
        frame = utils.textWithBackground(frame, f'cnt: {calibration_cnt}', FONTS, 1.0, (200, 50), bgOpacity=0.9,
                                         textThickness=2)

        # writing image for thumbnail drawing shape
        # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv.imshow('frame', frame)
        key = cv.waitKey(2)
        if key == ord('q') or key == ord('Q'):
            quit_condition = True
        if not is_calibrated and results.multi_face_landmarks and (key == ord('c') or key == ord('C')):
            if calibration_cnt < 2:
                right_eye_calibration.append(distanceRatio(right_key_points_coords[1][0], right_pupil_coords[0],
                                                           right_key_points_coords[1][0],
                                                           right_key_points_coords[0][0]))
                left_eye_calibration.append(distanceRatio(left_key_points_coords[0][0], left_pupil_coords[0],
                                                          left_key_points_coords[0][0],
                                                          left_key_points_coords[1][0]))
            else:
                right_eye_calibration.append(distanceRatio(right_mean_coords[0][1], right_pupil_coords[1],
                                                           right_mean_coords[0][1],
                                                           right_mean_coords[1][1]))
                left_eye_calibration.append(distanceRatio(left_mean_coords[0][1], left_pupil_coords[1],
                                                          left_mean_coords[0][1],
                                                          left_mean_coords[1][1]))
            calibration_cnt += 1
            if calibration_cnt >= 4:
                is_calibrated = True
        if key == ord('r') or key == ord('R'):
            is_calibrated = False
            calibration_cnt = 0
            right_eye_calibration.clear()
            left_eye_calibration.clear()

    cv.destroyAllWindows()
    camera.release()