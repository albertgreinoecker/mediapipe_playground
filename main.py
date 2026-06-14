import cv2
import mediapipe as mp
import numpy as np

# ===============================
# BODY LANDMARK GROUPS  (MediaPipe Pose – 33 Punkte)
# ===============================

BODY_LANDMARKS = {
    "FACE":           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "LEFT_ARM":       [11, 13, 15, 17, 19, 21],
    "RIGHT_ARM":      [12, 14, 16, 18, 20, 22],
    "TORSO":          [11, 12, 23, 24],
    "LEFT_LEG":       [23, 25, 27, 29, 31],
    "RIGHT_LEG":      [24, 26, 28, 30, 32],
}

# Verbindungen für das Skelett (Paare von Landmark-Indizes)
BODY_CONNECTIONS = [
    # Schultern – Hüften
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Arme
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    # Beine
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]
# ===============================
# FACE LANDMARK GROUPS
# ===============================

FACE_EYES = {
    "LEFT_EYE":  [33, 133, 160, 159, 158, 144, 145, 153],
    "RIGHT_EYE": [263, 362, 387, 386, 385, 373, 374, 380],
}

FACE_LIPS = {
    "OUTER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "INNER": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
}

FACE_NOSE = {
    "TIP": 1,
    "BRIDGE": [6, 197, 195, 5],
}

FACE_EARS = {
    "LEFT_EAR":  [234, 93, 132],
    "RIGHT_EAR": [454, 323, 361],
}

FACE_FOREHEAD = {
    "CENTER": [9, 10, 151, 337, 299, 333, 298],
}

FACE_EYEBROWS = {
    "LEFT_BROW":  [70, 63, 105, 66, 107],
    "RIGHT_BROW": [336, 296, 334, 293, 300],
}

FACE_EYES_FULL = {
    "LEFT_EYE": [
        33, 7, 163, 144, 145, 153, 154, 155,
        133, 173, 157, 158, 159, 160, 161, 246
    ],
    "RIGHT_EYE": [
        263, 249, 390, 373, 374, 380, 381, 382,
        362, 398, 384, 385, 386, 387, 388, 466
    ],
}

FACE_NOSE_FULL = {
    "NOSE": [
        1, 2, 98, 327, 326, 5, 195, 197, 6,
        168, 197, 195, 5
    ],
}

FACE_MOUTH = {
    "MOUTH": [
        61, 185, 40, 39, 37, 0, 267, 269,
        270, 409, 291
    ],
}

FACE_CHEEKS = { #Wange
    "LEFT_CHEEK":  [50, 187, 207, 216, 192],
    "RIGHT_CHEEK": [280, 411, 427, 436, 416],
}

FACE_JAW = {
    "JAWLINE": [
        10, 338, 297, 332, 284, 251, 389,
        356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176,
        149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67,
        109
    ],
}

FACE_CHIN = {
    "CHIN": [152, 377, 400, 378, 379, 365, 397, 288, 361],
}

ACE_OVAL = {
    "FACE_OUTLINE": [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ],
}


# ===============================
# HAND LANDMARK GROUPS
# ===============================

HAND_LANDMARKS = {
    "WRIST":  [0],
    "THUMB":  [1, 2, 3, 4],
    "INDEX":  [5, 6, 7, 8],
    "MIDDLE": [9, 10, 11, 12],
    "RING":   [13, 14, 15, 16],
    "PINKY":  [17, 18, 19, 20],
}

INDEX_TIP = 8
INDEX_BOTTOM = 5

PINKY_TIP = 20
PINKY_BOTTOM = 17

# ===============================
# MEDIAPIPE INIT
# ===============================

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===============================
# DRAW HELPER
# ===============================

def draw_points(image, landmarks, indices, color):
    h, w, _ = image.shape
    for idx in indices:
        lm = landmarks.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 2, color, -1)

def draw_connections(image, landmarks, connections, color):
    h, w, _ = image.shape
    for a, b in connections:
        lm_a = landmarks.landmark[a]
        lm_b = landmarks.landmark[b]
        pt_a = (int(lm_a.x * w), int(lm_a.y * h))
        pt_b = (int(lm_b.x * w), int(lm_b.y * h))
        cv2.line(image, pt_a, pt_b, color, 2)

def draw_filled_area(image, landmarks, indices, color):
    h, w, _ = image.shape

    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts.append([cx, cy])

    pts = np.array([pts], dtype=np.int32)  # shape (1, N, 2)
    cv2.fillPoly(image, pts, color)

# ===============================
# CAMERA LOOP
# ===============================

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ---------- OpenCV → MediaPipe ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False

    face_results = face_mesh.process(rgb)
    hand_results = hands.process(rgb)
    pose_results = pose.process(rgb)

    rgb.flags.writeable = True

    # ---------- MediaPipe → OpenCV ----------
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # ========== FACE ==========
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:

            # for eye in FACE_EYES.values():
            #     draw_points(output, face_landmarks, eye, (255, 0, 0))

            draw_points(output, face_landmarks, FACE_LIPS["OUTER"], (0, 0, 255)) # (B, G, R)
            draw_points(output, face_landmarks, FACE_LIPS["INNER"], (0, 128, 255))

            draw_points(output, face_landmarks, FACE_NOSE["BRIDGE"], (0, 255, 255))
            draw_points(output, face_landmarks, [FACE_NOSE["TIP"]], (0, 255, 255))

            draw_points(output, face_landmarks, FACE_EARS["LEFT_EAR"], (255, 255, 0))
            draw_points(output, face_landmarks, FACE_EARS["RIGHT_EAR"], (255, 255, 0))

    # ========== HANDS ==========
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for finger in HAND_LANDMARKS.values():
                draw_points(output, hand_landmarks, finger, (0, 255, 0))

            dist_tip_finger = face_landmarks.landmark[INDEX_BOTTOM].y - hand_landmarks.landmark[INDEX_TIP].y
            if dist_tip_finger > 0.3:
                draw_filled_area(output, face_landmarks, FACE_EYES_FULL["LEFT_EYE"], (0, 0, 255))
                draw_filled_area(output, face_landmarks, FACE_EYES_FULL["RIGHT_EYE"], (0, 0, 255))

            dist_tip_finger = hand_landmarks.landmark[PINKY_BOTTOM].y - hand_landmarks.landmark[PINKY_TIP].y
            if dist_tip_finger > 0.4:
                draw_points(output, face_landmarks, FACE_JAW["JAWLINE"], (0, 255, 255))

    # ========== BODY / POSE ==========
    if pose_results.pose_landmarks:
        lms = pose_results.pose_landmarks
        draw_connections(output, lms, BODY_CONNECTIONS, (200, 200, 200))
        draw_points(output, lms, BODY_LANDMARKS["LEFT_ARM"],  (0, 255, 128))
        draw_points(output, lms, BODY_LANDMARKS["RIGHT_ARM"], (0, 128, 255))
        draw_points(output, lms, BODY_LANDMARKS["TORSO"],     (255, 200, 0))
        draw_points(output, lms, BODY_LANDMARKS["LEFT_LEG"],  (0, 255, 128))
        draw_points(output, lms, BODY_LANDMARKS["RIGHT_LEG"], (0, 128, 255))

    cv2.imshow("MediaPipe Face, Hands & Body (0.10.31)", output)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
