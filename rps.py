
import cv2
import numpy as np
import random
import urllib.request
import os
import ssl


MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("Downloading hand_landmarker.task  (first run only)...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    with urllib.request.urlopen(MODEL_URL, context=ctx) as r, \
         open(MODEL_PATH, "wb") as f:
        f.write(r.read())
    print("Download complete!")

# ══════════════════════════════════════════
#  2. MediaPipe Tasks setup
# ══════════════════════════════════════════
from mediapipe.tasks            import python
from mediapipe.tasks.python     import vision
from mediapipe                  import Image, ImageFormat
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode

options = HandLandmarkerOptions(
    base_options        = python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode        = RunningMode.VIDEO,
    num_hands           = 1,
    min_hand_detection_confidence = 0.7,
    min_hand_presence_confidence  = 0.7,
    min_tracking_confidence       = 0.7,
)
detector = vision.HandLandmarker.create_from_options(options)

# ══════════════════════════════════════════
#  3. Finger counter
# ══════════════════════════════════════════
TIP_IDS = [8, 12, 16, 20]
PIP_IDS = [6, 10, 14, 18]

def count_fingers(landmarks, handedness_label):
    fingers = 0
    for tip, pip in zip(TIP_IDS, PIP_IDS):
        if landmarks[tip].y < landmarks[pip].y:
            fingers += 1
    if handedness_label == "Right":
        if landmarks[4].x < landmarks[3].x:
            fingers += 1
    else:
        if landmarks[4].x > landmarks[3].x:
            fingers += 1
    return fingers

def fingers_to_gesture(n):
    if n == 0:   return "Rock"
    if n == 2:   return "Scissors"
    if n >= 4:   return "Paper"
    return "Unknown"

# ══════════════════════════════════════════
#  4. Game logic
# ══════════════════════════════════════════
choices = ["Rock", "Paper", "Scissors"]
WINS    = {("Rock","Scissors"), ("Paper","Rock"), ("Scissors","Paper")}

player_score   = 0
computer_score = 0
player_move    = "No Hand"
computer_move  = "---"
result_text    = "Press SPACE to Play"

def get_result(player, computer):
    global player_score, computer_score
    if player == computer:
        return "Draw!"
    if (player, computer) in WINS:
        player_score += 1
        return "You Win!"
    computer_score += 1
    return "Computer Wins!"

# ══════════════════════════════════════════
#  5. HUD helper
# ══════════════════════════════════════════
def put_text(img, text, pos, scale=0.9, color=(255,255,255), thick=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thick + 2)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick)

# ══════════════════════════════════════════
#  6. Main loop
# ══════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # ── MediaPipe detection ──────────────
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_img, timestamp_ms)
    timestamp_ms += 33

    detected = "No Hand"

    if result.hand_landmarks:
        lm    = result.hand_landmarks[0]
        label = result.handedness[0][0].display_name
        n     = count_fingers(lm, label)
        detected = fingers_to_gesture(n)

        # draw skeleton
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),
            (0,17)
        ]
        pts = [(int(p.x * w), int(p.y * h)) for p in lm]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 220, 130), 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 5, (0, 180, 100),   2)

    player_move = detected

    # ── HUD ─────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 190), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    put_text(frame, f"Your Gesture : {player_move}",
             (15, 40),  scale=1.0, color=(80, 255, 100))
    put_text(frame, f"Computer     : {computer_move}",
             (15, 82),  scale=1.0, color=(100, 160, 255))

    rc = (0,255,160)  if "You Win"   in result_text else \
         (60, 60,255) if "Computer"  in result_text else (240,240,80)
    put_text(frame, result_text, (15, 124), scale=1.0, color=rc)
    put_text(frame, f"Score  You: {player_score}   AI: {computer_score}",
             (15, 164), scale=0.85, color=(210, 210, 210))
    put_text(frame, "SPACE = Play    ESC = Quit",
             (15, h - 15), scale=0.7, color=(160, 160, 160))

    cv2.imshow("Rock Paper Scissors  |  MediaPipe 0.10+", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 32:
        if player_move not in ("No Hand", "Unknown"):
            computer_move = random.choice(choices)
            result_text   = get_result(player_move, computer_move)
        else:
            result_text = "Show a clear hand gesture!"
    elif key == 27:
        break


detector.close()
cap.release()
cv2.destroyAllWindows()
