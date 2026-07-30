#!/usr/bin/python3
# coding=utf8
"""
Line Follow -> Pick Object -> Return -> Place
================================================
Run this ON THE PI.

This is a direct merge of:
  - The LINE FOLLOWING loop (LAB threshold on a lower ROI strip +
    center-offset steering) from the earlier line-following script.
  - The exact DETECTION + PICK logic from Pick_and_Place.py
    (ColorDetect / camera_to_world / getAreaMaxContour, with the same
    position-stability check using num / old_x / old_y).

FLOW:
  1. Robot follows the RED line.
  2. Every frame it also checks for the TARGET_COLOR object (ColorDetect,
     same as Pick_and_Place.py). The moment the object comes into view,
     the robot STOPS walking and just watches until the position is
     stable for STABLE_FRAMES frames (this is what avoids the earlier bug
     of walking past the object before picking).
  3. Once stable -> picks it up (with the -2/-6 offset correction).
  4. Turns ~180 degrees, walks back the same number of forward steps it
     took while following the line (so it returns to roughly the start).
  5. Places the object down at the fixed PLACE_LOCATION.

================================================================
BUGFIX NOTES (boundary-search rightward-drift issue)
================================================================
Two functions were modified: locate_place_zone() and
navigate_and_place_in_boundary(). Every changed line is marked with
a `# NEW`, `# MODIFIED`, or `# REMOVED` comment plus an explanation.
No other logic (detection, line following, pick logic, architecture)
was touched. Summary of the root cause and fix:

  1. `search_direction` and `sweep_turns_done` used to be local
     variables inside locate_place_zone(), reset to 1 / 0 on every
     single call. navigate_and_place_in_boundary() calls
     locate_place_zone() fresh on every creep attempt, so every
     attempt always started by sweeping RIGHT first. That's the
     direct cause of the observed rightward bias.
     FIX: search_direction is now owned by navigate_and_place_in_boundary()
     and threaded through locate_place_zone() as a parameter/return
     value, so it persists across creep attempts within one zone's
     search (but still starts fresh, unbiased, for each new zone).

  2. Search turns (ik.turn_right / ik.turn_left) physically rotate the
     robot chassis and nothing ever undid that rotation - not on a
     failed sweep (before creeping forward) and not on a successful
     lock (before computing camera_to_world(), which assumes a
     zero-rotation calibrated heading). This is what caused:
       - progressively compounding drift attempt after attempt,
       - a zone only ever being found from an extreme angle,
       - placement landing near an edge/corner instead of centered,
         because the world-coordinate math was computed from a
         rotated camera view.
     FIX: a `net_rotation` counter now tracks every search turn made
     during a locate_place_zone() call. Before returning a successful
     coordinate OR before returning from a failed timeout, that
     rotation is undone (turned back the opposite way) so the chassis
     heading returns to neutral before either computing world
     coordinates or creeping forward for the next attempt.

================================================================
BUGFIX NOTES ROUND 2 (both objects landing in "Zone 0")
================================================================
Traced against a log where Zone 1's tape was reached 7/8 (and 6/8,
4/8, 2/8 on other passes) required stable frames before a noisy
contour fragment reset the counter, every single attempt. Zone 1 was
being SEEN correctly almost the whole time -- it just never confirmed
stable before each attempt's frame budget ran out. This is a
detection-noise problem, not a search-direction or state-reset bug
(those were already fixed in round 1 and verified working: the
`DEBUG straightening heading by N deg` lines confirm net_rotation
correction is firing correctly).

The actual visible symptom ("both objects in Zone 0") came from a
DIFFERENT, smaller bug: when Zone 1 search exhausts all creep
attempts, it falls back to a single HARDCODED coordinate (0.0, 9.0)
cm shared by both zones. That fallback point happens to sit only
~3cm from where Zone 0 was actually detected earlier in the same
run (2.7, 10.2 cm) -- well within the gripper's placement footprint.
So the second object was never actually re-detecting Zone 0; it was
landing on a generic fallback point that coincidentally overlaps it.

Two additive, minimal fixes (no detection/line-following logic
touched):
  1. Each zone_index now gets its own distinct, separated fallback
     coordinate (ZONE_FALLBACK_POSITIONS) instead of one shared value.
  2. A `placed_zone_positions` list records the world coordinate of
     every successful/fallback placement. locate_place_zone() checks
     any new stable lock against this list and, if it's suspiciously
     close to an already-used position, treats it as a re-detection
     of that same physical zone and keeps searching instead of
     accepting it -- a safety net for if detection noise ever does
     produce a false lock in the future.
"""



import sys
sys.path.append('/home/pi/SpiderPi/')
import cv2
import math
import time
import Camera
import kinematics
import numpy as np
import yaml_handle
import apriltag  # NEW: SpiderPi SDK's built-in AprilTag detector (see ApriltagDetect.py),
                 # replaces the black-tape boundary detection for placement-zone finding
import ArmIK.ArmMoveIK as AMK
import HiwonderSDK.Misc as Misc
import HiwonderSDK.Board as Board
import threading  # NEW: needed for YoloServer's background accept/receive thread
import socket      # NEW: needed for YoloServer's listening socket
import struct      # NEW: needed for camera_stream_server()'s length-prefix framing (matches pc_yolo_sender.py's PiCameraReceiver protocol)

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)

ik = kinematics.IK()
AK = AMK.ArmIK()

# NEW: built-in SpiderPi SDK AprilTag detector (same setup as ApriltagDetect.py).
# Used for placement-zone detection (tag_id 0 -> zone 0, tag_id 1 -> zone 1)
# instead of the earlier black-tape boundary detection.
apriltag_detector = apriltag.Detector(searchpath=apriltag._get_demo_searchpath())

# ============================ CONFIG ====================================
LINE_COLOR     = 'red'            # color of the line to follow

# Two objects to pick, in order. Both are picked from the END of the line.
# MODIFIED: these must now be YOLO CLASS NAMES (see classes_of_interest in
# YOLO_Camera.py: ['cup', 'cylinder', 'gluStick', 'box']), NOT color names.
# Detection is no longer color-based, so 'blue'/'green' have no meaning
# here anymore -- set this to the actual two classes you want picked, in
# the order you want them picked.
TARGET_SEQUENCE = ['BlueBox', 'GreenBox']  # <-- SET THIS to your real target classes

# NEW: port pc_yolo_sender.py's PiControlSender connects to (the Pi is the
# SERVER here -- see YoloServer below -- since PiControlSender dials out).
YOLO_DETECTION_PORT = 5001

# NEW (ROOT-CAUSE FIX): port pc_yolo_sender.py's PiCameraReceiver connects to
# in order to RECEIVE the robot's camera frames. This was present in the old
# standalone pi_pick_and_place.py (as camera_stream_server()) but was never
# carried over during integration -- meaning the PC-side YOLO script had no
# frames to run detection on at all, regardless of the detection-message
# protocol being correct. This port number must match pc_yolo_sender.py's
# CAMERA_STREAM_PORT (6000).
CAMERA_STREAM_PORT = 6000
DEBUG_STREAM_PORT = 6002  # NEW: streams the annotated display_frame (line/AprilTag boxes) to the GUI

IMG_CENTER_X   = 320
LINE_ROI       = [(200, 260, 0, 640)]   # lower strip of the frame used for line following

INITIAL_COORD  = (0, 15, 5)       # arm resting/scan pose
PLACE_Z        = -5.5             # height to lower to when placing (ground level)

# MODIFIED: placement-zone detection now uses the SpiderPi SDK's built-in
# AprilTag detector (tag_id 0 -> zone 0, tag_id 1 -> zone 1) instead of
# black-tape boundary detection -- see find_apriltag_zones(). The
# color/shape/ROI settings below (BOUNDARY_COLOR, BOUNDARY_MIN_AREA,
# BOUNDARY_ASPECT_MAX, BOUNDARY_SEARCH_ROI) were only used by the old
# find_black_boundaries()/classify_zone() and are UNUSED now. Left here
# untouched (not deleted) in case you ever want to roll back.
BOUNDARY_COLOR         = 'black'
BOUNDARY_MIN_AREA      = 200      # MODIFIED (was 300): a partial corner view of an
                                   # oversized boundary square has less visible tape
                                   # area than the full square, so the old threshold
                                   # could reject a valid but partial detection.
BOUNDARY_ASPECT_MAX    = 6.0      # MODIFIED (was 1.6): the boundary region can be
                                   # bigger than the camera's field of view, so often
                                   # only two adjoining edges (a corner) are visible
                                   # instead of the whole square -- that looks like an
                                   # elongated/L-shaped blob, not a clean square, and
                                   # used to get rejected here. Raised so a partial
                                   # corner/edge view still counts as a valid boundary
                                   # detection instead of requiring all 4 sides in frame.
BOUNDARY_STABLE_FRAMES = 8
BOUNDARY_MOVE_TOLERANCE = 10      # px -- how much the detected center can drift and still count as "stable"

# Restrict boundary search to this region (y1, y2, x1, x2) of the frame.
# This EXCLUDES the bottom/outer edges where the robot's own black legs
# are visible, so they don't get mistaken for the boundary tape.
# TUNE THESE based on where your legs actually appear in the camera view.
BOUNDARY_SEARCH_ROI = (0, 300, 120, 520)

# If the boundary isn't seen right away, the robot ACTIVELY SWEEPS
# left/right to search for it instead of just standing still staring at
# a fixed view (this is the missing piece -- previously it would just
# time out after 200 frames of seeing nothing, even if the tape was
# simply just outside the camera's current field of view).
BOUNDARY_SEARCH_TURN_EVERY  = 10   # try a search turn every N frames of "not seen"
BOUNDARY_SEARCH_TURN_DEG    = 10   # degrees per search turn
BOUNDARY_SEARCH_SWEEP_COUNT = 4    # how many turns before reversing sweep direction

# --- CREEP SEARCH (replaces the earlier wide-view-pose approach) ---
# The wide/shallow camera pose caused severe perspective distortion --
# the same physical box's contour aspect-ratio would swing between ~1.0
# and ~8+ frame to frame as the robot turned even slightly, causing the
# boundary to flicker in and out of the filter and the robot to oscillate.
# Instead, we keep using the SAME reliable, calibrated, steep INITIAL_COORD
# view (which was consistently stable and accurate whenever the boundary
# was actually within its narrower field of view) and search by physically
# creeping forward a little + sweeping again if not found, rather than by
# trying to see farther via a distorted camera angle.
BOUNDARY_MAX_CREEP_ATTEMPTS = 5     # how many "sweep here, then creep forward" cycles to try
BOUNDARY_SWEEP_FRAMES       = 80    # frames given to sweep-search at each creep position
BOUNDARY_CREEP_STEPS        = 2     # UNUSED now (see fix below) -- small forward steps that used
                                     # to be taken between creep attempts. Left here, not deleted,
                                     # per the "leave superseded config for rollback" convention
                                     # already used elsewhere in this file (see BOUNDARY_COLOR etc.)

# =========================================================================
# FIX (leg-vs-box collision): the destination AprilTags sit on lightweight
# cardboard boxes. The search above used to sweep by TURNING THE CHASSIS
# (ik.turn_right/ik.turn_left) and, between attempts, by WALKING FORWARD
# (ik.go_forward) -- both move the legs, and the legs were bumping the
# boxes out of position before placement.
#
# FIX: once the robot is back at the start position, the destination-tag
# search now sweeps the ARM (which carries the wrist-mounted camera) left
# and right via the same AK.setPitchRangeMoving() IK path already used
# everywhere else in this file (pick_object/place_object), instead of
# turning/walking. The legs are never commanded during this search, so
# they can't nudge the boxes. See locate_place_zone() and
# navigate_and_place_in_boundary() below for the specific changes
# (tagged # NEW-ARM-SEARCH / # MODIFIED-ARM-SEARCH).
# =========================================================================
ARM_SEARCH_X_STEP    = 3     # cm the arm's x moves per sweep step
ARM_SEARCH_X_MAX     = 8     # cm max +/- offset from INITIAL_COORD's x the arm will sweep to
ARM_SEARCH_STEP_EVERY = 10   # try an arm sweep step every N frames of "not seen" (mirrors the
                              # old BOUNDARY_SEARCH_TURN_EVERY timing)
ARM_SEARCH_MOVETIME  = 350   # ms for the arm to reach each sweep step's pose
ARM_SEARCH_SETTLE_S  = 0.25  # time to let the arm/camera physically settle before trusting a new frame
ARM_SEARCH_Z_LIFT    = 7     # cm added to INITIAL_COORD's z ONLY while sweeping (arm clears the legs
                              # while panning). The lock/recenter poses still use plain INITIAL_COORD
                              # (unchanged z) so the world-coordinate calibration stays valid.
ARM_SEARCH_WIDEN_STEP = 4    # cm the max sweep range grows by on each failed attempt (in place of
                              # the old forward-creep, since the legs can no longer move)


GRIPPER_OPEN   = 120              # servo 25 pulse: open
GRIPPER_CLOSE  = 500              # servo 25 pulse: closed on object
STABLE_FRAMES  = 10               # consecutive stable frames before triggering pickup

# CORRECTED: the camera is mounted on the WRIST (servo 24), NOT on the
# gripper (servo 25). Servo 25 only opens/closes the gripper's fingers --
# rotating it does not move the camera at all, so the previous approach of
# pulsing servo 25 to get a "second look" never actually changed the view.
#
# Servo 24 is the first joint of the 4-servo IK chain (24, 23, 22, 21) that
# ArmIK.ArmMoveIK moves TOGETHER to hold the end-effector at a given (x, y, z)
# and pitch. Calling Board.setBusServoPulse(24, ...) directly would move the
# wrist physically but the IK model would have no idea it moved -- the next
# setPitchRangeMoving() call computes servo angles from where the arm SHOULD
# be, not where it actually is, so it would silently produce wrong poses
# after that. So to genuinely tilt the wrist/camera (and get a second look),
# we go through the IK solver itself: call AK.setPitchRangeMoving() again at
# the SAME (x, y, z) but with a different pitch (alpha). This finds a new
# combination of servo24/23/22/21 that holds the same point at a different
# camera angle, and keeps the IK model in sync since it's the same official
# path pick_object()/place_object() already use.
#
# IMPORTANT: at INITIAL_COORD = (0, 15, 5) itself, the arm is already close
# to full extension, so the IK/servo-range math only allows pitch angles
# from about -90 deg to -76 deg (checked against ArmIK's actual solver) --
# a 14-degree window. Any alpha outside that (e.g. -45 or 0) has NO valid
# servo solution there, so setPitchRangeMoving() silently settles on
# whatever's closest to -90 within that narrow window -- which looks like
# "barely any tilt at all". Raising z opens up a much wider pitch range
# (at z=10 the window is roughly -90 to -63, ~27 degrees), so the tilt
# check uses a separate, slightly-higher coordinate instead of tilting in
# place at INITIAL_COORD.
#
# TUNE ON HARDWARE: verify CONFIRM_TILT_COORD/CONFIRM_TILT_ALPHA is still
# within reach for YOUR arm's actual link lengths (this was checked against
# the default ArmIK dimensions) -- watch the live preview and adjust z/alpha
# until the object is visibly seen from a different angle without leaving
# the frame or the arm colliding with anything.
CONFIRM_TILT_COORD    = (0, 15, 10)  # (x, y, z) used for the second-look pose
CONFIRM_TILT_ALPHA    = -65   # pitch (degrees) used for the second-look tilt via IK
CONFIRM_TILT_ALPHA1   = -90   # lower bound of pitch search range for the tilt
CONFIRM_TILT_ALPHA2   = -60   # upper bound of pitch search range for the tilt
CONFIRM_TILT_MOVETIME = 600   # ms for the arm to reach the tilted pose
CONFIRM_TILT_SETTLE_S = 0.6   # time to let the arm/camera physically settle before trusting a new frame
CONFIRM_TIMEOUT_S     = 3.0   # how long to wait for a confirming detection at the tilted angle
AREA_THRESHOLD = 500              # min contour area (px) to consider a valid object detection

# A "box" should look roughly compact (width approx. equal to height).
# The line, being a long thin strip, will have a much higher ratio.
# max(width,height)/min(width,height) above this value -> treated as the
# line, NOT a valid object, even if the color matches TARGET color.
SHAPE_ASPECT_MAX = 2.2

PICK_OFFSET_X  = 0              # correction applied to detected world_x before picking
PICK_OFFSET_Y  =1   #NEW: minimum real-world distance (cm) a freshly-locked boundary candidate must be from any
# already-used placement position to be accepted. Prevents a search for one zone from quietly
# locking onto / falling back to a coordinate that's actually the SAME physical spot already
# used for a previous object's placement.
ZONE_DEDUP_MIN_DISTANCE_CM = 4.0  # NEW

# NEW: distinct, deliberately separated fallback coordinates per zone_index, used only if that
# zone's AprilTag is never reliably detected after all creep attempts (see
# navigate_and_place_in_boundary()) -- so it's important these roughly match your
# ACTUAL course layout.
# Your layout is one zone in front of the robot and one off to the right (not side-by-side
# left/right), so set these to the real approximate (world_x, world_y) cm of each zone --
# measure them (e.g. from the DEBUG boundary logs) and replace the placeholders below.
ZONE_FALLBACK_POSITIONS = {0: (0.0, 12.0), 1: (10.0, 6.0)}  # <-- SET THESE to your real zone positions
# ==========================================================================

range_rgb = {
    'red':   (0, 0, 255),
    'blue':  (255, 0, 0),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}

size = (640, 480)
lab_data = None
K, R, T = None, None, None

num = 0
old_x, old_y = 0, 0

# NEW: world (x, y) cm coordinates of zones already used for a placement this run. Populated by
# navigate_and_place_in_boundary() after each placement (successful or fallback); consulted by
# locate_place_zone() so it can refuse to lock onto / fall back to a coordinate that's actually
# the SAME physical zone already used for a previous object (see ZONE_DEDUP_MIN_DISTANCE_CM).
placed_zone_positions = []  # NEW


def load_config():
    """Load LAB color thresholds and camera calibration (K, R, T)."""
    global lab_data, K, R, T

    lab_data = yaml_handle.get_yaml_data(yaml_handle.lab_file_path)
    camera_cal = yaml_handle.get_yaml_data(yaml_handle.camera_file_path)['block_params']
    K = np.array(camera_cal['K'], dtype=np.float64).reshape(3, 3)
    R = np.array(camera_cal['R'], dtype=np.float64).reshape(3, 1)
    T = np.array(camera_cal['T'], dtype=np.float64).reshape(3, 1)


def init_pose():
    """Move the arm to its scanning pose and open the gripper."""
    Board.setBusServoPulse(25, GRIPPER_OPEN, 1000)
    ik.stand(ik.initial_pos, t=800)
    AK.setPitchRangeMoving(INITIAL_COORD, -90, -90, 100, 1500)
    time.sleep(1.5)


def get_area_max_contour(contours, min_area=10):
    """Return the largest contour (and its area) from a contour list."""
    contour_area_max = 0
    area_max_contour = None

    for c in contours:
        area = math.fabs(cv2.contourArea(c))
        if area > contour_area_max and area >= min_area:
            contour_area_max = area
            area_max_contour = c

    return area_max_contour, contour_area_max


def camera_to_world(cam_mtx, r, t, img_points):
    """Project a pixel coordinate onto the ground plane -> real-world (mm)."""
    inv_k = np.asmatrix(cam_mtx).I
    r_mat = np.zeros((3, 3), dtype=np.float64)
    cv2.Rodrigues(r, r_mat)
    inv_r = np.asmatrix(r_mat).I
    transPlaneToCam = np.dot(inv_r, np.asmatrix(t))

    world_pt = []
    coords = np.zeros((3, 1), dtype=np.float64)
    for img_pt in img_points:
        coords[0][0] = img_pt[0][0]
        coords[1][0] = img_pt[0][1]
        coords[2][0] = 1.0
        worldPtCam = np.dot(inv_k, coords)
        worldPtPlane = np.dot(inv_r, worldPtCam)
        scale = transPlaneToCam[2][0] / worldPtPlane[2][0]
        scale_worldPtPlane = np.multiply(scale, worldPtPlane)
        worldPtPlaneReproject = np.asmatrix(scale_worldPtPlane) - np.asmatrix(transPlaneToCam)
        pt = np.zeros((3, 1), dtype=np.float64)
        pt[0][0] = worldPtPlaneReproject[0][0]
        pt[1][0] = worldPtPlaneReproject[1][0]
        pt[2][0] = 0
        world_pt.append(pt.T.tolist())
    return world_pt


def setBuzzer(s):
    Board.setBuzzer(1)
    time.sleep(s)
    Board.setBuzzer(0)


# REPLACED (per request): black-tape find_black_boundaries() + classify_zone()
# are replaced by find_apriltag_zones(), using the SpiderPi SDK's built-in
# AprilTag detector (see apriltag.py / ApriltagDetect.py). Each physical
# placement zone is now marked with a printed AprilTag whose tag_id IS the
# zone_index directly (tag_id 0 -> zone 0, tag_id 1 -> zone 1), so there's
# no more nearest-position guessing needed -- classify_zone() is no longer
# used anywhere and has been removed along with it.
def find_apriltag_zones(img):
    """
    Detects AprilTags in the full camera frame using the SpiderPi SDK's
    built-in detector (same apriltag_detector.detect() call as
    ApriltagDetect.py's apriltagDetect()).

    Only tag_id 0 and tag_id 1 are treated as placement zones (any other
    tag_id seen is ignored). Returns a list of
    (tag_id, pixel_x, pixel_y, world_x, world_y) for every zone tag seen
    in this frame.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = apriltag_detector.detect(gray, return_image=False)

    results = []
    for detection in detections:
        tag_id = int(detection.tag_id)
        if tag_id not in (0, 1):
            continue  # only zone tags 0 and 1 are placement zones

        cx, cy = int(detection.center[0]), int(detection.center[1])
        center = np.array([cx, cy])
        w = camera_to_world(K, R, T, center.reshape((1, 1, 2)))[0][0]
        world_x, world_y = int(-w[0]) / 10, int(-w[1]) / 10
        results.append((tag_id, cx, cy, world_x, world_y))

    print("DEBUG apriltag zones found (tag_id, pixel, world):", results)
    return results  # list of (tag_id, pixel_x, pixel_y, world_x, world_y)


# =========================================================================
# MODIFIED FUNCTION: locate_place_zone()
# See "BUGFIX NOTES" at the top of the file for the full explanation.
# Every changed/added line below is tagged with # NEW / # MODIFIED / # REMOVED.
# =========================================================================
def locate_place_zone(camera, mapx, mapy, zone_index, timeout_frames=200, allow_fallback=True,
                       search_direction=1,  # MODIFIED: search_direction is now a parameter passed
                                              # in by the caller instead of being hard-reset to 1
                                              # inside this function every call. Previously, because
                                              # navigate_and_place_in_boundary() calls this function
                                              # fresh on every creep attempt, the sweep direction was
                                              # being reset to "start right" every single attempt --
                                              # that was the direct cause of the observed rightward bias.
                       arm_sweep_max_x=ARM_SEARCH_X_MAX):  # NEW-ARM-SEARCH: how far (+/- cm from
                                              # INITIAL_COORD's x) the arm sweeps this call. The
                                              # caller (navigate_and_place_in_boundary) widens this
                                              # on later attempts instead of creeping the legs forward.
    """
    Watches the live camera feed until the boundary at zone_index
    (0 = left, 1 = right) has been seen in a stable position for
    BOUNDARY_STABLE_FRAMES frames, then returns its real-world (x, y) cm.

    ACTIVELY SEARCHES if the boundary isn't immediately visible: sweeps
    the ARM (and its wrist-mounted camera) left/right every
    ARM_SEARCH_STEP_EVERY frames instead of standing still watching a
    fixed view. MODIFIED-ARM-SEARCH: this used to sweep by turning the
    chassis (legs), which was bumping the cardboard boxes the destination
    AprilTags sit on. The legs are now never commanded during this search
    -- only the arm moves.

    If allow_fallback is False and the boundary is never found, returns
    (None, None, search_direction) instead of a guessed coordinate -- lets
    the caller decide to try creeping to a different spot and searching
    again, rather than silently placing at a guess.

    Returns a 3-tuple: (world_x, world_y, search_direction). The returned
    search_direction lets the caller continue the sweep pattern smoothly
    across multiple calls (creep attempts) instead of restarting it.
    """
    print("Looking for placement boundary (zone %d)..." % zone_index)
    stable_count = 0
    last_cx, last_cy = None, None
    frames_checked = 0
    frames_since_seen = 0
    sweep_turns_done = 0
    # search_direction = 1  # REMOVED: this used to unconditionally reset the sweep direction to
                             # "right" at the start of every call. Since navigate_and_place_in_boundary()
                             # calls locate_place_zone() again for every creep attempt, this meant every
                             # attempt always began by turning right, which is why the robot's search
                             # pattern looked like a rightward drift instead of an actual left-right sweep.
    net_arm_offset = 0  # NEW-ARM-SEARCH: running total (in cm) of the arm's x offset from
                         # INITIAL_COORD caused by OUR OWN search sweep steps during this call.
                         # camera_to_world() assumes the camera is at the calibrated INITIAL_COORD
                         # arm pose. Sweep steps move the arm off that pose, so a lock made
                         # mid-sweep must have the arm re-centered first (see below) before its
                         # pixel coordinates are trusted for world-coordinate math. (Replaces the
                         # old net_rotation, which tracked chassis-turn degrees instead.)

    while frames_checked < timeout_frames:
        img = camera.frame
        if img is None:
            cv2.waitKey(1)
            time.sleep(0.02)
            continue
        frames_checked += 1

        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        display_frame = img.copy()

        zones = find_apriltag_zones(img)  # MODIFIED: AprilTag detection instead of black-tape
        for (_, bx, by, _, _) in zones:
            cv2.circle(display_frame, (bx, by), 8, (0, 255, 255), 2)

        # MODIFIED: tag_id IS the zone_index directly, no more nearest-position
        # guessing -- a single visible tag is unambiguously its own zone.
        match = None
        for (tag_id_seen, bx, by, world_x, world_y) in zones:
            if tag_id_seen == zone_index:
                match = (bx, by)
                break

        if match is not None:
            frames_since_seen = 0
            cx, cy = match
            if last_cx is not None and abs(cx - last_cx) < BOUNDARY_MOVE_TOLERANCE and abs(cy - last_cy) < BOUNDARY_MOVE_TOLERANCE:
                stable_count += 1
            else:
                stable_count = 0
            last_cx, last_cy = cx, cy

            cv2.putText(display_frame, "Zone %d locked: %d/%d" % (zone_index, stable_count, BOUNDARY_STABLE_FRAMES),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if stable_count >= BOUNDARY_STABLE_FRAMES:
                # MODIFIED-ARM-SEARCH: re-center the ARM (instead of straightening the chassis)
                # back to INITIAL_COORD BEFORE trusting camera_to_world() for this lock. A lock
                # made mid-sweep (arm still offset from INITIAL_COORD) would otherwise feed an
                # off-axis camera view into calibration math that assumes the calibrated
                # INITIAL_COORD pose, producing a shifted/off-center world coordinate. The legs
                # are never touched here -- only the arm moves back.
                if net_arm_offset != 0:  # NEW-ARM-SEARCH
                    print("DEBUG centering arm (was offset %.1f cm) before computing placement coords" % net_arm_offset)  # NEW-ARM-SEARCH
                    AK.setPitchRangeMoving(INITIAL_COORD, -90, -90, 100, ARM_SEARCH_MOVETIME)  # NEW-ARM-SEARCH
                    net_arm_offset = 0  # NEW-ARM-SEARCH
                    time.sleep(ARM_SEARCH_SETTLE_S)  # NEW-ARM-SEARCH: let the camera settle after the corrective move

                    # NEW-ARM-SEARCH: re-check from the now arm-centered frame; if the boundary
                    # briefly isn't visible right after centering, fall back to the last locked
                    # pixel rather than failing the whole detection.
                    fresh_img = camera.frame  # NEW
                    if fresh_img is not None:  # NEW
                        fresh_img = cv2.remap(fresh_img, mapx, mapy, cv2.INTER_LINEAR)  # NEW
                        fresh_zones = find_apriltag_zones(fresh_img)  # MODIFIED: AprilTag detection
                        # MODIFIED: tag_id IS the zone_index directly -- the other
                        # zone tag may not be visible in this fresh frame either,
                        # which is fine, we only need OUR zone_index's tag here.
                        for (ftag_id, fbx, fby, fworld_x, fworld_y) in fresh_zones:  # MODIFIED
                            if ftag_id == zone_index:  # MODIFIED
                                cx, cy = fbx, fby  # NEW
                                break  # NEW

                center = np.array([cx, cy])
                w = camera_to_world(K, R, T, center.reshape((1, 1, 2)))[0][0]
                world_x, world_y = int(-w[0]) / 10, int(-w[1]) / 10

                # NEW: reject this lock if it's suspiciously close to an already-used placement
                # position -- that means we've locked onto the SAME physical zone a previous
                # object was already placed in, not the target (unused) zone. Don't return;
                # clear the stability counters and keep searching instead.
                too_close_to_used_zone = False  # NEW
                for (used_x, used_y) in placed_zone_positions:  # NEW
                    if math.hypot(world_x - used_x, world_y - used_y) < ZONE_DEDUP_MIN_DISTANCE_CM:  # NEW
                        too_close_to_used_zone = True  # NEW
                        break  # NEW

                if too_close_to_used_zone:  # NEW
                    print("DEBUG lock at (%.1f, %.1f) is within %.1f cm of an already-used zone -- "
                          "treating as a re-detection of that zone, not the target. Continuing search."
                          % (world_x, world_y, ZONE_DEDUP_MIN_DISTANCE_CM))  # NEW
                    stable_count = 0  # NEW
                    last_cx, last_cy = None, None  # NEW
                else:  # NEW
                    print("Placement zone %d found at world (%.1f, %.1f) cm" % (zone_index, world_x, world_y))
                    publish_debug_frame(display_frame)  # NEW: stream this exact annotated frame to the GUI
                    cv2.imshow("Line Follow + Fetch", display_frame)
                    cv2.waitKey(1)
                    ik.stand(ik.initial_pos, t=300)  # settle in case we were mid-sweep-turn
                    return world_x, world_y, search_direction  # MODIFIED: also return the current
                                                                 # search_direction so the caller can
                                                                 # continue the sweep pattern next time
        else:
            stable_count = 0
            last_cx, last_cy = None, None
            frames_since_seen += 1

            # MODIFIED-ARM-SEARCH: sweep the ARM left/right instead of turning/walking the
            # chassis. The legs are never commanded here, so they can't bump the cardboard
            # boxes the destination AprilTags sit on. This moves net_arm_offset back and forth
            # between +/-arm_sweep_max_x, reversing direction when it hits either edge (instead
            # of reversing after a fixed turn count, like the old leg-turn version did).
            if frames_since_seen % ARM_SEARCH_STEP_EVERY == 0:
                net_arm_offset += search_direction * ARM_SEARCH_X_STEP  # NEW-ARM-SEARCH
                if net_arm_offset >= arm_sweep_max_x:  # NEW-ARM-SEARCH: hit right edge, reverse
                    net_arm_offset = arm_sweep_max_x
                    search_direction = -1
                elif net_arm_offset <= -arm_sweep_max_x:  # NEW-ARM-SEARCH: hit left edge, reverse
                    net_arm_offset = -arm_sweep_max_x
                    search_direction = 1

                AK.setPitchRangeMoving(  # NEW-ARM-SEARCH: pan the arm/camera, legs stay planted
                    (INITIAL_COORD[0] + net_arm_offset, INITIAL_COORD[1], INITIAL_COORD[2] + ARM_SEARCH_Z_LIFT),
                    -90, -90, 100, ARM_SEARCH_MOVETIME)
                time.sleep(ARM_SEARCH_SETTLE_S)  # NEW-ARM-SEARCH: let the camera settle
                sweep_turns_done += 1
                print("DEBUG boundary search: arm sweeping to x-offset %.1f cm" % net_arm_offset)

            cv2.putText(display_frame, "Searching for zone %d... (arm sweeping)" % zone_index, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        publish_debug_frame(display_frame)  # NEW: stream this exact annotated frame to the GUI
        cv2.imshow("Line Follow + Fetch", display_frame)
        cv2.waitKey(1)

    # MODIFIED-ARM-SEARCH: re-center the arm (instead of undoing a chassis turn) before
    # returning from a failed sweep. The legs were never moved during this search, so there's
    # no chassis drift to correct -- only the arm needs to go back to INITIAL_COORD so the next
    # attempt (or the caller) starts from the known, calibrated pose.
    if net_arm_offset != 0:  # NEW-ARM-SEARCH
        print("DEBUG centering arm (was offset %.1f cm) after failed sweep" % net_arm_offset)  # NEW-ARM-SEARCH
        AK.setPitchRangeMoving(INITIAL_COORD, -90, -90, 100, ARM_SEARCH_MOVETIME)  # NEW-ARM-SEARCH
        net_arm_offset = 0  # NEW-ARM-SEARCH
        time.sleep(ARM_SEARCH_SETTLE_S)  # NEW-ARM-SEARCH

    if allow_fallback:
        fallback_x, fallback_y = ZONE_FALLBACK_POSITIONS.get(zone_index, (0.0, 9.0))  # MODIFIED: per-zone
                                                                                        # fallback instead of
                                                                                        # one shared (0.0, 9.0)
                                                                                        # that coincided with
                                                                                        # Zone 0's real location
        print("WARNING: could not reliably find placement zone %d, using fallback position (%.1f, %.1f) cm."
              % (zone_index, fallback_x, fallback_y))  # MODIFIED: log now shows which fallback was used
        return fallback_x, fallback_y, search_direction  # MODIFIED: return the per-zone fallback
    else:
        print("Zone %d not found from this spot." % zone_index)
        return None, None, search_direction  # MODIFIED: also return search_direction


# =========================================================================
# MODIFIED FUNCTION: navigate_and_place_in_boundary()
# See "BUGFIX NOTES" at the top of the file for the full explanation.
# Every changed/added line below is tagged with # NEW / # MODIFIED.
# =========================================================================
def navigate_and_place_in_boundary(camera, mapx, mapy, zone_index):
    """
    THE FIX for "boundary sometimes not seen after the return trip":

    Keeps using the SAME reliable, calibrated, steep INITIAL_COORD camera
    view for detection the whole time (this view was proven stable and
    accurate whenever the boundary was actually inside its field of view --
    see the consistent (129,6)/(130,7)/(129,6)... pixel readings in earlier
    tests). A wider/shallower camera angle was tried instead, but it
    introduced heavy perspective distortion -- the same physical box's
    contour aspect ratio would swing between ~1.0 and ~8+ from one frame
    to the next as the robot turned even slightly, so the filter kept
    accepting/rejecting it inconsistently and the robot oscillated,
    "looking here, then there," never settling.

    Instead of trying to see farther by distorting the camera angle, this
    searches by sweeping the ARM (camera) left/right in place and looking
    (locate_place_zone's existing stable-lock logic, reused as-is); if the
    boundary isn't found after a full sweep, the sweep range is WIDENED
    and tried again from the SAME spot. Up to BOUNDARY_MAX_CREEP_ATTEMPTS
    attempts before giving up and using a fallback position.

    MODIFIED-ARM-SEARCH (leg-vs-box collision fix): this used to creep the
    CHASSIS forward with ik.go_forward() between attempts. Since the
    destination AprilTags sit on lightweight cardboard boxes that the legs
    were bumping into position, the chassis is no longer moved at all
    during this search -- only the arm/camera sweeps, and now sweeps a
    wider range on each retry instead of physically creeping closer.

    The returned placement coordinate is the DETECTED CENTER of the
    boundary square itself, so as long as it was actually detected (not
    the fallback), the object lands inside the boundary, not just "near" it.

    BUGFIX: search_direction is now tracked across creep attempts (see NEW
    lines below) instead of being reset to "start right" on every attempt,
    which was the direct cause of the reported rightward search bias.

    BUGFIX ROUND 2: every placement made here (successful detection OR
    exhausted-fallback) is now recorded in `placed_zone_positions` and the
    exhausted-fallback path now uses a per-zone-distinct coordinate instead
    of a single shared one -- see "BUGFIX NOTES ROUND 2" at the top of the
    file for why this was needed.
    """
    print("Searching for placement boundary (zone %d)..." % zone_index)

    search_direction = 1  # NEW: owned here now so it PERSISTS across creep attempts within this
                           # zone's search, instead of being reset to 1 inside locate_place_zone()
                           # on every single attempt. Still freshly initialized to 1 each time
                           # navigate_and_place_in_boundary() itself is called (i.e. once per zone),
                           # so Zone 0 and Zone 1 each still start their search unbiased.

    arm_sweep_max_x = ARM_SEARCH_X_MAX  # NEW-ARM-SEARCH: widened on each failed attempt below,
                                         # in place of physically creeping the legs forward

    for attempt in range(1, BOUNDARY_MAX_CREEP_ATTEMPTS + 1):
        print("Sweep attempt %d/%d (arm sweep range +/-%.1f cm)..." %
              (attempt, BOUNDARY_MAX_CREEP_ATTEMPTS, arm_sweep_max_x))
        place_world_x, place_world_y, search_direction = locate_place_zone(  # MODIFIED: capture the
            camera, mapx, mapy, zone_index,                                  # returned search_direction
            timeout_frames=BOUNDARY_SWEEP_FRAMES, allow_fallback=False,
            search_direction=search_direction,  # MODIFIED: pass the current direction in so the
                                                 # sweep continues smoothly instead of restarting
            arm_sweep_max_x=arm_sweep_max_x)  # NEW-ARM-SEARCH

        if place_world_x is not None:
            print("Boundary zone %d found on attempt %d -> placing INSIDE it at (%.1f, %.1f) cm."
                  % (zone_index, attempt, place_world_x, place_world_y))
            place_object(place_world_x, place_world_y)
            placed_zone_positions.append((place_world_x, place_world_y))  # NEW: remember this
                                                                            # zone's world position so
                                                                            # future zone searches (and
                                                                            # any fallback) can avoid
                                                                            # re-landing on it
            return

        if attempt < BOUNDARY_MAX_CREEP_ATTEMPTS:
            # MODIFIED-ARM-SEARCH: legs stay planted -- no ik.go_forward() creep anymore.
            # Widen the arm's sweep range instead, so the next attempt looks farther
            # side-to-side from the exact same spot.
            arm_sweep_max_x += ARM_SEARCH_WIDEN_STEP
            print("Not found yet -- widening arm sweep range to +/-%.1f cm and trying again "
                  "(legs stay still)..." % arm_sweep_max_x)

    fallback_x, fallback_y = ZONE_FALLBACK_POSITIONS.get(zone_index, (0.0, 9.0))  # MODIFIED: use the
                                                                                    # per-zone fallback
                                                                                    # instead of a single
                                                                                    # shared (0.0, 9.0)
                                                                                    # that overlapped
                                                                                    # Zone 0's real spot
    print("WARNING: boundary zone %d never found after %d attempts -- using fallback position (%.1f, %.1f) cm."
          % (zone_index, BOUNDARY_MAX_CREEP_ATTEMPTS, fallback_x, fallback_y))  # MODIFIED: log shows the fallback used
    place_object(fallback_x, fallback_y)  # MODIFIED: place at the per-zone fallback
    placed_zone_positions.append((fallback_x, fallback_y))  # NEW: record this fallback placement too,
                                                              # so a later zone's search/fallback still
                                                              # avoids colliding with it


# =========================================================
# OBJECT DETECTION (exact logic from Pick_and_Place.py's
# ColorDetect, just returning state instead of using globals
# + a thread)
# =========================================================
def ColorDetect(img, color):
    """
    Returns: (stable, world_x, world_y, seen)
        stable  -> True once the object's position has been steady for
                   STABLE_FRAMES frames (safe to pick now)
        world_x, world_y -> real-world coords (cm) if stable, else None
        seen    -> True if a valid BOX-SHAPED candidate is visible at all
                   this frame (caller should HOLD STILL, not keep walking,
                   while this is True but stable is still False)
    """
    global num, old_x, old_y

    img_h, img_w = img.shape[:2]
    frame_resize = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    frame_gb = cv2.GaussianBlur(frame_resize, (3, 3), 3)
    frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)
    frame_mask = cv2.inRange(
        frame_lab,
        (lab_data[color]['min'][0], lab_data[color]['min'][1], lab_data[color]['min'][2]),
        (lab_data[color]['max'][0], lab_data[color]['max'][1], lab_data[color]['max'][2]),
    )
    eroded = cv2.erode(frame_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    areaMaxContour, area_max = get_area_max_contour(contours, min_area=100)

    if area_max > AREA_THRESHOLD:
        # --- SHAPE CHECK: reject elongated (line-like) contours ---
        # This is what lets a box be the SAME color as the line without
        # the line itself being mistaken for the box.
        rect = cv2.minAreaRect(areaMaxContour)
        rw, rh = rect[1]
        if rw < 1 or rh < 1:
            num = 0
            return False, None, None, False
        aspect_ratio = max(rw, rh) / min(rw, rh)
        print("DEBUG ColorDetect(%s): area=%d aspect_ratio=%.2f (max allowed=%.2f)" %
              (color, int(area_max), aspect_ratio, SHAPE_ASPECT_MAX))
        if aspect_ratio > SHAPE_ASPECT_MAX:
            # too long/thin -- this is the line, not a box. Ignore it.
            num = 0
            return False, None, None, False

        (centerX, centerY), radius = cv2.minEnclosingCircle(areaMaxContour)
        centerX = int(Misc.map(centerX, 0, size[0], 0, img_w))
        centerY = int(Misc.map(centerY, 0, size[1], 0, img_h))
        radius = int(Misc.map(radius, 0, size[0], 0, img_w))
        print("DEBUG ColorDetect(%s): pixel center=(%d, %d) num=%d/%d" %
              (color, centerX, centerY, num, STABLE_FRAMES))

        cv2.circle(img, (centerX, centerY), radius, range_rgb[color], 2)
        cv2.putText(img, "Color: " + color, (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, range_rgb[color], 2)

        # Check whether the position has stopped moving
        if abs(centerX - old_x) < 8 and abs(centerY - old_y) < 8:
            num += 1
        else:
            num = 0
            old_x, old_y = centerX, centerY

        if num > STABLE_FRAMES:
            center = np.array([centerX, centerY])
            w = camera_to_world(K, R, T, center.reshape((1, 1, 2)))[0][0]
            world_x, world_y = int(-w[0]) / 10, int(-w[1]) / 10
            print("Object at world coords:", world_x, world_y)
            num = 0
            return True, world_x, world_y, True

        return False, None, None, True  # seen but not stable yet -> hold still

    else:
        num = 0
        return False, None, None, False  # not seen -> keep following the line


# NEW (ROOT-CAUSE FIX): ported from pi_pick_and_place.py's camera_stream_server().
# Only the CONNECTION/STREAMING SETUP was reused, as instructed -- none of that
# file's AprilTag/pick/place logic is used here. This is the missing half of
# the pipeline: without it, pc_yolo_sender.py's PiCameraReceiver connects to
# CAMERA_STREAM_PORT and just waits forever, since nothing on the Pi was ever
# listening/serving on that port -- meaning YOLO on the PC had zero frames to
# run detection on, regardless of the detection-message protocol being
# correct. Uses the SAME `camera` object main() already opens for the robot's
# own line-following/detection use -- does not open a second camera or change
# anything about how the robot itself sees.
def camera_stream_server(camera_obj, mapx, mapy):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', CAMERA_STREAM_PORT))
    server.listen(5)
    print("Camera stream server listening on %d" % CAMERA_STREAM_PORT)

    while True:
        conn, addr = server.accept()
        print("PC connected for camera stream: %s" % str(addr))
        try:
            while True:
                frame = camera_obj.frame
                if frame is None:
                    time.sleep(0.01)
                    continue
                # === CHANGE 3: undistort before sending. camera_to_world()
                # (used when a STABLE detection comes back) was calibrated
                # against the UNDISTORTED view -- the same cv2.remap() the
                # Pi's own line-following/boundary code already applies.
                # Sending the raw/distorted frame to YOLO means the returned
                # pixel (cx, cy) doesn't line up with that calibration,
                # causing a systematic placement-accuracy error (worse near
                # the frame edges). ===
                frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret:
                    continue
                data = jpg.tobytes()
                conn.sendall(struct.pack(">L", len(data)) + data)
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            print("PC disconnected from camera stream.")
        finally:
            conn.close()


_debug_frame_holder = {"frame": None}  # NEW: latest annotated display_frame, shared with debug_stream_server
_debug_frame_lock = threading.Lock()   # NEW


def publish_debug_frame(frame):
    # NEW: called right where display_frame is normally handed to cv2.imshow(),
    # so the GUI can see the exact same line/AprilTag-box overlay the Pi's own
    # local preview window shows. Does not affect cv2.imshow() itself or
    # anything else about the existing preview/debug behaviour.
    with _debug_frame_lock:
        _debug_frame_holder["frame"] = frame


def debug_stream_server():
    # NEW: same accept/encode/send pattern as camera_stream_server() above,
    # just serving _debug_frame_holder's latest annotated frame instead of
    # the raw camera frame, on its own port so it doesn't collide with the
    # existing raw-frame stream.
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', DEBUG_STREAM_PORT))
    server.listen(5)
    print("Debug stream server listening on %d" % DEBUG_STREAM_PORT)

    while True:
        conn, addr = server.accept()
        print("PC connected for debug stream: %s" % str(addr))
        try:
            while True:
                with _debug_frame_lock:
                    frame = _debug_frame_holder["frame"]
                if frame is None:
                    time.sleep(0.01)
                    continue
                ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if not ret:
                    continue
                data = jpg.tobytes()
                conn.sendall(struct.pack(">L", len(data)) + data)
                time.sleep(0.03)
        except (BrokenPipeError, ConnectionResetError):
            print("PC disconnected from debug stream.")
        finally:
            conn.close()


class RemoteStopRequested(Exception):
    # NEW: raised from inside the mission loops when the GUI's "Disconnect"
    # button sends a STOP command, so main() can unwind cleanly to the same
    # `finally: ik.stand(...)` cleanup already used for 'q' / Ctrl+C.
    pass


class YoloServer:
    """
    MODIFIED: replaces the earlier YoloClient. pc_yolo_sender.py's
    PiControlSender CONNECTS OUT to the Pi on YOLO_DETECTION_PORT, so the
    Pi needs to be the SERVER here, not a client dialing out to the PC.

    Accepts that connection and parses its "TYPE,cls_name,cx,cy\n" messages
    (TYPE is SEEN / STABLE / LOST) into the latest known state, retrievable
    via get_latest().
    """
    def __init__(self, port):
        self.port = port
        self.status = None       # 'SEEN' / 'STABLE' / 'LOST' / None
        self.cls_name = None
        self.cx, self.cy = None, None
        self.remote_start_requested = False  # NEW: set True once by a "START" message
        self.remote_stop_flag = False        # NEW: stays True from "STOP" until the next "START"
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._serve_loop, daemon=True)
        self.thread.start()

    def _serve_loop(self):
        while self.running:
            srv = None
            try:
                srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind(('0.0.0.0', self.port))
                srv.listen(1)
                print("Waiting for PC YOLO control connection on port %d..." % self.port)
                conn, addr = srv.accept()
                print("PC connected for YOLO control messages: %s" % str(addr))
                buf = b""
                with conn:
                    while self.running:
                        chunk = conn.recv(4096)
                        if not chunk:
                            raise ConnectionError("PC control connection closed")
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            self._parse_line(line)
            except Exception as e:
                print("YOLO control server error: %s -- reopening in 2s" % e)
                time.sleep(2)
            finally:
                if srv is not None:
                    srv.close()

    def _parse_line(self, line):
        try:
            parts = line.decode('utf-8').strip().split(',')
            if len(parts) != 4:
                print("YOLO control message ignored (expected 4 comma-fields, got %d): %r"
                      % (len(parts), line))
                return
            status, cls_name, cx_s, cy_s = parts

            # NEW: START/STOP are remote mission-control signals from the GUI's
            # Pi-Camera Connect/Disconnect buttons, not detection events -- handled
            # here separately so they don't overwrite the last known SEEN/STABLE/LOST
            # detection state that YoloObjectDetect() relies on.
            if status == "START":
                with self.lock:
                    self.remote_start_requested = True
                    self.remote_stop_flag = False
                print("DEBUG Pi received: remote START (GUI Connect clicked)")
                return
            if status == "STOP":
                with self.lock:
                    self.remote_stop_flag = True
                print("DEBUG Pi received: remote STOP (GUI Disconnect clicked)")
                return

            with self.lock:
                self.status = status
                self.cls_name = cls_name if cls_name else None
                self.cx = int(cx_s) if cx_s not in ('', 'None') else None
                self.cy = int(cy_s) if cy_s not in ('', 'None') else None
            # === CHANGE 2: confirm receipt on the Pi side. Previously only
            # parse ERRORS printed anything -- a successfully-received message
            # was completely silent, so there was no way to tell "the Pi got
            # nothing" apart from "the Pi got messages but something after
            # this point is wrong." ===
            print("DEBUG Pi received: status=%s cls_name=%s cx=%s cy=%s"
                  % (self.status, self.cls_name, self.cx, self.cy))
        except Exception as e:
            print("YOLO control message parse error: %s (line=%r)" % (e, line))

    def get_latest(self):
        with self.lock:
            return self.status, self.cls_name, self.cx, self.cy

    def get_priority_override(self):
        # NEW: consumes a "PRIORITY,cls_name,0,0" message sent by the GUI's
        # Prioritize buttons (see pi_link.py's request_priority()). Returns
        # the requested class name once, then clears it, so it doesn't keep
        # re-triggering every time this is checked. Reuses the exact same
        # message format/parsing as SEEN/STABLE/LOST -- no protocol change.
        with self.lock:
            if self.status == "PRIORITY":
                requested = self.cls_name
                self.status = None
                self.cls_name = None
                return requested
            return None

    def consume_start_request(self):
        # NEW: returns True exactly once per GUI "Connect" click, then clears
        # itself -- used by main()'s outer wait-loop to gate mission start.
        with self.lock:
            if self.remote_start_requested:
                self.remote_start_requested = False
                return True
            return False

    def is_remote_stop(self):
        # NEW: True from the moment GUI "Disconnect" is clicked until the
        # next "Connect". Checked inside the mission loops (same checkpoints
        # as the existing 'q' keyboard-quit check) so the robot halts
        # promptly once it reaches a safe checkpoint, not mid-pick/mid-place.
        with self.lock:
            return self.remote_stop_flag

    def reset(self):
        # NEW: clears the last-known detection state. Used before the
        # camera-tilt confirmation check so a stale SEEN/STABLE message
        # from BEFORE the tilt can't be mistaken for a fresh confirming
        # detection AFTER the tilt.
        with self.lock:
            self.status = None
            self.cls_name = None
            self.cx, self.cy = None, None

    def stop(self):
        self.running = False


yolo_server = None  # NEW: set in main()


def YoloObjectDetect(target_class):
    """
    MODIFIED: rewritten against pc_yolo_sender.py's ACTUAL protocol.

    Drop-in replacement for ColorDetect(img, color) at the call site, but
    the stability model is different by necessity: pc_yolo_sender.py
    already does its own class-based stability check (STABILITY_FRAMES_REQUIRED
    consecutive frames of the same class) and sends STABLE only ONCE per
    approach -- it does NOT stream continuous per-frame updates the way
    ColorDetect() polled every frame locally. So the old num/old_x/old_y
    PIXEL-stability check doesn't apply here: there's only one STABLE
    message to react to, not a series of frames to compare positions across.

    SEEN vs STABLE still map onto ColorDetect's contract:
      - SEEN   -> seen=True, stable=False  (hold still, don't walk past it)
      - STABLE -> seen=True, stable=True   (safe to pick -- PC already
                  confirmed class-stability; this converts its pixel
                  center to world coords via the SAME camera_to_world()
                  used everywhere else)
      - LOST / None / class mismatch -> seen=False, stable=False (keep
                  following the line)

    Returns: (stable, world_x, world_y, seen)
    """
    status, cls_name, cx, cy = yolo_server.get_latest() if yolo_server else (None, None, None, None)

    if status is None:
        return False, None, None, False

    if cls_name != target_class:
        # === CHANGE 2b: this used to fail completely silently. If cls_name
        # from the PC never matches target_class (e.g. a spelling/casing
        # mismatch between TARGET_SEQUENCE here and model.names on the PC),
        # this branch would return False forever with zero evidence why.
        # Edge-triggered (only prints when the mismatch actually changes) to
        # avoid flooding the console every single frame. ===
        mismatch_key = (status, cls_name, target_class)
        if getattr(YoloObjectDetect, "_last_mismatch", None) != mismatch_key:
            print("DEBUG YoloObjectDetect: got status=%s for class '%s' but currently "
                  "targeting '%s' -- ignoring (not a match)." % (status, cls_name, target_class))
            YoloObjectDetect._last_mismatch = mismatch_key
        return False, None, None, False

    if status == "LOST":
        return False, None, None, False

    if status == "SEEN":
        return False, None, None, True

    if status == "STABLE":
        print("DEBUG YoloObjectDetect(%s): STABLE at pixel=(%s, %s)" % (target_class, cx, cy))
        center = np.array([cx, cy])
        w = camera_to_world(K, R, T, center.reshape((1, 1, 2)))[0][0]
        world_x, world_y = int(-w[0]) / 10, int(-w[1]) / 10
        print("Object at world coords:", world_x, world_y)
        return True, world_x, world_y, True

    return False, None, None, False


# =========================================================
# LINE FOLLOWING
# =========================================================
def find_line_center(img, draw_on=None):
    frame_gb = cv2.GaussianBlur(img, (3, 3), 3)
    for (y1, y2, x1, x2) in LINE_ROI:
        strip = frame_gb[y1:y2, x1:x2]
        frame_lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB)
        mask = cv2.inRange(frame_lab,
                            tuple(lab_data[LINE_COLOR]['min']),
                            tuple(lab_data[LINE_COLOR]['max']))
        eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)[-2]
        largest, area = get_area_max_contour(contours, min_area=40)
        if largest is not None and area > 40:
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box[:, 1] = box[:, 1] + y1
            pt1_x, pt1_y = box[0][0], box[0][1]
            pt3_x, pt3_y = box[2][0], box[2][1]
            center_x = (pt1_x + pt3_x) / 2.0
            if draw_on is not None:
                cv2.drawContours(draw_on, [box], -1, (0, 255, 255), 2)
            return center_x
    return None


LINE_MISS_HYSTERESIS = 3  # consecutive no-detection frames tolerated before reacting as truly lost


def follow_line_step(img, last_center, miss_count=0, draw_on=None):
    """Take exactly one line-following action.
    Returns (status_text, new_last_center, stepped_forward, new_miss_count)."""
    center_x = find_line_center(img, draw_on=draw_on)
    stepped_forward = False

    if center_x is not None:
        miss_count = 0
        offset = center_x - IMG_CENTER_X
        if abs(offset) < 60:
            status = "FORWARD"
            ik.go_forward(ik.initial_pos, 2, 30, 50, 1)
            stepped_forward = True
        elif offset >= 60:
            status = "TURN RIGHT"
            ik.turn_right(ik.initial_pos, 2, 10, 50, 1)
        else:
            status = "TURN LEFT"
            ik.turn_left(ik.initial_pos, 2, 10, 50, 1)
        last_center = center_x
    else:
        miss_count += 1
        if miss_count < LINE_MISS_HYSTERESIS:
            # Momentary flicker (thin/partial blob near the line's end) --
            # don't react yet, just hold this frame instead of jerking
            # forward/back on noise.
            status = "LINE FLICKER - holding (%d/%d)" % (miss_count, LINE_MISS_HYSTERESIS)
        else:
            if last_center >= IMG_CENTER_X:
                status = "LINE LOST - turning left"
                ik.turn_left(ik.initial_pos, 2, 10, 50, 1)
            else:
                status = "LINE LOST - turning right"
                ik.turn_right(ik.initial_pos, 2, 10, 50, 1)

    return status, last_center, stepped_forward, miss_count


# =========================================================
# PICK / PLACE / RETURN
# =========================================================
def confirm_detection_at_tilt(target_class):
    """
    CORRECTED: second-look confirmation using the wrist-mounted camera
    (servo 24) at a different pitch, BEFORE committing to a pick. Tilts the
    arm through AK.setPitchRangeMoving() -- same (x, y, z) as INITIAL_COORD,
    different pitch -- instead of poking a servo directly, so the IK model
    always stays in sync with where the arm physically is.

    Does NOT recompute or use any world coordinates from the tilted view --
    camera_to_world()'s calibration (K, R, T) is only valid at the normal
    straight-down pose, so the tilted frame is used purely to re-classify,
    never to re-locate.

    Returns True if a matching SEEN or STABLE for target_class arrives
    within CONFIRM_TIMEOUT_S at the tilted angle, False otherwise (treat
    the original detection as a likely false positive).
    """
    print("Confirming detection: tilting wrist camera for a second look...")
    result = AK.setPitchRangeMoving(CONFIRM_TILT_COORD, CONFIRM_TILT_ALPHA,
                                     CONFIRM_TILT_ALPHA1, CONFIRM_TILT_ALPHA2,
                                     CONFIRM_TILT_MOVETIME)
    if result is None:
        print("WARNING: no IK solution for the confirm-tilt pose -- skipping "
              "second look, treating original detection as unconfirmed.")
        return False
    time.sleep(CONFIRM_TILT_SETTLE_S)

    if yolo_server is not None:
        yolo_server.reset()  # discard any pre-tilt detection state

    confirmed = False
    start = time.time()
    while time.time() - start < CONFIRM_TIMEOUT_S:
        status, cls_name, cx, cy = yolo_server.get_latest() if yolo_server else (None, None, None, None)
        if status in ("SEEN", "STABLE") and cls_name == target_class:
            print("Confirmed at tilted angle: status=%s cls_name=%s" % (status, cls_name))
            confirmed = True
            break
        time.sleep(0.05)

    if not confirmed:
        print("NOT confirmed at tilted angle -- treating original detection as a false positive.")

    # Always return the arm (and camera) to its normal scanning pose/pitch
    # before proceeding (whether we're about to pick or about to abort).
    AK.setPitchRangeMoving(INITIAL_COORD, -90, -90, 100, CONFIRM_TILT_MOVETIME)
    time.sleep(CONFIRM_TILT_SETTLE_S)

    if yolo_server is not None:
        yolo_server.reset()  # NEW: also discard whatever came in during the
                              # tilted look so the main loop doesn't act on
                              # a detection captured from the wrong angle.

    return confirmed


def pick_object(world_x, world_y):
    corrected_x = world_x + PICK_OFFSET_X
    corrected_y = world_y + PICK_OFFSET_Y
    print("Picking at corrected world coords: (%.1f, %.1f)" % (corrected_x, corrected_y))

    Board.setBusServoPulse(25, GRIPPER_OPEN, 500)
    x = INITIAL_COORD[0] + corrected_x
    y = INITIAL_COORD[1] + corrected_y

    # NEW (diagnostic only -- does not change behavior/flow): setPitchRangeMoving()
    # returns None when the requested (x, y, z) is outside the arm's physical
    # reach, in which case the arm silently does NOT move at all. Previously
    # this return value was never checked, so an out-of-reach pick target
    # would fail with zero indication anything went wrong. This only logs a
    # warning -- it does not abort, retry, or alter any coordinate math.
    result = AK.setPitchRangeMoving((x, y, -5), -90, -90, 100, 2000)
    if result is None:
        print("WARNING: pick target (x=%.1f, y=%.1f, z=-5) is OUT OF THE ARM'S REACH -- "
              "the arm did NOT move for this command. This may explain an unreliable pickup." % (x, y))
    time.sleep(2)

    Board.setBusServoPulse(25, GRIPPER_CLOSE, 500)
    time.sleep(0.5)

    AK.setPitchRangeMoving((x, y, 8), -90, -90, 100, 1000)
    time.sleep(1)

    AK.setPitchRangeMoving(INITIAL_COORD, -90, -90, 100, 1500)
    time.sleep(1.5)
    print("Object picked.")


def place_object(world_x, world_y):
    print("Placing object at detected boundary world (%.1f, %.1f) cm" % (world_x, world_y))
    place_x = INITIAL_COORD[0] + world_x
    place_y = INITIAL_COORD[1] + world_y

    AK.setPitchRangeMoving((place_x, place_y, 8), -90, -90, 100, 1500)
    time.sleep(1.5)
    AK.setPitchRangeMoving((place_x, place_y, PLACE_Z), -90, -90, 100, 800)
    time.sleep(0.8)

    Board.setBusServoPulse(25, GRIPPER_OPEN, 500)
    time.sleep(0.5)

    AK.setPitchRangeMoving((place_x, place_y, 8), -90, -90, 100, 1000)
    time.sleep(1)
    AK.setPitchRangeMoving(INITIAL_COORD, -90, -90, 100, 1500)
    time.sleep(1.5)

    setBuzzer(0.1)
    print("Object placed.")


def turn_180():
    print("Turning 180 degrees...")
    for _ in range(12):  # ~15 deg each -> 180 total
        ik.turn_left(ik.initial_pos, 2, 15, 60, 1)
    ik.stand(ik.initial_pos, t=500)


def walk_back(steps):
    print("Walking back %d steps to starting point..." % steps)
    for _ in range(steps):
        ik.go_forward(ik.initial_pos, 2, 30, 50, 1)
    ik.stand(ik.initial_pos, t=500)


def return_via_line(camera, target_steps):
    """
    Retrace the path back using LIVE line-following, walking approximately
    target_steps forward-steps worth of distance.

    BUG FIX: previously, if the line was briefly lost for RETURN_LOST_LIMIT
    consecutive frames RIGHT AFTER turn_180() (e.g. because the camera
    hadn't settled yet from the turn), the function would declare "arrived"
    and stop almost instantly -- having taken 0-1 actual steps back. That
    made the boundary search start right next to where the object was
    picked, not back at the true start of the line, which looked like
    "it's searching for the zone immediately after picking."

    Fix: early-stop on line-lost is now only allowed once at least
    MIN_RETURN_STEPS_FRACTION of target_steps has actually been walked.
    Before that point, a "lost" streak just resets and keeps trying
    (assumed to be a momentary glitch, not the true end of the line). A
    short settle pause is also added right at the start so the first few
    frames aren't read while the robot is still mid-turn.

    Alternating micro-search turns (right, left, right, left...) are still
    used during recovery so the robot doesn't drift its heading while
    searching for a lost line.

    MODIFIED: target_steps is now a SAFETY CEILING only, not the primary
    stop condition. The outbound step-count (forward_steps counted on the
    way to the object) drifts run-to-run (motor/battery/surface variance),
    so stopping exactly at that count was landing the robot 5-6 steps
    short/long of the real start position, which then made the arm-sweep
    AprilTag search bump the boxes. The REAL stop signal is now purely
    visual: keep line-following until the line is genuinely gone (lost for
    RETURN_LOST_LIMIT consecutive frames, after the min-steps settle
    safeguard above). target_steps is only used to size a generous ceiling
    (2x + margin) so the loop can't run forever if something goes wrong.
    """
    print("Returning via line-following (walking until the line visually ends, ~%d steps expected)..." % target_steps)
    RETURN_LOST_LIMIT = 8  # consecutive lost frames before considering "maybe arrived"
    RETURN_MISS_HYSTERESIS = 3  # NEW: consecutive lost frames tolerated before physically micro-searching
                                 # (mirrors follow_line_step's LINE_MISS_HYSTERESIS) -- avoids jerking on
                                 # a 1-2 frame flicker near the line's end.
    min_steps_before_early_stop = max(1, int(target_steps * 0.5))
    max_return_steps = target_steps * 2 + 20  # NEW: safety ceiling only -- normal stop is the
                                               # line-lost detection below, not this count
    max_frames = max_return_steps * 15  # NEW: hard ceiling on FRAMES (not just forward steps taken).
                                         # steps_taken only increments on a successful forward step, so if
                                         # the line is never re-detected on a return trip, steps_taken can
                                         # stay 0 forever and the old `while steps_taken < max_return_steps`
                                         # condition never ends the loop -- this is the infinite jerking
                                         # loop reported on some second-object return trips. This frame
                                         # ceiling guarantees the loop always terminates regardless.
    print("Return safeguard: at least %d step(s) must be walked before line-lost can end the trip early "
          "(hard safety ceiling: %d steps / %d frames)." % (min_steps_before_early_stop, max_return_steps, max_frames))

    time.sleep(0.4)  # let the camera settle after turn_180() before reading frames

    last_center = IMG_CENTER_X
    steps_taken = 0
    lost_count = 0
    frames_elapsed = 0  # NEW: counts every loop iteration, see max_frames above

    while steps_taken < max_return_steps and frames_elapsed < max_frames:  # MODIFIED: added frames_elapsed
        frames_elapsed += 1  # NEW
        img = camera.frame
        if img is None:
            cv2.waitKey(1)
            time.sleep(0.02)
            continue

        display_frame = img.copy()
        center_x = find_line_center(img, draw_on=display_frame)

        if center_x is not None:
            lost_count = 0
            offset = center_x - IMG_CENTER_X
            if abs(offset) < 60:
                status = "RETURN FORWARD"
                ik.go_forward(ik.initial_pos, 2, 30, 50, 1)
                steps_taken += 1
            elif offset >= 60:
                status = "RETURN TURN RIGHT"
                ik.turn_right(ik.initial_pos, 2, 10, 50, 1)
            else:
                status = "RETURN TURN LEFT"
                ik.turn_left(ik.initial_pos, 2, 10, 50, 1)
            last_center = center_x

        else:
            lost_count += 1
            # MODIFIED: don't physically react on a 1-2 frame flicker (thin/partial line blob
            # near the line's end) -- only start the alternating micro-search turns once the
            # miss has persisted past RETURN_MISS_HYSTERESIS frames. This is the fix for the
            # repeated jerking: previously every single lost frame fired a turn command.
            if lost_count <= RETURN_MISS_HYSTERESIS:
                status = "RETURN LINE FLICKER - holding (%d/%d)" % (lost_count, RETURN_MISS_HYSTERESIS)
            else:
                # MODIFIED (return-journey fix): search CLOCKWISE (right) first, not
                # anticlockwise (left). After the pick, the robot has already turned
                # 180 degrees to head back -- searching left first was rotating it back
                # toward the original pickup side instead of toward the return path,
                # wasting time and sometimes drifting back to the pickup area.
                # `search_attempt` counts physical search turns starting at 1 (not
                # `lost_count`, which already includes the RETURN_MISS_HYSTERESIS hold
                # frames) so the very first turn is guaranteed to be right/clockwise,
                # regardless of the hysteresis value above.
                search_attempt = lost_count - RETURN_MISS_HYSTERESIS  # NEW: 1 on the first physical search turn
                if search_attempt % 2 == 1:
                    ik.turn_right(ik.initial_pos, 2, 8, 50, 1)  # MODIFIED: clockwise first
                    status = "RETURN LINE LOST - micro-search right/clockwise (%d/%d)" % (lost_count, RETURN_LOST_LIMIT)
                else:
                    ik.turn_left(ik.initial_pos, 2, 8, 50, 1)
                    status = "RETURN LINE LOST - micro-search left/anticlockwise (%d/%d)" % (lost_count, RETURN_LOST_LIMIT)

            if lost_count >= RETURN_LOST_LIMIT:
                if steps_taken >= min_steps_before_early_stop:
                    print("Line has been missing for %d frames after walking %d/%d steps - "
                          "assuming the start of the line has been reached. Stopping here."
                          % (lost_count, steps_taken, target_steps))
                    break
                else:
                    print("Line lost for %d frames but only %d/%d steps walked so far (need >= %d) "
                          "- treating as a momentary glitch, continuing to search."
                          % (lost_count, steps_taken, target_steps, min_steps_before_early_stop))
                    lost_count = 0

        cv2.putText(display_frame, "RETURNING " + status + " steps=%d/%d" % (steps_taken, target_steps),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        publish_debug_frame(display_frame)  # NEW: stream this exact annotated frame to the GUI
        cv2.imshow("Line Follow + Fetch", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    ik.stand(ik.initial_pos, t=500)
    if steps_taken >= max_return_steps:
        print("WARNING: hit the %d-step safety ceiling without the line ever visually ending -- "
              "stopped here anyway. Check the line/camera if this keeps happening." % max_return_steps)
    elif frames_elapsed >= max_frames:
        print("WARNING: hit the %d-frame safety ceiling (line was never re-detected long enough) -- "
              "stopped here anyway. Check the line/camera if this keeps happening." % max_frames)
    print("Return trip complete (%d steps walked, expected ~%d)." % (steps_taken, target_steps))


# =========================================================
# MAIN
# =========================================================
def main():
    from CameraCalibration.CalibrationConfig import calibration_param_path

    load_config()

    # --- Lens undistortion setup (THIS WAS MISSING BEFORE) ---
    # Every one of the original scripts (Pick_and_Place.py, VisualPatrol.py,
    # block_fetch.py, ...) undistorts every frame BEFORE running color
    # detection or camera_to_world(). Skipping this step is why the merged
    # script's pick coordinates were unreliable even though standalone
    # Pick_and_Place.py worked.
    param_data = np.load(calibration_param_path + '.npz')
    mtx = param_data['mtx_array']
    dist = param_data['dist_array']
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (640, 480), 0, (640, 480))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (640, 480), 5)
    # -----------------------------------------------------------

    camera = Camera.Camera()
    camera.camera_open()
    time.sleep(1)

    # NEW (ROOT-CAUSE FIX): start streaming this camera's frames to the PC.
    # Without this thread, pc_yolo_sender.py's PiCameraReceiver connects to
    # CAMERA_STREAM_PORT and waits forever with nothing sending it frames --
    # YOLO on the PC then has nothing to detect, and never sends anything
    # back, regardless of everything else being correct.
    threading.Thread(target=camera_stream_server, args=(camera, mapx, mapy), daemon=True).start()

    # NEW: start streaming the annotated debug frame (line/AprilTag boxes) too,
    # on its own port, so the GUI's third panel can show it.
    threading.Thread(target=debug_stream_server, daemon=True).start()

    global yolo_server  # NEW
    yolo_server = YoloServer(YOLO_DETECTION_PORT)  # NEW: waits for pc_yolo_sender.py to connect

    init_pose()

    print("Ready. Waiting for GUI 'Connect' (Pi Camera panel) to start the mission. "
          "Press 'q' in the preview window to stop a running mission.")

    # NEW: outer loop -- gates each mission run on a remote START (GUI's
    # Pi-Camera "Connect" button) and, when that mission ends (either by
    # GUI "Disconnect", 'q', or completing every object), loops back here to
    # wait for the next Connect instead of exiting the whole script. The
    # camera / debug-stream / YoloServer threads started above keep running
    # the entire time, so the camera feed panel in the GUI never drops.
    try:
        while True:
            while not yolo_server.consume_start_request():
                time.sleep(0.1)
            print("\nConnect received from GUI -- starting mission.")

            stop_requested = False
            last_successful_steps = None  # set after the first object is picked

            # MODIFIED: TARGET_SEQUENCE is copied into a mutable list so a
            # priority override from the GUI (see get_priority_override() above)
            # can reorder what's picked NEXT, AND can now interrupt a search
            # already in progress too (see the check inside "while not picked"
            # below) -- safe because it only ever fires before a pick has
            # physically started, never mid pick/turn/return/place.
            remaining_targets = list(TARGET_SEQUENCE)
            idx = -1
            # NEW: counts objects actually PICKED, not search attempts. Needed
            # now that a search attempt can be abandoned mid-way (priority
            # interrupt) without a pick happening -- navigate_and_place_in_boundary()
            # and the "is this the last object" check must use the real pick
            # count, not the attempt count, or they'd pick the wrong placement
            # zone / end the mission one object early after an interrupt.
            picked_count = 0
            while remaining_targets:
                if stop_requested:
                    break

                # NEW: a GUI "Disconnect" click is honored here too, between
                # objects -- same checkpoint granularity as the existing 'q'
                # key check inside the per-frame loop below.
                if yolo_server.is_remote_stop():
                    print("Disconnect received from GUI -- stopping mission "
                          "(between objects).")
                    stop_requested = True
                    break

                # NEW: honor a pending priority request from the GUI, if any,
                # by moving that class to the front of what's left to search for.
                priority_request = yolo_server.get_priority_override()
                if priority_request in remaining_targets:
                    remaining_targets.remove(priority_request)
                    remaining_targets.insert(0, priority_request)
                    print("Priority override from GUI: searching for %s next." % priority_request)

                target_color = remaining_targets.pop(0)
                idx += 1

                print("\n=== Now looking for the %s object ===" % target_color)
                forward_steps = 0
                last_center = IMG_CENTER_X
                line_miss_count = 0
                picked = False
                global num, old_x, old_y
                num = 0
                old_x, old_y = 0, 0

                # Grace period: right after turning back around toward the line's
                # start, the line itself is close/visible and could otherwise be
                # falsely detected as the object. Skip color-checking until we've
                # walked most of the distance it took to reach the previous
                # object (only relevant for the 2nd+ object in the sequence).
                # DISABLED: this was only needed for the old ColorDetect() (LAB
                # color-mask) picking, where the line could be falsely matched as
                # the object if colors were similar. YOLO (YoloObjectDetect)
                # identifies objects by shape/class, not color, so the line can
                # never be mistaken for a target object -- this safety skip is no
                # longer needed and was instead causing YOLO checks to be skipped
                # for too long when starting mid-line, letting the robot walk
                # past the object. Commented out (not deleted) in case you want
                # it back later.
                # if last_successful_steps is not None:
                #     min_steps_before_detect = max(last_successful_steps - 5, 0)
                # else:
                #     min_steps_before_detect = 0
                min_steps_before_detect = 0  # NEW: grace period always off -- YOLO checks every frame

                print("DEBUG main: idx=%d target_color=%s last_successful_steps=%s "
                      "min_steps_before_detect=%d" %
                      (idx, target_color, str(last_successful_steps), min_steps_before_detect))

                # === NEW: track consecutive "line lost" frames WHILE still inside
                # the grace period. If the line disappears (a strong signal we're
                # near/at the end of the line) before min_steps_before_detect is
                # satisfied, the old code would just keep walking blind, silently
                # skipping ColorDetect() the whole time -- exactly the bug reported.
                # This clears the grace period early so detection can still fire.
                grace_period_line_lost_count = 0
                GRACE_PERIOD_LOST_LIMIT = 8

                # === Edge-triggered debug state (prints ONLY when something
                # actually changes, instead of every single frame -- flooding the
                # console with a print on every frame was what broke the first
                # loop's detection timing in the previous version). ==============
                prev_in_grace_period = None
                prev_seen = False
                prev_following = False
                interrupted = False  # NEW: set True if a priority override abandons this search

                while not picked:
                    img = camera.frame
                    if img is None:
                        time.sleep(0.02)
                        continue

                    # NEW: check for a priority override that should interrupt THIS
                    # search, not just reorder what comes after it. Safe to act on
                    # every frame here because we're still only searching/following
                    # the line -- nothing physical (pick/turn/return/place) starts
                    # until "stable" further below, and this check always runs
                    # before that on any given frame, so it can never fire once a
                    # pick is already underway.
                    priority_request = yolo_server.get_priority_override()
                    if priority_request and priority_request != target_color \
                            and priority_request in TARGET_SEQUENCE:
                        print("Priority override from GUI: abandoning search for %s, "
                              "switching to %s now." % (target_color, priority_request))
                        if target_color not in remaining_targets:
                            remaining_targets.insert(0, target_color)
                        if priority_request in remaining_targets:
                            remaining_targets.remove(priority_request)
                        remaining_targets.insert(0, priority_request)
                        interrupted = True
                        break

                    img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)  # lens undistort (was missing)
                    display_frame = img.copy()

                    in_grace_period = forward_steps < min_steps_before_detect

                    if in_grace_period != prev_in_grace_period:
                        if in_grace_period:
                            print("Grace period active (forward_steps=%d < min=%d)" %
                                  (forward_steps, min_steps_before_detect))
                            print("Skipping ColorDetect")
                        else:
                            print("Grace period ended (forward_steps=%d >= min=%d)" %
                                  (forward_steps, min_steps_before_detect))
                            print("Entering ColorDetect")
                        prev_in_grace_period = in_grace_period

                    if in_grace_period:
                        # still in the grace period -- just follow the line,
                        # don't even look for the object yet
                        stable, world_x, world_y, seen = False, None, None, False
                    else:
                        # MODIFIED: ColorDetect(display_frame, target_color) -> YoloObjectDetect(target_color).
                        # target_color now holds a YOLO class name (see TARGET_SEQUENCE above), and the
                        # pixel center comes from the PC's YOLO pipeline instead of a local color threshold.
                        # Everything downstream (stability check, pick trigger, world coords) is unchanged.
                        stable, world_x, world_y, seen = YoloObjectDetect(target_color)

                    if seen != prev_seen:
                        print("Object seen" if seen else "Object no longer seen")
                        prev_seen = seen

                    if stable:
                        print("Object stable -- picking directly (tilt confirmation disabled)")
                        ik.stand(ik.initial_pos, t=500)
                        time.sleep(0.5)

                        # DISABLED: second-look tilt confirmation was rejecting
                        # valid detections because the object left the camera's
                        # view during the tilt maneuver (see confirm_detection_at_tilt()
                        # below -- left intact/unused in case you want it back later).
                        # Now we trust the first STABLE detection and pick right away.
                        cv2.putText(display_frame, "Picking %s object..." % target_color, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        publish_debug_frame(display_frame)  # NEW: stream this exact annotated frame to the GUI
                        cv2.imshow("Line Follow + Fetch", display_frame)
                        cv2.waitKey(1)
                        pick_object(world_x, world_y)
                        picked = True
                        last_successful_steps = forward_steps

                        # Turn around to face back toward the start of the line
                        turn_180()
                        return_via_line(camera, forward_steps)

                        # Search for the boundary at the return point (creep +
                        # sweep with the reliable calibrated view) and place
                        # the object INSIDE it.
                        # NEW: navigate_and_place_in_boundary/is_last_object now use
                        # picked_count (objects actually picked) instead of idx
                        # (search attempts) -- idx can now run ahead of actual
                        # picks after a priority interrupt, picked_count can't.
                        navigate_and_place_in_boundary(camera, mapx, mapy, picked_count)
                        print("%s object done." % target_color)

                        # If there's another object still to fetch, turn back
                        # around to face the line's end again before the next
                        # loop iteration starts searching for it.
                        picked_count += 1
                        is_last_object = (picked_count == len(TARGET_SEQUENCE))
                        if not is_last_object:
                            print("Turning back around to head to the line's end again...")
                            turn_180()

                    elif seen:
                        # Object candidate visible but not confirmed stable yet ->
                        # hold still, do NOT keep walking (avoids overshoot)
                        prev_following = False
                        cv2.putText(display_frame, "%s object in view - holding still..." % target_color,
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                    else:
                        if not prev_following:
                            print("Following line")
                            prev_following = True

                        status, last_center, stepped, line_miss_count = follow_line_step(
                            img, last_center, miss_count=line_miss_count, draw_on=display_frame)
                        if stepped:
                            forward_steps += 1

                        # === NEW: safety valve for the grace period ==========================
                        # If we're still inside the grace period (haven't reached
                        # min_steps_before_detect yet) AND the line keeps coming back "LOST",
                        # that's a strong sign we're near/at the true end of the line already --
                        # the previous object's step-count was a bad estimate for this trip.
                        # Force the grace period to end NOW instead of silently continuing to
                        # skip ColorDetect() until we've walked straight past (or through) the
                        # actual object.
                        if in_grace_period and "LOST" in status:
                            grace_period_line_lost_count += 1
                            if grace_period_line_lost_count >= GRACE_PERIOD_LOST_LIMIT:
                                print("Grace period cleared early -- line lost %d times before "
                                      "reaching the expected step count (forward_steps=%d). "
                                      "ColorDetect() will run starting next frame." %
                                      (GRACE_PERIOD_LOST_LIMIT, forward_steps))
                                min_steps_before_detect = forward_steps  # ends the grace period now
                                grace_period_line_lost_count = 0
                        elif in_grace_period:
                            grace_period_line_lost_count = 0
                        # ======================================================================

                        label = status + " steps=%d" % forward_steps
                        if forward_steps < min_steps_before_detect:
                            label += " (grace period, not checking yet)"
                        cv2.putText(display_frame, label, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    publish_debug_frame(display_frame)  # NEW: stream this exact annotated frame to the GUI
                    cv2.imshow("Line Follow + Fetch", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    # NEW: yolo_server.is_remote_stop() honors the GUI's Disconnect
                    # button at the same checkpoint as the existing 'q' key -- i.e.
                    # between frames while searching/following the line. Like 'q',
                    # this does NOT interrupt a pick/place/return already in progress
                    # (those are uninterruptible physical motions on the Pi, same as
                    # before) -- it stops at the next safe frame-level checkpoint.
                    if key == ord('q') or yolo_server.is_remote_stop():
                        stop_requested = True
                        break

            if not stop_requested:
                print("\nAll objects picked and placed. Task complete.")
            else:
                print("\nMission stopped (GUI Disconnect or 'q' key).")
                ik.stand(ik.initial_pos, t=500)  # NEW: settle to a safe pose before waiting for next Connect

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        ik.stand(ik.initial_pos, t=500)
        camera.camera_close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
