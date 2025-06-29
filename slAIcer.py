import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def extract_frames(video_path, fps=30):
	"""Extract frames and return list of (frame_index, image)."""
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise ValueError(f"Could not open video file: {video_path}")
	video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
	interval = int(round(video_fps / fps))
	frames = []
	idx = 0
	saved = 0
	while True:
		ret, frame = cap.read()
		if not ret:
			break
		if idx % interval == 0:
			frames.append((saved, frame.copy()))
			saved += 1
		idx += 1
	cap.release()
	return frames

def auto_detect_board_y(frame):
	"""
	Automatically detect the likely diving board y position in the frame.
	Returns the y coordinate (int) of the detected board or None if not found.
	"""
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (9,9), 0)
	edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
	lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=90, minLineLength=frame.shape[1]//5, maxLineGap=20)
	y_lines = []
	h, w = frame.shape[:2]
	if lines is not None:
		for l in lines:
			x1, y1, x2, y2 = l[0]
			# Only consider (nearly) horizontal lines away from the bottom
			if abs(y2 - y1) < 10 and min(y1,y2)>h//6 and max(y1,y2)<int(h*0.97):
				y_lines.append((y1 + y2) // 2)
		if y_lines:
			# The lowest horizontal except for the bottom of the image
			return int(sorted(y_lines)[-1])
	return None

def detect_start_end(frames, board_y_norm, water_y_norm=0.95, takeoff_thresh=0.9, debug=False):
	"""
	Given normalized board Y and water line, returns start_idx, end_idx.
	Entry is detected using both pose and splash (motion) cues.
	If debug=True, overlays splash zone and splash detection info.
	"""
	pose = mp_pose.Pose(static_image_mode=True)
	start, end = None, None
	consecutive_entry = 0
	entry_required = 3
	missed_frames = 0
	max_missed = 5

	entry_landmarks = [
		mp_pose.PoseLandmark.NOSE,
		mp_pose.PoseLandmark.LEFT_WRIST,
		mp_pose.PoseLandmark.RIGHT_WRIST,
		mp_pose.PoseLandmark.LEFT_ANKLE,
		mp_pose.PoseLandmark.RIGHT_ANKLE,
		mp_pose.PoseLandmark.LEFT_KNEE,
		mp_pose.PoseLandmark.RIGHT_KNEE,
		mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
		mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
	]

	last_above_water_idx = None
	after_takeoff = False

	# For splash detection
	prev_gray = None
	splash_detected_idx = None
	splash_band_height =  70 # pixels
	splash_thresh = 7.5  # tune this value if needed

	for idx, frame in frames:
		h, w = frame.shape[:2]
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# --- Splash detection ---
		board_y_px = int(board_y_norm * h)
		water_y_px = int(water_y_norm * h)
		band_bot = water_y_px
		band_top = max(water_y_px - splash_band_height, 0)
		splash_band = gray[band_top:band_bot, :]
		splash_score = 0
		splash_this_frame = False
		if prev_gray is not None:
			prev_band = prev_gray[band_top:band_bot, :]
			diff = cv2.absdiff(splash_band, prev_band)
			splash_score = np.mean(diff)
			if splash_score > splash_thresh and after_takeoff:
				splash_detected_idx = idx
				splash_this_frame = True
		prev_gray = gray

		# --- Debug overlays ---
		if debug:
			overlay = frame.copy()
			# Draw splash band (now just above water line)
			cv2.rectangle(overlay, (0, band_top), (w, band_bot), (255,0,255), 2)
			# If splash detected, fill band
			if splash_this_frame:
				cv2.rectangle(overlay, (0, band_top), (w, band_bot), (0,0,255), -1)
				cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
				cv2.putText(frame, f"SPLASH DETECTED! ({splash_score:.1f})", (20, band_top-10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
			else:
				cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
				cv2.putText(frame, f"Splash score: {splash_score:.1f}", (20, band_top-10),
					cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)
			# Draw board and water lines
			cv2.line(frame, (0, board_y_px), (w, board_y_px), (0,255,255), 2)
			cv2.line(frame, (0, water_y_px), (w, water_y_px), (255,128,0), 2)
			# Show frame for debug
			cv2.imshow("Debug Splash Detection", frame)
			key = cv2.waitKey(1)
			if key == 27: # ESC to quit debug
				debug = False
				cv2.destroyWindow("Debug Splash Detection")

		# --- Pose detection ---
		res = pose.process(rgb)
		if not res.pose_landmarks:
			missed_frames += 1
			if after_takeoff and last_above_water_idx is not None and (idx - last_above_water_idx) <= max_missed:
				if missed_frames >= max_missed:
					# Only allow entry if splash was detected recently
					if splash_detected_idx is not None and abs(idx - splash_detected_idx) <= 2:
						end = idx
						break
			continue

		missed_frames = 0
		lm = res.pose_landmarks.landmark

		# ankles for takeoff
		ay_l = lm[mp_pose.PoseLandmark.LEFT_ANKLE].y
		ay_r = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y

		# detect takeoff
		if start is None:
			if ay_l < board_y_norm - takeoff_thresh or ay_r < board_y_norm - takeoff_thresh:
				start = idx
				after_takeoff = True
		else:
			# Robust entry: count how many entry landmarks are below water line
			below = 0
			above = 0
			total = 0
			for lidx in entry_landmarks:
				ly = lm[lidx].y
				if ly > water_y_norm:
					below += 1
				else:
					above += 1
				total += 1
			# If majority of entry landmarks are below water line AND splash detected, count as entry
			if below >= total // 2 and splash_detected_idx is not None and abs(idx - splash_detected_idx) <= 2:
				consecutive_entry += 1
				if consecutive_entry >= entry_required:
					end = idx
					break
			else:
				consecutive_entry = 0
				# If majority are above water, update last_above_water_idx
				if above > total // 2:
					last_above_water_idx = idx

	pose.close()
	if debug:
		cv2.destroyAllWindows()
	return start, end

def annotate_and_save(frames, start, end, board_y_px, water_y_px, output_path, fps=30):
	"""
	Draw horizontal lines, pose landmarks, and markers at start/end, then save video.
	Freezes for 0.5s at start and end.
	"""
	if not frames:
		raise ValueError("No frames to write.")
	h, w = frames[0][1].shape[:2]
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

	mp_pose = mp.solutions.pose
	mp_drawing = mp.solutions.drawing_utils
	pose = mp_pose.Pose(static_image_mode=True)

	freeze_frames = int(0.1 * fps)
	for idx, frame in frames:
		# draw pose landmarks
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		res = pose.process(rgb)
		if res.pose_landmarks:
			mp_drawing.draw_landmarks(
				frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS,
				mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
				mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
			)
		# draw board line
		# cv2.line(frame, (0, board_y_px), (w, board_y_px), (0,255,255), 2)
		# draw water line
		# cv2.line(frame, (0, water_y_px), (w, water_y_px), (255,128,0), 2)
		if idx == start:
			cv2.putText(frame, "TAKEOFF", (50,50),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
			for _ in range(freeze_frames):
				out.write(frame)
		elif idx == end:
			cv2.putText(frame, "ENTRY", (50,100),
				cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
			for _ in range(freeze_frames):
				out.write(frame)
		out.write(frame)

	pose.close()
	out.release()

def get_board_y_px(frame):
	"""
	Show the frame and let the user click to select the board y coordinate.
	Returns the y coordinate (int).
	Displays a moving horizontal line under the mouse.
	"""
	clicked = {'y': None}
	mouse_y = {'y': None}
	clone = frame.copy()

	def on_mouse(event, x, y, flags, param):
		mouse_y['y'] = y
		if event == cv2.EVENT_LBUTTONDOWN:
			clicked['y'] = y
			cv2.destroyWindow("Select Board Line")

	cv2.namedWindow("Select Board Line")
	cv2.setMouseCallback("Select Board Line", on_mouse)
	print("Click on the board line in the image window. Press 'a' to auto-detect.")

	while clicked['y'] is None:
		display = clone.copy()
		if mouse_y['y'] is not None:
			cv2.line(display, (0, mouse_y['y']), (display.shape[1], mouse_y['y']), (0,255,255), 2)
		cv2.imshow("Select Board Line", display)
		key = cv2.waitKey(20) & 0xFF
		if key == 27:	# ESC to quit
			break
		elif key == ord('a'):
			board_y = auto_detect_board_y(display)
			if board_y is not None:
				clicked['y'] = board_y
				print(f"Auto-detected board y: {board_y}")
				cv2.line(display, (0, board_y), (display.shape[1], board_y), (255,0,0), 2)
				cv2.imshow("Select Board Line", display)
				cv2.waitKey(700)
				break
	cv2.destroyAllWindows()
	return clicked['y']

def calibrate_takeoff_thresh(frames, board_y_norm, initial_thresh=0.1, water_y_norm=0.95, max_frames=100):
    """
    Interactive tool to calibrate takeoff_thresh.
    Use UP/DOWN arrows to adjust threshold, 'q' to finish.
    """
    pose = mp_pose.Pose(static_image_mode=True)
    thresh = initial_thresh
    idx = 0
    while idx < min(max_frames, len(frames)):
        _, frame = frames[idx]
        h, w = frame.shape[:2]
        board_y_px = int(board_y_norm * h)
        water_y_px = int(water_y_norm * h)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        ankle_l, ankle_r = None, None
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            ankle_l = lm[mp_pose.PoseLandmark.LEFT_ANKLE].y
            ankle_r = lm[mp_pose.PoseLandmark.RIGHT_ANKLE].y
        # Overlay
        disp = frame.copy()
        cv2.line(disp, (0, board_y_px), (w, board_y_px), (0,255,255), 2)
        cv2.line(disp, (0, water_y_px), (w, water_y_px), (255,128,0), 2)
        cv2.putText(disp, f"takeoff_thresh: {thresh:.3f} (UP/DOWN to adjust, q to quit)", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if ankle_l is not None and ankle_r is not None:
            ankle_l_px = int(ankle_l * h)
            ankle_r_px = int(ankle_r * h)
            cv2.circle(disp, (w//2-40, ankle_l_px), 8, (0,255,0), -1)
            cv2.circle(disp, (w//2+40, ankle_r_px), 8, (0,255,0), -1)
            # Show if takeoff would be detected
            if ankle_l < board_y_norm - thresh or ankle_r < board_y_norm - thresh:
                cv2.putText(disp, "TAKEOFF DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 3)
        cv2.imshow("Calibrate Takeoff Threshold", disp)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        elif key == 82:  # UP arrow
            thresh += 0.01
        elif key == 84:  # DOWN arrow
            thresh = max(0.01, thresh - 0.01)
        elif key == 83:  # RIGHT arrow
            idx = min(idx+1, len(frames)-1)
        elif key == 81:  # LEFT arrow
            idx = max(idx-1, 0)
        else:
            idx += 1
    cv2.destroyWindow("Calibrate Takeoff Threshold")
    return thresh

def main():
	video_path = "data/raw_videos/dive.mp4"
	# Step 1: extract frames at 30 fps
	frames = extract_frames(video_path, fps=30)

	# Step 2: user specifies board_y_px on the first frame
	first_frame = frames[0][1]
	board_y_px = get_board_y_px(first_frame)
	if board_y_px is None:
		print("No board line selected. Exiting.")
		return

	# convert to normalized y
	board_y_norm = board_y_px / first_frame.shape[0]
	water_y_norm = 0.95  # bottom 5% is water
	water_y_px = int(water_y_norm * first_frame.shape[0])

	# Step 3: detect start/end
	# takeoff_thresh = calibrate_takeoff_thresh(frames, board_y_norm, initial_thresh=0.1,
	# 	water_y_norm=water_y_norm, max_frames=100)
	# print(f"Calibrated takeoff threshold: {takeoff_thresh:.3f}")
	start_idx, end_idx = detect_start_end(
		frames, board_y_norm, water_y_norm, takeoff_thresh=0.1)

	print(f"Detected takeoff at frame {start_idx}, entry at frame {end_idx}")

	# Step 4: annotate and re-encode video
	output_path = "data/pose_analysis/output_annotated.mp4"
	annotate_and_save(frames, start_idx, end_idx,
		board_y_px, water_y_px, output_path, fps=30)
	print(f"Annotated video saved to {output_path}")

if __name__ == "__main__":
	main()
