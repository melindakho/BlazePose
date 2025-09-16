import os
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path

import mediapipe as mp

LEFT_LANDMARKS = [
    4, 5, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32
]

RIGHT_LANDMARKS = [
    1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
]

def estimate_poses(args):
    mp_pose = mp.solutions.pose.Pose(
        model_complexity=args.complexity,
        enable_segmentation=False,
    )

    input_dir = Path(args.input)
    video_paths = []
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    for ext in video_extensions:
        video_paths.extend(sorted(list(input_dir.glob(f"*{ext}"))))
    
    os.makedirs(args.output, exist_ok=True)
    for video_path in tqdm(video_paths):
        filename = os.path.basename(video_path)
        out_path = os.path.join(args.output, filename)

        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_pose.process(rgb)

            if results.pose_landmarks:
                for i, point in enumerate(results.pose_landmarks.landmark):
                    x = int(point.x * frame.shape[1])
                    y = int(point.y * frame.shape[0])
                    if i in LEFT_LANDMARKS:
                        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                    elif i in RIGHT_LANDMARKS:
                        cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)
                    else:
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            out.write(frame)
            if args.render:
                cv2.imshow("Pose Estimation", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved to {out_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", "-r", action="store_true", help="Render the output video with landmarks.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Directory to input video file(s).")
    parser.add_argument("--output", "-o", type=str, default="out", help="Directory to save output.")
    parser.add_argument("--complexity", "-c", type=int, choices=[0, 1, 2], default=1, help="Model complexity: 0, 1, or 2.")
    args = parser.parse_args()
    args.output = os.path.join(args.output, f"complexity_{args.complexity}")
    return args

if __name__ == "__main__":
    args = parse_args()
    estimate_poses(args)