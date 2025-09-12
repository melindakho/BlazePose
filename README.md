# Pose Estimation with BlazePose (MediaPipe)

## Description

This project uses [MediaPipe](https://mediapipe.dev/)'s BlazePose for human pose estimation on video files. It processes videos, detects pose landmarks, and saves annotated output videos.

## Installation

```bash
conda env create -f environment.yaml
conda activate blazepose
```

## Example Videos

Example videos are **not included** in the repository.  
Download them directly using the command below:

```bash
gdown https://drive.google.com/drive/folders/117mGYVpyfPYuKmhTAdGDGLqhAxHQjIhF -O examples --folder
```

## How to Run

Render the output video during processing:

```bash
python main.py --input examples/PD_anterior.mp4 --render
```

Set model complexity (0, 1, or 2):

```bash
python main.py --input examples/PD_anterior.mp4 --complexity 1
```

## Arguments

- `--input`, `-i`: Path to input video file (required)
- `--output`, `-o`: Directory to save output video (default: `out`)
- `--complexity`, `-c`: Model complexity (0, 1, or 2; default: 1)
- `--render`, `-r`: Render the output video with landmarks

## Output

Processed videos are saved in the specified output directory, organized by model complexity.
