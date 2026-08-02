import glob
import os

import cv2

VIDEO_DIR = "./videos"
OUTPUT_DIR = "./images"
TARGET_FPS = 15
JPEG_QUALITY = 95


def get_next_counter(output_dir: str) -> int:
    pattern = os.path.join(output_dir, "frame*.jpg")
    max_id = 0
    for path in glob.glob(pattern):
        name = os.path.basename(path)
        stem, ext = os.path.splitext(name)
        if ext.lower() != ".jpg":
            continue
        if not stem.startswith("frame"):
            continue
        digits = stem[5:]
        if digits.isdigit():
            max_id = max(max_id, int(digits))
    return max_id + 1


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files: list[str] = []
    for p in video_patterns:
        video_files.extend(glob.glob(os.path.join(VIDEO_DIR, p)))
        video_files.extend(glob.glob(os.path.join(VIDEO_DIR, p.upper())))
    video_files = sorted(set(video_files))

    counter = get_next_counter(OUTPUT_DIR)
    total_videos_processed = 0
    total_frames_extracted = 0

    if not video_files:
        print(f"No videos found in: {VIDEO_DIR}")
        print(
            f"Summary: total videos processed=0, total frames extracted=0, output folder={OUTPUT_DIR}"
        )
        return

    for video_path in video_files:
        filename = os.path.basename(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: unreadable video skipped: {filename}")
            continue

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if native_fps is None or native_fps <= 0:
            native_fps = float(TARGET_FPS)

        if native_fps <= TARGET_FPS:
            skip_interval = 1
        else:
            skip_interval = max(1, int(round(native_fps / TARGET_FPS)))

        extracted_this_video = 0
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % skip_interval == 0:
                out_name = f"frame{counter:06d}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                wrote = cv2.imwrite(
                    out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]
                )
                if wrote:
                    counter += 1
                    extracted_this_video += 1
                    total_frames_extracted += 1

            frame_idx += 1

        cap.release()
        total_videos_processed += 1
        print(
            f"Video: {filename} | native fps: {native_fps:.2f} | "
            f"total frames: {total_frames} | frames extracted: {extracted_this_video}"
        )

    print(
        f"Summary: total videos processed={total_videos_processed}, "
        f"total frames extracted={total_frames_extracted}, output folder={OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
