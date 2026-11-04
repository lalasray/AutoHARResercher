from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from autohar_researcher.imutube_signal import pose_to_imu, required_landmarks, validate_pose_generation


def extract_segments(path, config, profile):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'Cannot decode video: {path}')
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or not 1 <= fps <= 240:
        capture.release()
        raise ValueError('Video needs a valid constant frame rate between 1 and 240 Hz')
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=config['pose_model']),
        running_mode=vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=config.get('confidence', .7),
        min_pose_presence_confidence=config.get('confidence', .7),
        min_tracking_confidence=config.get('confidence', .7))
    needed = sorted({i for c in profile['channel_map'] for i in required_landmarks(c['joint'])})
    segment, frame_index, last_time = [], 0, None
    previous_center = None
    try:
        with vision.PoseLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index >= config.get('max_frames', 54000):
                    raise ValueError('Video exceeds configured frame limit; curate a shorter clip')
                timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
                expected = frame_index * 1000 / fps

                if not np.isfinite(timestamp) or abs(timestamp - expected) > max(2., 100 / fps):
                    raise ValueError('Video timestamps disagree with constant frame rate; transcode the clip first')
                if last_time is not None and timestamp <= last_time:
                    raise ValueError('Video timestamps must increase')
                last_time = timestamp
                frame_index += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(round(timestamp)))
                valid = len(result.pose_world_landmarks) == 1
                if valid:
                    landmarks = result.pose_world_landmarks[0]
                    valid = all(landmarks[i].visibility >= config.get('confidence', .7)
                                and landmarks[i].presence >= config.get('confidence', .7) for i in needed)
                if valid:
                    image_pose = result.pose_landmarks[0]
                    center = np.mean([[image_pose[i].x, image_pose[i].y] for i in (23, 24)], axis=0)

                    if previous_center is not None and np.linalg.norm(center - previous_center) > config.get('max_center_jump', .15):
                        if segment:
                            yield np.asarray(segment), fps
                        segment = []
                    previous_center = center
                    pose = np.asarray([[v.x, v.y, v.z] for v in landmarks])
                    if np.isfinite(pose).all():
                        segment.append(pose)
                        continue
                if segment:
                    yield np.asarray(segment), fps
                segment, previous_center = [], None
            if segment:
                yield np.asarray(segment), fps
    finally:
        capture.release()


def generate(config, request):
    validate_pose_generation(request['profile'], diagnostic_only=config.get('diagnostic_only', False), return_provenance=True)
    rng = np.random.default_rng(request['seed'])
    tracks, rejected = [], 0
    for video in request['video_paths']:
        for pose, fps in extract_segments(video, config, request['profile']):
            try:
                converted = pose_to_imu(pose, fps, request['sampling_hz'], request['profile'],
                                     gravity=config.get('gravity', [0, 9.80665, 0]),
                                     smoothing_seconds=config.get('smoothing_seconds', .2),
                                     coordinate_mode='hip-relative', diagnostic_only=True, return_provenance=True)
                signal = converted['samples']
            except ValueError:
                rejected += 1
                continue
            if len(signal) >= request['window_steps']:
                tracks.append(signal)

    available = np.asarray([len(t) - request['window_steps'] + 1 for t in tracks], dtype=np.int64)
    total = int(available.sum())
    if total < request['count']:
        raise ValueError(f'Only {total} valid window starts for {request["count"]} requested samples; '
                         f'{rejected} pose segments rejected. Supply longer/clearer curated videos.')
    chosen = rng.choice(total, size=request['count'], replace=False)
    ends = np.cumsum(available)
    output = []
    for index in chosen:
        track = int(np.searchsorted(ends, index, side='right'))
        start = int(index - (ends[track - 1] if track else 0))
        output.append(tracks[track][start:start + request['window_steps']])
    return np.asarray(output, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Video-to-IMU worker')
    parser.add_argument('--config', required=True)
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    request = json.loads(Path(args.request).read_text())
    np.save(args.output, generate(config, request))
    provenance = validate_pose_generation(request['profile'], diagnostic_only=config.get('diagnostic_only', False), return_provenance=True)
    Path(args.output).with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print('Video-to-IMU extraction complete', file=sys.stderr)


if __name__ == '__main__':
    main()
