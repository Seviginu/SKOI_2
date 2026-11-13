import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from baseline_tracker import BaselineTracker

ROOT = Path(__file__).resolve().parent
COLORS = {"initial": (70, 220, 70), "recovered": (70, 220, 70),
          "ecc": (255, 220, 0), "flow": (0, 140, 255)}


def project(points, matrix):
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), matrix).reshape(-1, 2)

def normalize(image):
    image = image.astype(np.float32)
    detail = image - cv2.GaussianBlur(image, (0, 0), 7)
    contrast = np.sqrt(cv2.GaussianBlur(detail * detail, (0, 0), 7) + 25)
    return detail / contrast

class PlanarTracker:

    def __init__(self, gray):
        h, w = gray.shape
        if min(h, w) < 32:
            raise ValueError('Слишком маленький кадр')
        self.shape = gray.shape
        self.corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        self.matrix = np.eye(3)
        self.sift = cv2.SIFT_create(nfeatures=3000, contrastThreshold=0.015)
        reference, descriptors = [], []
        for scale in (1.0, 0.5, 0.25):
            if min(w, h) * scale < 16:
                continue
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            points, features = self.sift.detectAndCompute(small, None)
            if features is not None:
                factor = np.float32([w / small.shape[1], h / small.shape[0]])
                reference.extend([np.float32(p.pt) * factor for p in points])
                descriptors.append(features)
        if len(reference) < 8:
            raise ValueError('В шаблоне мало деталей: выберите текстурный объект')
        self.reference, self.descriptors = np.float32(reference), np.vstack(descriptors)
        factor = min(1.0, 256 / max(w, h))
        size = (max(16, round(w * factor)), max(16, round(h * factor)))
        self.template = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        self.normalized = normalize(self.template)
        self.scale = np.diag([w / size[0], h / size[1], 1.0])
        self.inverse_scale = np.linalg.inv(self.scale)
        self.mask = np.zeros_like(self.template)
        mx, my = max(2, size[0] // 20), max(2, size[1] // 20)
        self.mask[my:-my, mx:-mx] = 255
        self.previous = gray
        self.sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        self.points = self.seed(gray)

    def seed(self, gray):
        polygon = project(self.corners, self.matrix)
        center = polygon.mean(axis=0)
        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, np.int32(center + (polygon - center) * 0.92), 255)
        return cv2.goodFeaturesToTrack(gray, 300, 0.005, 4, mask=mask, blockSize=5)

    def valid(self, matrix, previous=None):
        if matrix is None or not np.isfinite(matrix).all():
            return False
        denominator = np.c_[self.corners, np.ones(4)] @ matrix[2]
        if np.min(np.abs(denominator)) < 1e-08 or np.any(denominator * denominator[0] <= 0):
            return False
        polygon = project(self.corners, matrix)
        height, width = self.shape
        if (not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon)
                or np.abs(polygon).max() > 3 * max(width, height)):
            return False
        area = cv2.contourArea(polygon, oriented=True)
        if not 150 < area < 4 * width * height:
            return False
        if previous is not None:
            old = project(self.corners, previous)
            ratio = area / cv2.contourArea(old)
            movement = np.linalg.norm(polygon - old, axis=1).max()
            if not 0.65 < ratio < 1.55 or movement > np.sqrt(area) * 0.35 + 8:
                return False
        return True

    def flow(self, gray):
        if self.points is None or len(self.points) < 8:
            return None
        options = dict(winSize=(25, 25), maxLevel=4, criteria=(3, 30, 0.01))
        target, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None, **options)
        if target is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, target, None, **options)
        if backward is None:
            return None
        error = np.linalg.norm((backward - self.points).reshape(-1, 2), axis=1)
        good = (status.ravel() == 1) & (back_status.ravel() == 1) & (error < 1.0)
        if good.sum() < 8:
            return None
        source, target = self.points[good], target[good]
        delta, mask = cv2.findHomography(source, target, cv2.RANSAC, 1.5)
        if delta is None or mask.sum() < 8 or mask.mean() < 0.5:
            return None
        matrix = delta @ self.matrix
        return matrix if self.valid(matrix, self.matrix) else None

    def warp(self, gray, matrix):
        size = (self.template.shape[1], self.template.shape[0])
        return cv2.warpPerspective(gray, matrix @ self.scale, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)

    def appearance(self, gray, matrix):
        warped = self.warp(gray, matrix)
        scores = []
        for reference, image in ((self.template, warped), (self.normalized, normalize(warped))):
            a = cv2.GaussianBlur(reference.astype(np.float32), (5, 5), 0)[self.mask > 0]
            b = cv2.GaussianBlur(image.astype(np.float32), (5, 5), 0)[self.mask > 0]
            a, b = a - a.mean(), b - b.mean()
            scores.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-06)))
        return max(scores)

    def refine(self, gray, matrix, max_correction=None):
        if not self.valid(matrix):
            return None
        warped = self.warp(gray, matrix)
        variants = ((self.normalized, normalize(warped), 0.6), (self.template, warped, 0.65))
        for reference, image, threshold in variants:
            try:
                score, delta = cv2.findTransformECC(
                    reference, image, np.eye(3, dtype=np.float32), cv2.MOTION_HOMOGRAPHY,
                    (3, 35, 1e-4), self.mask, 5)
            except cv2.error:
                continue
            refined = matrix @ self.scale @ delta @ self.inverse_scale
            if score >= threshold and self.valid(refined, matrix):
                if max_correction is not None:
                    correction = project(self.corners, refined) - project(self.corners, matrix)
                    if np.linalg.norm(correction, axis=1).max() > max_correction:
                        continue
                return refined
        return None

    def locate(self, gray):
        points, features = self.sift.detectAndCompute(gray, None)
        if features is None or len(features) < 2:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(self.descriptors, features, k=2)
        matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.78 * p[1].distance]
        unique = {}
        for match in sorted(matches, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        if len(matches) < 8:
            return None
        source = np.float32([self.reference[m.queryIdx] for m in matches])
        target = np.float32([points[m.trainIdx].pt for m in matches])
        matrix, mask = cv2.findHomography(source, target, cv2.USAC_MAGSAC, 2.0)
        if matrix is None or mask.sum() < 8:
            return None
        return matrix if self.valid(matrix) else None

    def update(self, gray):
        if gray.shape != self.shape:
            raise ValueError('Размер кадров должен быть постоянным')
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        if sharpness < 0.2 * self.sharpness:
            self.points = None
            self.previous = gray
            return None, 'blurred'
        motion = self.flow(gray)
        if self.points is None:
            matrix = self.refine(gray, self.locate(gray))
            method = 'recovered'
        else:
            proposal = motion if motion is not None else self.matrix
            limit = 3.0 * max(self.shape) / 640 if motion is not None else None
            matrix = self.refine(gray, proposal, limit)
            method = 'ecc'
        if matrix is None and motion is not None and (self.appearance(gray, motion) >= 0.5):
            matrix, method = motion, 'flow'
        if matrix is None and self.points is not None:
            matrix = self.refine(gray, self.locate(gray))
            method = 'recovered'
        if matrix is not None:
            self.matrix = matrix / matrix[2, 2]
            self.points = self.seed(gray)
            self.sharpness = 0.9 * self.sharpness + 0.1 * sharpness
        else:
            self.points = None
        self.previous = gray
        return (project(self.corners, self.matrix), method) if matrix is not None else (None, 'lost')


def paint(frame, polygon, method, points):
    if polygon is None:
        return
    cv2.polylines(frame, [np.int32(np.rint(polygon))], True, COLORS[method], 2, cv2.LINE_AA)
    for point in points.reshape(-1, 2) if points is not None else []:
        center = tuple(np.int32(np.rint(point)))
        cv2.circle(frame, center, 3, (15, 15, 15), -1, cv2.LINE_AA)
        cv2.circle(frame, center, 2, (0, 255, 255), -1, cv2.LINE_AA)


def run(video, name):
    capture = cv2.VideoCapture(str(ROOT / "test-videos" / video))
    writers, counts = [], {}
    with tempfile.TemporaryDirectory(dir=ROOT) as temp:
        try:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Не удалось открыть {video}")
            h, w = frame.shape[:2]
            fps = capture.get(cv2.CAP_PROP_FPS)
            fps = fps if np.isfinite(fps) and fps > 0 else 25.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            baseline = BaselineTracker(gray) if name == "mona-lisa" else None
            tracker = PlanarTracker(gray)
            names = [name] + ([name + "-comparison"] if baseline is not None else [])
            for stem in names:
                writer = cv2.VideoWriter(str(Path(temp) / (stem + ".avi")),
                                         cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
                writers.append(writer)
                if not writer.isOpened():
                    raise ValueError("Не удалось создать видео")
            while ok:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                old = baseline.update(gray) if baseline is not None else None
                polygon, method = (tracker.update(gray) if counts else
                                   (project(tracker.corners, tracker.matrix), "initial"))
                if baseline is not None:
                    comparison = frame.copy()
                    if old is not None:
                        cv2.polylines(comparison, [np.int32(np.rint(old))], True,
                                      (210, 60, 220), 5, cv2.LINE_AA)
                    paint(comparison, polygon, method, tracker.points)
                    writers[1].write(comparison)
                paint(frame, polygon, method, tracker.points)
                writers[0].write(frame)
                counts[method] = counts.get(method, 0) + 1
                ok, frame = capture.read()
        finally:
            capture.release()
            for writer in writers:
                writer.release()
        for stem in names:
            output = Path(temp) / (stem + ".mp4")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(Path(temp) / (stem + ".avi")),
                            "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", str(output)], check=True)
            output.replace(ROOT / "results" / output.name)
    print(f"{name}: {sum(counts.values())} кадров, {counts}")


if __name__ == "__main__":
    (ROOT / "results").mkdir(exist_ok=True)
    try:
        run("mona-lisa.avi", "mona-lisa")
        run("video_2026-11-13_14-20-13.mp4", "banknote")
    except (ValueError, OSError, cv2.error, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Ошибка: {error}")
