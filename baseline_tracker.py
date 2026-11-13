import cv2
import numpy as np


class BaselineTracker:
    def __init__(self, gray):
        h, w = gray.shape
        self.corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.02)
        reference, descriptors = [], []
        for scale in (1.0, 0.5, 0.25):
            if min(h, w) * scale < 16:
                continue
            small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            points, features = self.sift.detectAndCompute(small, None)
            if features is not None:
                sx, sy = w / small.shape[1], h / small.shape[0]
                reference.extend([(p.pt[0] * sx, p.pt[1] * sy) for p in points])
                descriptors.append(features)
        if len(reference) < 8:
            raise ValueError("В шаблоне мало деталей")
        self.reference, self.descriptors = np.float32(reference), np.vstack(descriptors)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.previous = self.state = None
        self.index = 0

    def estimate(self, source, target, shape):
        if len(source) < 8:
            return None, None
        matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        if matrix is None or mask is None or mask.sum() < max(6, len(source) * 0.5):
            return None, None
        polygon = cv2.perspectiveTransform(self.corners, matrix)
        h, w = shape
        if (not np.isfinite(polygon).all() or np.abs(polygon).max() > 2 * max(w, h)
                or not cv2.isContourConvex(polygon) or not 100 < cv2.contourArea(polygon) < 4 * w * h):
            return None, None
        inliers = mask.ravel().astype(bool)
        if cv2.contourArea(cv2.convexHull(source[inliers])) < 0.1 * cv2.contourArea(self.corners):
            return None, None
        return polygon, (source[inliers], target[inliers])

    def locate(self, gray):
        points, features = self.sift.detectAndCompute(gray, None)
        if features is None or len(features) < 2:
            return None, None
        pairs = self.matcher.knnMatch(self.descriptors, features, k=2)
        matches = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        unique = {}
        for match in sorted(matches, key=lambda m: m.distance):
            unique.setdefault(match.trainIdx, match)
        matches = list(unique.values())
        source = np.float32([self.reference[m.queryIdx] for m in matches])
        target = np.float32([points[m.trainIdx].pt for m in matches])
        return self.estimate(source, target, gray.shape)

    def follow(self, gray):
        source, points = self.state
        options = dict(winSize=(21, 21), maxLevel=3)
        target, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, points, None, **options)
        if target is None:
            return None, None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, target, None, **options)
        if backward is None:
            return None, None
        good = (status.ravel() == 1) & (back_status.ravel() == 1)
        good &= np.linalg.norm(backward - points, axis=1) < 1.0
        return self.estimate(source[good], target[good], gray.shape)

    def update(self, gray):
        polygon = None
        if self.state is None or self.index % 15 == 0:
            polygon, detected = self.locate(gray)
            if detected is not None:
                self.state = detected
        if polygon is None and self.state is not None:
            polygon, self.state = self.follow(gray)
        if self.index == 0:
            polygon = self.corners.copy()
        self.previous = gray
        self.index += 1
        return polygon
