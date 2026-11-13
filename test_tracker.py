import unittest
from unittest.mock import patch
import cv2
import numpy as np
from tracker import COLORS, PlanarTracker, paint, project

def texture():
    rng = np.random.default_rng(42)
    image = np.full((200, 320), 100, np.uint8)
    for _ in range(180):
        x, y = rng.integers([5, 5], [315, 195])
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 8)), int(rng.integers(20, 240)), -1)
    cv2.putText(image, 'PLANAR TEST', (20, 110), 0, 1.1, 255, 2, cv2.LINE_AA)
    return cv2.GaussianBlur(image, (3, 3), 0.5)

def render(image, matrix):
    return cv2.warpPerspective(image, matrix, (640, 400))

class TrackingTest(unittest.TestCase):

    def setUp(self):
        cv2.setRNGSeed(42)
        self.image = texture()
        self.initial = np.float64([[1, 0, 40], [0, 1, 50], [0, 0, 1]])
        self.tracker = PlanarTracker(self.image)
        first = render(self.image, self.initial)
        self.tracker.shape = first.shape
        self.tracker.matrix = self.initial.copy()
        self.tracker.previous = first
        self.tracker.sharpness = float(cv2.Laplacian(first, cv2.CV_32F).var())
        self.tracker.points = self.tracker.seed(first)

    def test_perspective_scale_and_lighting(self):
        errors = []
        for i in range(1, 31):
            angle = i * 0.01
            scale = 1 - i * 0.015
            c, s = (np.cos(angle) * scale, np.sin(angle) * scale)
            matrix = np.float64([[c, -s, 40 + i * 3], [s, c, 50 + i], [i * 6e-06, i * -2e-06, 1]])
            gray = render(self.image, matrix)
            gradient = np.linspace(0.65, 1.15, gray.shape[1])[None, :]
            gray = np.uint8(np.clip(gray * gradient + 12, 0, 255))
            polygon, method = self.tracker.update(gray)
            self.assertIsNotNone(polygon, (i, method))
            expected = project(self.tracker.corners, matrix)
            errors.append(np.linalg.norm(polygon - expected, axis=1).max())
        self.assertLess(max(errors), 4.0, errors)

    def test_disappearance_and_global_recovery(self):
        blank = np.zeros((400, 640), np.uint8)
        self.assertIsNone(self.tracker.update(blank)[0])
        noise = np.random.default_rng(2).integers(0, 256, blank.shape, dtype=np.uint8)
        self.assertIsNone(self.tracker.update(noise)[0])
        matrix = np.float64([[0.65, -0.1, 310], [0.1, 0.65, 160], [0.0002, 0, 1]])
        polygon, method = self.tracker.update(render(self.image, matrix))
        self.assertIsNotNone(polygon, method)
        error = np.linalg.norm(polygon - project(self.tracker.corners, matrix), axis=1).max()
        self.assertLess(error, 4.0)

    def test_blur_does_not_deform_reference(self):
        before = self.tracker.matrix.copy()
        frame = render(self.image, self.initial)
        blurred = cv2.GaussianBlur(frame, (51, 51), 12)
        polygon, method = self.tracker.update(blurred)
        self.assertIsNone(polygon)
        self.assertEqual(method, 'blurred')
        np.testing.assert_array_equal(self.tracker.matrix, before)
        self.assertIsNotNone(self.tracker.update(frame)[0])

    def test_bad_template_and_geometry(self):
        with self.assertRaises(ValueError):
            PlanarTracker(np.zeros((100, 100), np.uint8))
        with self.assertRaises(ValueError):
            PlanarTracker(np.zeros((16, 16), np.uint8))
        reflected = np.float64([[-1, 0, 500], [0, 1, 0], [0, 0, 1]])
        self.assertFalse(self.tracker.valid(reflected))
        horizon = np.float64([[1, 0, 0], [0, 1, 0], [0.01, 0, -1]])
        self.assertFalse(self.tracker.valid(horizon))

    def test_ecc_cannot_pull_one_corner_away_from_motion(self):
        tracker = self.tracker
        frame = render(self.image, tracker.matrix)
        corners = project(tracker.corners, tracker.matrix)
        corners[2] += (20, 10)
        bad = cv2.getPerspectiveTransform(tracker.corners, corners)
        delta = tracker.inverse_scale @ np.linalg.inv(tracker.matrix) @ bad @ tracker.scale
        with patch('tracker.cv2.findTransformECC', return_value=(0.99, delta.astype(np.float32))):
            result = tracker.refine(frame, tracker.matrix, max_correction=3.0)
        self.assertIsNone(result)
        result = tracker.refine(frame, tracker.matrix, max_correction=3.0)
        self.assertIsNotNone(result)

    def test_steep_perspective_with_lighting_changes(self):
        start = project(self.tracker.corners, self.initial)
        finish = np.float32([[190, 75], [380, 155], [325, 315], [145, 255]])
        errors = []
        for i in range(1, 51):
            corners = start + (finish - start) * min(i / 30, 1)
            matrix = cv2.getPerspectiveTransform(self.tracker.corners, corners)
            gray = render(self.image, matrix).astype(np.float32)
            lighting = 1 + 0.25 * np.sin(i * 1.7) * np.linspace(-1, 1, 640)[None, :]
            gray = np.uint8(np.clip(gray * lighting + 10, 0, 255))
            polygon, method = self.tracker.update(gray)
            self.assertIsNotNone(polygon, (i, method))
            errors.append(np.linalg.norm(polygon - corners, axis=1).max())
        self.assertLess(max(errors), 4.0, errors)

    def test_colors_and_no_overlay_when_lost(self):
        self.assertEqual(COLORS['initial'], COLORS['recovered'])
        self.assertEqual(len({COLORS['recovered'], COLORS['ecc'], COLORS['flow']}), 3)
        frame = np.zeros((100, 100, 3), np.uint8)
        paint(frame, None, 'lost', np.float32([[[20, 20]]]))
        self.assertFalse(frame.any())
        polygon = np.float32([[10, 10], [90, 10], [90, 90], [10, 90]])
        for method in ['recovered', 'ecc', 'flow']:
            frame.fill(0)
            paint(frame, polygon, method, None)
            np.testing.assert_array_equal(frame[10, 50], COLORS[method])
if __name__ == '__main__':
    unittest.main()
