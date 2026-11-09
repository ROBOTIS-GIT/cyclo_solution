"""Depth-only cup region and upright rim estimation; no robot commands."""

import cv2
import numpy as np


def cup_candidates(meters, keep, intrinsics, diameter=0.10):
    """Diagnostic heuristics, not semantic recognition or rim reconstruction.

    Image connectivity proposes regions; metric spans reject unlikely sizes.
    No closing/filling: do not invent depth across the gripper's occlusion.
    """
    fx, fy, cx, cy = intrinsics
    # Cut depth discontinuities so a few image pixels cannot join cup and table.
    connected = keep.copy()
    for axis in (0, 1):
        before, after = [slice(None)] * 2, [slice(None)] * 2
        before[axis], after[axis] = slice(None, -1), slice(1, None)
        before, after = tuple(before), tuple(after)
        with np.errstate(invalid="ignore"):
            edge = keep[before] & keep[after] & (np.abs(meters[before] - meters[after]) > 0.015)
        connected[before] &= ~edge
        connected[after] &= ~edge
    clean = cv2.morphologyEx(connected.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    candidates = []
    height, width = meters.shape
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < 80 or x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            continue  # Clipped regions cannot provide a reliable size estimate.
        rows, cols = np.nonzero(labels == label)
        z = meters[rows, cols]
        points = np.column_stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z))
        spans = np.percentile(points, 95, axis=0) - np.percentile(points, 5, axis=0)
        projected = np.array([w / fx, h / fy]) * np.median(z)
        major, minor = max(projected), min(projected)
        fill = area / (w * h)
        if not (
            0.5 * diameter <= major <= 1.8 * diameter
            and 0.25 * diameter <= minor <= 1.5 * diameter
            and major / minor <= 3
            and fill >= 0.3
            and max(spans) <= 0.22
        ):
            continue
        score = float(np.exp(-abs(major - diameter) / (0.5 * diameter)) * fill)
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates.append(
            {
                "xyz": np.median(points, axis=0).tolist(),
                "pixel": [int(np.median(cols)), int(np.median(rows))],
                "span_m": spans.tolist(),
                "bbox": [x, y, w, h],
                "score": score,
                "contours": contours,
                "_mask": mask,
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def rotation_matrix(q):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q)
    if not np.all(np.isfinite(q)) or norm < 0.99 or norm > 1.01:
        raise ValueError("Invalid transform quaternion")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def deproject(meters, mask, intrinsics):
    rows, cols = np.nonzero(mask)
    z = meters[rows, cols]
    fx, fy, cx, cy = intrinsics
    points = np.column_stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z))
    return points, np.column_stack((cols, rows))


def _fit_free_radius_rim(points, inner_radius=0.045, wall=0.005):
    """Fit a horizontal rim in gravity-aligned coordinates, rejecting short arcs.

    Only the top 8 mm of the region is considered. This does not reconstruct an
    invisible rim from the body centroid. A failed fit is never a motion target.
    """
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 50:
        return None, "RIM_UNCERTAIN: too few points"
    top = np.percentile(points[:, 2], 98)
    band = points[(points[:, 2] >= top - 0.008) & (points[:, 2] <= top + 0.003)]
    if len(band) < 30:
        return None, "RIM_UNCERTAIN: insufficient upper edge depth"
    # Bound work deterministically; do not repeatedly use one angular cluster.
    if len(band) > 1000:
        band = band[np.linspace(0, len(band) - 1, 1000).astype(int)]
    xy = band[:, :2]
    rng = np.random.default_rng(71)
    best = None
    for _ in range(180):
        sample = xy[rng.choice(len(xy), 3, replace=False)]
        a = 2 * (sample[1:] - sample[0])
        if abs(np.linalg.det(a)) < 1e-5:
            continue
        center = np.linalg.solve(a, np.sum(sample[1:] ** 2, axis=1) - np.sum(sample[0] ** 2))
        radius = np.linalg.norm(center - sample[0])
        if not inner_radius - 0.004 <= radius <= inner_radius + wall + 0.004:
            continue
        errors = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
        inliers = errors < 0.003
        if best is None or np.count_nonzero(inliers) > best[0]:
            best = (np.count_nonzero(inliers), inliers)
    if best is None or best[0] < 30 or best[0] / len(band) < 0.55:
        return None, "RIM_UNCERTAIN: upper edge does not match cup radius"
    inliers = best[1]
    selected = xy[inliers]
    # Refit and re-evaluate consensus, conditioning, coverage, and vertical spread.
    a = np.column_stack((2 * selected, np.ones(len(selected))))
    solution, _, _, _ = np.linalg.lstsq(a, np.sum(selected**2, axis=1), rcond=None)
    center = solution[:2]
    radius_squared = solution[2] + np.dot(center, center)
    if radius_squared <= 0:
        return None, "RIM_UNCERTAIN: invalid circle"
    radius = np.sqrt(radius_squared)
    errors = np.abs(np.linalg.norm(xy - center, axis=1) - radius)
    inliers = errors < 0.003
    if np.count_nonzero(inliers) < 30 or np.mean(inliers) < 0.55:
        return None, "RIM_UNCERTAIN: unstable circle fit"
    angles = np.sort(
        np.mod(np.arctan2(xy[inliers, 1] - center[1], xy[inliers, 0] - center[0]), 2 * np.pi)
    )
    coverage = 2 * np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))
    sectors = len(np.unique(np.floor(angles / (np.pi / 6)).astype(int)))
    scatter = np.std(band[inliers, 2])
    residual = np.sqrt(np.mean(errors[inliers] ** 2))
    if (
        not inner_radius - 0.004 <= radius <= inner_radius + wall + 0.004
        or coverage < np.deg2rad(120)
        or sectors < 5
        or scatter > 0.0035
    ):
        return None, "RIM_UNCERTAIN: short arc, incorrect size, or non-level edge"
    xyz = np.r_[center, np.median(band[inliers, 2])]
    # This is a quality bound for gating, not a calibrated uncertainty guarantee.
    uncertainty = max(0.006, residual * 3 + scatter + 0.003)
    return {
        "center": xyz,
        "radius": radius,
        "points": band[inliers],
        "coverage_deg": float(np.rad2deg(coverage)),
        "error": float(uncertainty),
        "method": "measured_radius",
    }, "RIM_FIT"


def _fit_known_radius_rim(points, inner_radius, wall):
    """Estimate a center from measured partial arcs using the known rim annulus.

    Fit the middle of the configured inner/outer radii. Both centers of each
    chord are tested; competing centers remain invalid, not arbitrarily chosen.
    """
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 50:
        return None, "too few points"
    top = np.percentile(points[:, 2], 98)
    band = points[(points[:, 2] >= top - .008) & (points[:, 2] <= top + .003)]
    if len(band) < 30:
        return None, "insufficient upper edge depth"
    if len(band) > 1000:
        band = band[np.linspace(0, len(band) - 1, 1000).astype(int)]
    xy = band[:, :2]
    radius = inner_radius + wall / 2
    rng = np.random.default_rng(71)
    solutions = []
    for _ in range(180):
        pair = xy[rng.choice(len(xy), 2, replace=False)]
        delta = pair[1] - pair[0]
        length = np.linalg.norm(delta)
        if length < radius * .6 or length >= 2 * radius:
            continue
        normal = np.array([-delta[1], delta[0]]) / length
        offset = np.sqrt(radius ** 2 - length ** 2 / 4) * normal
        for center in (pair.mean(axis=0) + offset, pair.mean(axis=0) - offset):
            # Robust fixed-radius least squares; never synthesize occluded points.
            for _ in range(8):
                vectors = center - xy
                lengths = np.linalg.norm(vectors, axis=1)
                inliers = (np.abs(lengths - radius) < .003) & (lengths > 1e-6)
                if np.count_nonzero(inliers) < 30:
                    break
                jacobian = vectors[inliers] / lengths[inliers, None]
                update, _, _, _ = np.linalg.lstsq(jacobian, lengths[inliers] - radius, rcond=None)
                center = center - update
                if np.linalg.norm(update) < 1e-6:
                    break
            vectors = xy - center
            lengths = np.linalg.norm(vectors, axis=1)
            errors = np.abs(lengths - radius)
            inliers = (errors < .003) & (lengths > 1e-6)
            count = np.count_nonzero(inliers)
            if count < 30 or count / len(band) < .55:
                continue
            directions = vectors[inliers] / lengths[inliers, None]
            eigenvalues = np.linalg.eigvalsh(directions.T @ directions / count)
            if eigenvalues[0] < .03:
                continue  # Nearly collinear patches cannot constrain both center axes.
            residual = float(np.sqrt(np.mean(errors[inliers] ** 2)))
            solutions.append((count, residual, center, inliers, eigenvalues[0]))
    if not solutions:
        return None, "known radius lacks well-conditioned depth support"
    solutions.sort(key=lambda item: (-item[0], item[1]))
    count, residual, center, inliers, conditioning = solutions[0]
    for other_count, other_residual, other_center, _, _ in solutions[1:]:
        if (other_count >= count * .9 and other_residual <= residual + .0005
                and np.linalg.norm(other_center - center) > .008):
            return None, "known radius gives ambiguous centers"
    angles = np.sort(np.mod(np.arctan2(xy[inliers, 1] - center[1],
                                      xy[inliers, 0] - center[0]), 2 * np.pi))
    coverage = 2 * np.pi - np.max(np.diff(np.r_[angles, angles[0] + 2 * np.pi]))
    scatter = float(np.std(band[inliers, 2]))
    if coverage < np.deg2rad(75) or scatter > .0035:
        return None, f"known-radius arc={np.rad2deg(coverage):.0f}deg height_spread={scatter:.4f}m"
    # Include annulus uncertainty and sensitivity to arc geometry, not just fit residual.
    uncertainty = max(.006, 3 * residual + scatter + .003,
                      (wall / 2 + residual) / np.sqrt(conditioning))
    if uncertainty > .015:
        return None, f"known-radius center uncertainty={uncertainty:.4f}m exceeds .015m"
    return {
        "center": np.r_[center, np.median(band[inliers, 2])],
        "radius": radius,
        "points": band[inliers],
        "coverage_deg": float(np.rad2deg(coverage)),
        "error": float(uncertainty),
        "method": "known_radius",
    }, "RIM_FIT_KNOWN_RADIUS"


def fit_upright_rim(points, inner_radius=0.045, wall=0.005):
    fit, reason = _fit_free_radius_rim(points, inner_radius, wall)
    if fit is not None:
        return fit, reason
    fit, fallback_reason = _fit_known_radius_rim(points, inner_radius, wall)
    if fit is not None:
        return fit, fallback_reason
    return None, reason + "; " + fallback_reason


def fit_rim_candidates(meters, candidates, intrinsics, rotation, translation,
                       inner_radius=0.045, wall=0.005):
    """Validate each visible region before deciding whether the scene is ambiguous.

    Missing/self-masked depth stays missing. A valid circle must be supported by
    the measured arc; size scores alone never authorize a motion target.
    """
    evaluated = []
    for candidate in candidates:
        points, _ = deproject(meters, candidate["_mask"], intrinsics)
        fit, detail = fit_upright_rim(points @ rotation.T + translation, inner_radius, wall)
        evaluated.append((candidate, fit, detail))
    valid = [(candidate, fit) for candidate, fit, _ in evaluated if fit is not None]
    if len(valid) == 1:
        candidate, fit = valid[0]
        return candidate, fit, "RIM_FIT", evaluated
    if len(valid) > 1:
        return None, None, "RIM_UNCERTAIN: multiple geometrically valid rims", evaluated
    detail = "; ".join(f"candidate {i + 1}: {reason.removeprefix('RIM_UNCERTAIN: ')}"
                       for i, (_, _, reason) in enumerate(evaluated))
    return None, None, "RIM_UNCERTAIN: " + (detail or "no candidates"), evaluated
