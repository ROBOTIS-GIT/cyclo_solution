"""Closed-form IK of the 7-joint arm: joint 7 swept on a grid, joints 1-6 solved exactly.

Shoulder axes 1-3 meet in one point (S), wrist axes 5-6 in another (W). With q7 fixed the
target gives W; q4 follows from |SW|, q5 and q6 turn one known vector onto another, and q1-q3
are what is left of the shoulder rotation. Every solution on the q7 grid is found, so a pose
without any really has none (within the grid step). Line joins one solution per waypoint with
the least joint motion: shortest_path() over every COARSE-th waypoint, then the waypoints
between are filled on the same branches and the whole path is smoothed.
"""
import math
import time

from ffw_pick.kinematics import axis_matrix, transform
import numpy as np

Q7_STEP = math.radians(1.0)
AXES = ((0, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0))
CELL_KEY = 10000  # cell index packing in pairs() (cells per joint range << this)
CHECK_TOL = 1e-6  # m and rotation-matrix units: solutions are re-checked by forward kinematics
GAP = 0.02  # rad: neighbours on a branch further apart than this get a q7 between them
COARSE = 2  # Line searches every this many waypoints (5 mm points -> 1 cm search steps)
FILL_TRIES = 50  # Line: searched paths whose points between could not be filled
FILL_SLACK = 0.01  # rad: a point between may lengthen its two steps this much before every
# solution there is tried
SMOOTH_SWEEPS = 3  # passes of smooth() over a path
SMOOTH_SAMPLES = 9  # q7 values tried per waypoint in a pass, within +-one grid step
WRAP_GAP = 1.0  # rad: a bigger difference is a +-pi wrap or another branch, not refined
REFINE_LEVELS = 6  # halvings of the q7 step at most (1 deg -> 0.016 deg)


def rot(axis, q):
    """(n, 3, 3) rotations about x/y/z (axis 0/1/2) by the angles q."""
    q = np.asarray(q, float)
    c, s = np.cos(q), np.sin(q)
    i, j = ((1, 2), (2, 0), (0, 1))[axis]
    out = np.zeros(q.shape + (3, 3))
    out[..., axis, axis] = 1.0
    out[..., i, i] = out[..., j, j] = c
    out[..., i, j], out[..., j, i] = -s, s
    return out


class ArmIK:
    """Every IK solution of the 7 arm joints of chain, shoulder first (the lift and other
    joints stay as given)."""

    def __init__(self, chain, joints, margin=0.0, q7_step=Q7_STEP):
        by_name = {j['name']: j for j in chain.joints}
        arm = [by_name[n] for n in joints]
        for j, axis in zip(arm, AXES):
            if not np.allclose(j['axis'], axis) or np.abs(j['rpy']).max() > 1e-9:
                raise ValueError(f'{j["name"]}: axis or origin rotation differs from the model')
        k = chain.joints.index(arm[-1])
        if any(j['type'] != 'fixed' or np.abs(j['rpy']).max() > 1e-9
               for j in chain.joints[k + 1:]):
            raise ValueError('the tip must sit on joint 7 by fixed, unrotated joints')
        # S lies on axes 1 and 3, W on axis 5: those offsets run along the axes
        if np.abs(arm[1]['xyz'][[0, 2]]).max() > 1e-9 or np.abs(arm[2]['xyz'][:2]).max() > 1e-9 \
                or np.abs(arm[5]['xyz'][:2]).max() > 1e-9:
            raise ValueError('shoulder axes 1-3 or wrist axes 5-6 do not meet')
        self.names = list(joints)
        self.before = chain.joints[:chain.joints.index(arm[0]) + 1]
        self.s_offset = arm[1]['xyz']
        self.c = arm[2]['xyz'] + arm[3]['xyz']  # S -> axis 4, link 3 frame
        self.e = arm[4]['xyz'] + arm[5]['xyz']  # axis 4 -> W, link 4 frame
        self.t7 = arm[6]['xyz']  # W -> joint 7, link 6 frame
        self.tip = sum((j['xyz'] for j in chain.joints[k + 1:]), np.zeros(3))
        limits = np.array([j['limit'] for j in arm])
        self.lower, self.upper = limits[:, 0] + margin, limits[:, 1] - margin
        self.q7_step = q7_step
        self.q7 = np.arange(self.lower[6], self.upper[6] + 1e-12, q7_step)

    def base(self, values):
        """Frame of joint 1 (before its rotation) for these lift/body joint values."""
        out = np.eye(4)
        for j in self.before:
            out = out @ j['static']
            q = values.get(j['name'], 0.0)
            if j is self.before[-1]:
                break
            if j['type'] in ('revolute', 'continuous'):
                out = out @ transform(axis_matrix(j['axis'], q), np.zeros(3))
            elif j['type'] == 'prismatic':
                out = out @ transform(np.eye(3), j['axis'] * q)
        return out

    def solve(self, values, position, rotation, q7=None):
        """(Q (n, 7), branch (n,)): every solution within the limits (see solve_many); or the
        solutions at the given q7 only."""
        if q7 is not None:
            return self.solve_at(values, position, rotation, q7)
        return self.solve_many(values, [position], rotation)[0]

    def solve_many(self, values, positions, rotation):
        """solve() for several positions at one rotation, computed together: [(Q, branch)].
        The q7 grid is halved (REFINE_LEVELS times at most) where a branch changes between
        missing, outside and inside the limits (a piece narrower than the grid step is found
        that way), and where two inside neighbours differ by more than GAP (so the solutions
        lie about evenly along each branch)."""
        positions = np.asarray(positions, float).reshape(-1, 3)
        layer = np.repeat(np.arange(len(positions)), len(self.q7))
        q7 = np.tile(self.q7, len(positions))
        q, state = self.branches(values, positions[layer], rotation, q7)
        finest = self.q7_step / 2 ** REFINE_LEVELS
        for _ in range(REFINE_LEVELS):
            same = (layer[:-1] == layer[1:]) & (np.diff(q7) > 1.5 * finest)
            gap = np.abs(np.diff(q, axis=1)).max(axis=2)
            inside = (state[:, :-1] == 2) & (state[:, 1:] == 2)
            want = (state[:, :-1] != state[:, 1:]) | (inside & (gap > GAP) & (gap < WRAP_GAP))
            pick = want.any(axis=0) & same
            if not pick.any():
                break
            mid = (q7[:-1][pick] + q7[1:][pick]) / 2
            mid_layer = layer[:-1][pick]
            q_mid, state_mid = self.branches(values, positions[mid_layer], rotation, mid)
            layer, q7 = np.concatenate((layer, mid_layer)), np.concatenate((q7, mid))
            order = np.lexsort((q7, layer))
            layer, q7 = layer[order], q7[order]
            q = np.concatenate((q, q_mid), axis=1)[:, order]
            state = np.concatenate((state, state_mid), axis=1)[:, order]
        branch, column = np.nonzero(state == 2)
        rows, branch, owner = q[branch, column], branch, layer[column]
        order = np.argsort(owner, kind='stable')
        cuts = np.searchsorted(owner[order], np.arange(1, len(positions)))
        return [(r, b) for r, b in zip(np.split(rows[order], cuts), np.split(branch[order], cuts))]

    def solve_at(self, values, position, rotation, q7):
        """solve() at the given q7 values only."""
        q, state = self.branches(values, position, rotation, q7)
        branch, column = np.nonzero(state == 2)
        return q[branch, column], branch

    def branches(self, values, position, rotation, q7):
        """(q (8, n, 7), state (8, n)) for the 8 branches at the q7 values (n,), at position
        (3,) or one position per q7 (n, 3): state 0 = no solution, 1 = outside the limits,
        2 = inside."""
        frame = self.base(values)
        r0 = frame[:3, :3]
        s = frame[:3, 3] + r0 @ self.s_offset
        rt = r0.T @ rotation
        q7 = np.atleast_1d(np.asarray(q7, float))
        p = np.broadcast_to((np.asarray(position, float) - s) @ r0, (len(q7), 3))
        r7 = rot(0, q7)
        r6 = rt @ r7.transpose(0, 2, 1)
        d = p - rt @ self.tip - r6 @ self.t7  # S -> W
        v = np.einsum('nji,nj->ni', r6, d)  # the same in link 6
        c, e = self.c, self.e
        a, b = c[0] * e[0] + c[2] * e[2], c[0] * e[2] - c[2] * e[0]
        h = math.hypot(a, b)
        ratio4 = (((d * d).sum(1) - c @ c - e @ e) / 2 - c[1] * e[1]) / h
        ok4 = np.abs(ratio4) <= 1.0
        base4, spread4 = math.atan2(b, a), np.arccos(np.clip(ratio4, -1, 1))
        out, found = np.zeros((8, len(q7), 7)), np.zeros((8, len(q7)), bool)
        for i4, sign4 in enumerate((1.0, -1.0)):
            q4 = base4 + sign4 * spread4
            r4 = rot(1, q4)
            u = np.einsum('nji,j->ni', r4, c) + e  # S -> W in link 4
            # q6 makes (Ry(q6) v)_z = u_z, then q5 turns u onto Ry(q6) v about z
            m = np.hypot(v[:, 0], v[:, 2])
            ratio6 = u[:, 2] / np.maximum(m, 1e-12)
            ok6 = ok4 & (np.abs(ratio6) <= 1.0) & (m > 1e-12)
            g, spread6 = np.arctan2(-v[:, 0], v[:, 2]), np.arccos(np.clip(ratio6, -1, 1))
            for i6, sign6 in enumerate((1.0, -1.0)):
                q6 = g + sign6 * spread6
                w = np.einsum('nij,nj->ni', rot(1, q6), v)
                q5 = np.arctan2(u[:, 1], u[:, 0]) - np.arctan2(w[:, 1], w[:, 0])
                r123 = r6 @ (r4 @ rot(2, q5) @ rot(1, q6)).transpose(0, 2, 1)
                sb = np.clip(-r123[:, 1, 2], -1, 1)
                for i2, q2 in enumerate((np.arcsin(sb), math.pi - np.arcsin(sb))):
                    sign = np.where(np.cos(q2) < 0, -1.0, 1.0)
                    q1 = np.arctan2(sign * r123[:, 0, 2], sign * r123[:, 2, 2])
                    q3 = np.arctan2(sign * r123[:, 1, 0], sign * r123[:, 1, 1])
                    k = 4 * i4 + 2 * i6 + i2
                    out[k] = np.remainder(np.column_stack((q1, q2, q3, q4, q5, q6, q7)) + math.pi,
                                          2 * math.pi) - math.pi
                    found[k] = ok6
        inside = found & np.all((out >= self.lower) & (out <= self.upper), axis=2)
        # re-checked by forward kinematics where it counts (outside ones only steer the grid)
        k, col = np.nonzero(inside)
        if len(k):
            inside[k, col] = self.check(out[k, col], rt, p[col])
        return out, np.where(inside, 2, np.where(found, 1, 0))

    def check(self, q, rt, p):
        """True where forward kinematics of q (n, 7) reaches rt, p (joint 1 frame, from S)."""
        r3 = rot(1, q[:, 0]) @ rot(0, q[:, 1]) @ rot(2, q[:, 2])
        r4 = r3 @ rot(1, q[:, 3])
        r6 = r4 @ rot(2, q[:, 4]) @ rot(1, q[:, 5])
        r7 = r6 @ rot(0, q[:, 6])
        reach = r3 @ self.c + r4 @ self.e + r6 @ self.t7 + r7 @ self.tip
        return (np.linalg.norm(reach - p, axis=1) < CHECK_TOL) \
            & (np.abs(r7 - rt).max(axis=(1, 2)) < CHECK_TOL)

    def smooth(self, values, points, rotation, path, branches, start, fixed, jump):
        """path (n, 7) through points moved along its own branches to continuous q7 so that the
        steps are even (least sum of squared steps; the grid alone zigzags). The first point
        of a free start keeps its place (its move from start was chosen by the search). Every
        other point at once, then the rest, SMOOTH_SWEEPS times."""
        path, start = np.array(path, float), np.asarray(start, float)
        points, branches = np.asarray(points, float), np.asarray(branches)
        offsets = np.linspace(-self.q7_step, self.q7_step, SMOOTH_SAMPLES)
        n, first = len(path), (0 if fixed else 1)
        for _ in range(SMOOTH_SWEEPS):
            for parity in (0, 1):
                idx = np.arange(first + parity, n, 2)
                if not len(idx):
                    continue
                k, samples = len(idx), len(offsets)
                q, state = self.branches(values, np.repeat(points[idx], samples, axis=0), rotation,
                                         (path[idx, 6][:, None] + offsets).ravel())
                cols = np.arange(k * samples)
                own = np.repeat(branches[idx], samples)
                cand = q[own, cols].reshape(k, samples, 7)
                ok = (state[own, cols] == 2).reshape(k, samples)
                prev = np.where((idx > 0)[:, None], path[np.maximum(idx - 1, 0)], start)[:, None]
                nxt = path[np.minimum(idx + 1, n - 1)][:, None]
                last = (idx + 1 >= n)[:, None]
                cost = ((cand - prev) ** 2).sum(2) + np.where(last, 0.0, ((nxt - cand) ** 2).sum(2))
                ok &= np.abs(cand - prev).max(axis=2) <= jump
                ok &= last | (np.abs(nxt - cand).max(axis=2) <= jump)
                # the same piece of the branch only (not a far solution with the same signs)
                ok &= np.abs(cand - path[idx][:, None]).max(axis=2) <= jump
                cost = np.where(ok, cost, np.inf)
                best = cost.argmin(axis=1)
                good = np.isfinite(cost[np.arange(k), best])
                path[idx[good]] = cand[np.arange(k), best][good]
        return path


def line_points(start, end, step):
    """Points every step (or less) from start to end, both included."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    count = max(1, int(math.ceil(np.linalg.norm(end - start) / step)))
    return [start + (end - start) * i / count for i in range(count + 1)]


def pairs(a, b, width):
    """Index pairs (i, j) of rows a (n, 7), b (m, 7) that may be within width on every joint:
    cells of width on b's three most spread joints, neighbouring cells included."""
    if not len(a) or not len(b):
        return np.zeros(0, int), np.zeros(0, int)
    cols = np.argsort(np.ptp(b, axis=0))[-3:]
    cell_a = np.floor(a[:, cols] / width).astype(np.int64)
    cell_b = np.floor(b[:, cols] / width).astype(np.int64)

    def pack(cells):
        return (cells[:, 0] * CELL_KEY + cells[:, 1]) * CELL_KEY + cells[:, 2]
    key_b = pack(cell_b)
    order = np.argsort(key_b)
    sorted_b = key_b[order]
    out_i, out_j = [], []
    for shift in np.array(np.meshgrid((-1, 0, 1), (-1, 0, 1), (-1, 0, 1))).reshape(3, -1).T:
        key = pack(cell_a + shift)
        lo = np.searchsorted(sorted_b, key, 'left')
        counts = np.searchsorted(sorted_b, key, 'right') - lo
        out_i.append(np.repeat(np.arange(len(a)), counts))
        offsets = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        out_j.append(order[np.repeat(lo, counts) + offsets])
    return np.concatenate(out_i), np.concatenate(out_j)


def shortest_path(layers, start, jump, banned=None, fixed=False):
    """(rows, cost, '') through one solution per layer with the least joint motion, or
    (None, inf, why). layers: [(Q, branch)] per waypoint; start: (7,) joints now. fixed: the
    path leaves from start itself (the layers follow it); else it starts anywhere in layer 0
    and pays the move from start there. Cost = sum of the largest joint change per step; a step
    over jump rad on any joint is not allowed. banned: per layer, rows left out."""
    start = np.asarray(start, float)
    alive = [np.ones(len(q), bool) if banned is None else ~banned[i]
             for i, (q, _) in enumerate(layers)]

    def where(i):
        return f'{i * 100 // max(len(layers) - 1, 1)}% of the line'
    if fixed:
        prev, cost, first = start[None], np.zeros(1), 0
    else:
        prev, first = layers[0][0], 1
        cost = np.where(alive[0], np.abs(prev - start).max(axis=1) if len(prev) else 0.0, np.inf)
    back = []
    for i in range(first, len(layers)):
        if not np.isfinite(cost).any():
            break
        q = layers[i][0]
        a, b = pairs(prev, q, jump)
        step = np.abs(prev[a] - q[b]).max(axis=1)
        keep = (step <= jump) & alive[i][b] & np.isfinite(cost[a])
        a, b, total = a[keep], b[keep], cost[a[keep]] + step[keep]
        new, parent = np.full(len(q), np.inf), np.full(len(q), -1)
        order = np.lexsort((total, b))
        head = order[np.r_[True, b[order][1:] != b[order][:-1]]] if len(order) else order
        new[b[head]], parent[b[head]] = total[head], a[head]
        back.append(parent)
        prev, cost = q, new
        if not np.isfinite(cost).any():
            if not len(q):
                return None, math.inf, f'no IK solution at {where(i)}'
            if not alive[i].any():
                return None, math.inf, f'every solution collides at {where(i)}'
            return None, math.inf, f'no continuous path at {where(i)} (joint step > {jump} rad)'
    if not np.isfinite(cost).any():  # layer 0 of a free start
        if not len(layers[0][0]):
            return None, math.inf, f'no IK solution at {where(0)}'
        return None, math.inf, f'every solution collides at {where(0)}'
    rows = [int(np.argmin(cost))]
    for parent in reversed(back):
        rows.append(int(parent[rows[-1]]))
    rows.reverse()
    return (rows[1:] if fixed else rows), float(cost.min()), ''


class Line:
    """Least-motion joint path through points (fingertip waypoints) at one rotation.

    path() searches every COARSE-th point (and the last) with shortest_path, fills the points
    between on the chosen branches and smooths the whole path. ban() leaves out a searched
    solution (it collides) so that the next path() finds the next shortest one.
    """

    def __init__(self, arm, values, points, rotation, coarse=COARSE):
        self.arm, self.values, self.rotation = arm, values, rotation
        self.points = np.asarray(points, float)
        self.keys = list(range(0, len(points), coarse))
        if self.keys[-1] != len(points) - 1:
            self.keys.append(len(points) - 1)
        self.times = {'ik': 0.0, 'search': 0.0, 'fill': 0.0, 'smooth': 0.0}
        self.layers, self.why = [], ''
        self.solve(1)  # the first point alone: start_bound() may make the rest unneeded
        self.rows, self.cost = None, math.inf

    def solve(self, count=None):
        """Solves the searched points up to count (all by default) not solved yet; self.why
        tells the first one without any solution."""
        count = len(self.keys) if count is None else count
        todo = self.keys[len(self.layers):count]
        if self.why or not todo:
            return
        started = time.time()
        self.layers += self.arm.solve_many(self.values, self.points[todo], self.rotation)
        self.times['ik'] += time.time() - started
        self.banned = [np.zeros(len(q), bool) for q, _ in self.layers]
        for n, (q, _) in enumerate(self.layers):
            if not len(q):
                i = self.keys[n]
                where = i * 100 // max(len(self.points) - 1, 1)
                self.why = f'no IK solution at {where}% of the line'
                break

    def start_bound(self, start):
        """Least possible move from start to the first point (largest joint change)."""
        q = self.layers[0][0]
        alive = q[~self.banned[0]]
        return float(np.abs(alive - start).max(axis=1).min()) if len(alive) else math.inf

    def path(self, start, fixed, jump, bound=math.inf):
        """((n, 7) joints for every point, '') or (None, why). start: (7,) joints now; fixed:
        the path leaves from start (else it starts anywhere and pays the move from start).
        self.cost is the searched cost (a lower bound of this line's motion, up to the grid);
        a cost of bound or more is not filled in (None, 'over bound')."""
        self.solve()
        if self.why:
            return None, self.why
        start = np.asarray(start, float)
        for _ in range(FILL_TRIES):
            started = time.time()
            rows, self.cost, why = shortest_path(self.layers, start, jump, self.banned, fixed)
            self.times['search'] += time.time() - started
            if rows is None:
                return None, why
            if self.cost >= bound:
                return None, 'over bound'
            self.rows = rows
            started = time.time()
            path, branches, bad = self.fill(rows, start, fixed, jump)
            self.times['fill'] += time.time() - started
            if bad is None:
                started = time.time()
                path = self.arm.smooth(self.values, self.points, self.rotation, path, branches,
                                       start, fixed, jump)
                self.times['smooth'] += time.time() - started
                return path, ''
            self.banned[bad][rows[bad]] = True  # its points before cannot be joined to it
        return None, f'no continuous path after {FILL_TRIES} tries'

    def fill(self, rows, start, fixed, jump):
        """(path, branches, None), or (.., .., searched layer that could not be reached). The
        points between two searched ones get the solution that joins its neighbours best: near
        the straight guess first, every solution when that detours (a branch turning point
        lies between); all points the same distance after a searched one at once."""
        count = len(self.points)
        path, branches = np.zeros((count, 7)), np.zeros(count, int)
        keys = np.array(self.keys)
        for n, (i, r) in enumerate(zip(self.keys, rows)):
            path[i], branches[i] = self.layers[n][0][r], self.layers[n][1][r]
        offsets = np.linspace(-self.arm.q7_step, self.arm.q7_step, SMOOTH_SAMPLES)
        owner = np.searchsorted(keys, np.arange(count))  # searched layer that ends each step
        between = ~np.isin(np.arange(count), keys)
        after = np.arange(count) - keys[np.maximum(owner - 1, 0)]
        bad = []
        for t in range(1, int(after[between].max()) + 1 if between.any() else 1):
            idx = np.flatnonzero(between & (after == t))
            if not len(idx):
                continue
            prev, nxt = path[idx - 1], path[keys[owner[idx]]]
            guess = prev + (nxt - prev) / (keys[owner[idx]] - idx + 1)[:, None]
            k, samples = len(idx), len(offsets)
            q, state = self.arm.branches(self.values, np.repeat(self.points[idx], samples, axis=0),
                                         self.rotation, (guess[:, 6][:, None] + offsets).ravel())
            cand = q.reshape(8, k, samples, 7).transpose(1, 0, 2, 3).reshape(k, 8 * samples, 7)
            ok = (state.reshape(8, k, samples).transpose(1, 0, 2).reshape(k, -1) == 2)
            cand_branch = np.repeat(np.arange(8), samples)
            cost = (np.abs(cand - prev[:, None]).max(axis=2)
                    + np.abs(nxt[:, None] - cand).max(axis=2))
            cost = np.where(ok, cost, np.inf)
            best = cost.argmin(axis=1)
            direct = np.abs(nxt - prev).max(axis=1)
            chosen, chosen_branch = cand[np.arange(k), best], cand_branch[best]
            detour = cost[np.arange(k), best] > direct + FILL_SLACK
            if detour.any():
                full = self.arm.solve_many(self.values, self.points[idx[detour]], self.rotation)
                for m, (fq, fb) in zip(np.flatnonzero(detour), full):
                    if len(fq):
                        fc = np.abs(fq - prev[m]).max(axis=1) + np.abs(nxt[m] - fq).max(axis=1)
                        if fc.min() < cost[m, best[m]]:
                            chosen[m], chosen_branch[m] = fq[fc.argmin()], fb[fc.argmin()]
                            cost[m, best[m]] = fc.min()
            missing = ~np.isfinite(cost[np.arange(k), best])
            bad += list(owner[idx[missing | (np.abs(chosen - prev).max(axis=1) > jump)]])
            path[idx], branches[idx] = chosen, chosen_branch
        # the searched points: from the point before (or from start for a fixed first one)
        for n, k1 in enumerate(self.keys):
            if k1 == 0:
                if fixed and np.abs(path[0] - start).max() > jump:
                    bad.append(0)
            elif np.abs(path[k1] - path[k1 - 1]).max() > jump:
                bad.append(n)
        return path, branches, (min(bad) if bad else None)

    def ban(self, point, collides):
        """Leaves out the searched solution nearest point (index into points) in the last path,
        and its neighbours along the branch while collides(q) says they collide too."""
        n = int(np.abs(np.array(self.keys) - point).argmin())
        q, branch = self.layers[n]
        r = self.rows[n]
        self.banned[n][r] = True
        same = np.flatnonzero((branch == branch[r]) & ~self.banned[n])
        for side in (same[q[same, 6] > q[r, 6]], same[q[same, 6] < q[r, 6]]):
            for j in side[np.argsort(np.abs(q[side, 6] - q[r, 6]))]:
                if np.abs(q[j] - q[r]).max() > WRAP_GAP or not collides(q[j]):
                    break
                self.banned[n][j] = True
