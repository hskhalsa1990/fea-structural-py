"""
fea_core.py - compact finite element solvers for structural mechanics.

Contents
--------
* Truss2D            : 2D pin-jointed truss (axial bar elements)
* Truss3D            : 3D pin-jointed space truss
* euler_buckling     : Euler buckling check for compression members
* Frame2D            : 2D rigid-jointed frame (beam-column), static + modal
* Beam               : Euler-Bernoulli beam (Hermite cubic elements), static + modal
* PlaneStressQ4      : 4-node bilinear quadrilateral for plane stress
* rect_mesh, quarter_hole_mesh : structured mesh generators
* MATERIALS          : typical handbook material properties (SI units)

Everything is SI: metres, newtons, pascals, kilograms.
Only NumPy and SciPy are required.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as sla

# Typical handbook values (room temperature). Always confirm against your datasheet.
MATERIALS = {
    "Steel (structural)": dict(E=210e9, nu=0.30, rho=7850.0, sy=250e6),
    "Aluminium 6061-T6": dict(E=68.9e9, nu=0.33, rho=2700.0, sy=276e6),
    "Titanium Ti-6Al-4V": dict(E=113.8e9, nu=0.342, rho=4430.0, sy=880e6),
}


# ----------------------------------------------------------------------------
# Common helper
# ----------------------------------------------------------------------------
def solve_constrained(K, F, fixed):
    """Solve K u = F with the given fixed dofs set to zero.

    Returns (u, R) where R = K u - F (non-zero only at fixed dofs = reactions).
    Works with dense arrays and SciPy sparse matrices.
    """
    n = K.shape[0]
    fixed = np.unique(np.asarray(fixed, dtype=int))
    free = np.setdiff1d(np.arange(n), fixed)
    u = np.zeros(n)
    if sp.issparse(K):
        Kc = K.tocsr()
        u[free] = spla.spsolve(Kc[free][:, free].tocsc(), F[free])
        R = Kc @ u - F
    else:
        u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        R = K @ u - F
    return u, R


# ----------------------------------------------------------------------------
# 2D truss
# ----------------------------------------------------------------------------
class Truss2D:
    """2D pin-jointed truss. Axial force N is positive in tension."""

    def __init__(self, nodes, members, E, A):
        self.nodes = np.asarray(nodes, float)
        self.members = np.asarray(members, int)
        m = len(self.members)
        self.E = np.broadcast_to(np.asarray(E, float), (m,)).copy()
        self.A = np.broadcast_to(np.asarray(A, float), (m,)).copy()

    def geometry(self, k):
        i, j = self.members[k]
        d = self.nodes[j] - self.nodes[i]
        L = float(np.hypot(*d))
        return i, j, L, d[0] / L, d[1] / L

    def solve(self, loads, fixed):
        """loads: {node: (Fx, Fy)};  fixed: [(node, 0|1), ...] (0 = x, 1 = y)."""
        n = len(self.nodes)
        K = np.zeros((2 * n, 2 * n))
        F = np.zeros(2 * n)
        for k in range(len(self.members)):
            i, j, L, c, s = self.geometry(k)
            kk = self.E[k] * self.A[k] / L * np.array([
                [c * c, c * s, -c * c, -c * s],
                [c * s, s * s, -c * s, -s * s],
                [-c * c, -c * s, c * c, c * s],
                [-c * s, -s * s, c * s, s * s]])
            dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]
            K[np.ix_(dofs, dofs)] += kk
        for node, (fx, fy) in loads.items():
            F[2 * node] += fx
            F[2 * node + 1] += fy
        fixed_dofs = [2 * nd + d for nd, d in fixed]
        u, R = solve_constrained(K, F, fixed_dofs)
        N = np.zeros(len(self.members))
        for k in range(len(self.members)):
            i, j, L, c, s = self.geometry(k)
            du = u[[2 * j, 2 * j + 1]] - u[[2 * i, 2 * i + 1]]
            N[k] = self.E[k] * self.A[k] / L * (c * du[0] + s * du[1])
        reactions = {(nd, d): R[2 * nd + d] for nd, d in fixed}
        return dict(u=u.reshape(-1, 2), N=N, stress=N / self.A, reactions=reactions)


# ----------------------------------------------------------------------------
# Euler-Bernoulli beam
# ----------------------------------------------------------------------------
class Beam:
    """Euler-Bernoulli beam, 2 dofs per node: deflection v (+y up) and rotation theta.

    Dof numbering: v of node k is 2k, theta of node k is 2k+1.
    Sign convention: loads positive upward (+y). Use negative values for downward loads.
    Bending moment is positive when sagging; shear V = dM/dx.
    """

    def __init__(self, x, E, I):
        self.x = np.asarray(x, float)
        self.n = len(self.x)
        self.ne = self.n - 1
        self.L = np.diff(self.x)
        self.E = np.broadcast_to(np.asarray(E, float), (self.ne,)).copy()
        self.I = np.broadcast_to(np.asarray(I, float), (self.ne,)).copy()

    @staticmethod
    def ke(EI, L):
        return EI / L**3 * np.array([
            [12, 6 * L, -12, 6 * L],
            [6 * L, 4 * L**2, -6 * L, 2 * L**2],
            [-12, -6 * L, 12, -6 * L],
            [6 * L, 2 * L**2, -6 * L, 4 * L**2]])

    @staticmethod
    def me(rho_A, L):
        return rho_A * L / 420.0 * np.array([
            [156, 22 * L, 54, -13 * L],
            [22 * L, 4 * L**2, 13 * L, -3 * L**2],
            [54, 13 * L, 156, -22 * L],
            [-13 * L, -3 * L**2, -22 * L, 4 * L**2]])

    def _dofs(self, e):
        return np.arange(2 * e, 2 * e + 4)

    def stiffness(self):
        K = np.zeros((2 * self.n, 2 * self.n))
        for e in range(self.ne):
            d = self._dofs(e)
            K[np.ix_(d, d)] += self.ke(self.E[e] * self.I[e], self.L[e])
        return K

    def solve(self, fixed, point_loads=None, moments=None, udl=0.0):
        """fixed: list of dof indices. point_loads/moments: {node: value}. udl: N/m (+y), scalar or per element."""
        q = np.broadcast_to(np.asarray(udl, float), (self.ne,)).copy()
        K = self.stiffness()
        F = np.zeros(2 * self.n)
        feq = np.zeros((self.ne, 4))
        for e in range(self.ne):
            L = self.L[e]
            feq[e] = q[e] * L / 2.0 * np.array([1, L / 6.0, 1, -L / 6.0])
            F[self._dofs(e)] += feq[e]
        for node, P in (point_loads or {}).items():
            F[2 * node] += P
        for node, M in (moments or {}).items():
            F[2 * node + 1] += M
        u, R = solve_constrained(K, F, fixed)
        self.u, self.q, self.fixed = u, q, np.asarray(fixed, int)
        self.R = R
        self.f_end = np.array([
            self.ke(self.E[e] * self.I[e], self.L[e]) @ u[self._dofs(e)] - feq[e]
            for e in range(self.ne)])
        return u

    def reactions(self):
        return {int(d): float(self.R[d]) for d in self.fixed}

    def diagram(self, npts=40):
        """Return x, deflection v, shear V, bending moment M sampled along the beam."""
        xs, vs, Vs, Ms = [], [], [], []
        for e in range(self.ne):
            L = self.L[e]
            x = np.linspace(0.0, L, npts)
            xi = x / L
            N1 = 1 - 3 * xi**2 + 2 * xi**3
            N2 = L * (xi - 2 * xi**2 + xi**3)
            N3 = 3 * xi**2 - 2 * xi**3
            N4 = L * (-xi**2 + xi**3)
            ue = self.u[self._dofs(e)]
            q = self.q[e]
            EI = self.E[e] * self.I[e]
            v = N1 * ue[0] + N2 * ue[1] + N3 * ue[2] + N4 * ue[3]
            v = v + q * x**2 * (L - x) ** 2 / (24.0 * EI)  # exact UDL correction
            f1, f2 = self.f_end[e, 0], self.f_end[e, 1]
            Vs.append(f1 + q * x)
            Ms.append(x * f1 - f2 + q * x**2 / 2.0)
            xs.append(self.x[e] + x)
            vs.append(v)
        return np.concatenate(xs), np.concatenate(vs), np.concatenate(Vs), np.concatenate(Ms)

    def interp(self, dofvec, npts=30):
        """Hermite interpolation of any dof vector (e.g. a mode shape)."""
        xs, vs = [], []
        for e in range(self.ne):
            L = self.L[e]
            x = np.linspace(0.0, L, npts)
            xi = x / L
            N1 = 1 - 3 * xi**2 + 2 * xi**3
            N2 = L * (xi - 2 * xi**2 + xi**3)
            N3 = 3 * xi**2 - 2 * xi**3
            N4 = L * (-xi**2 + xi**3)
            ue = dofvec[self._dofs(e)]
            xs.append(self.x[e] + x)
            vs.append(N1 * ue[0] + N2 * ue[1] + N3 * ue[2] + N4 * ue[3])
        return np.concatenate(xs), np.concatenate(vs)

    def modal(self, rho, A, fixed, nmodes=4):
        """Natural frequencies [Hz] and mode shapes (consistent mass matrix)."""
        K = self.stiffness()
        M = np.zeros_like(K)
        for e in range(self.ne):
            d = self._dofs(e)
            M[np.ix_(d, d)] += self.me(rho * A, self.L[e])
        free = np.setdiff1d(np.arange(2 * self.n), np.asarray(fixed, int))
        w2, vec = sla.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
        w = np.sqrt(np.abs(w2[:nmodes]))
        shapes = np.zeros((2 * self.n, nmodes))
        shapes[free, :] = vec[:, :nmodes]
        for k in range(nmodes):
            peak = np.max(np.abs(shapes[0::2, k]))
            shapes[:, k] /= peak if peak > 0 else 1.0
        return w / (2 * np.pi), shapes


# ----------------------------------------------------------------------------
# 4-node plane stress element
# ----------------------------------------------------------------------------
_GP = 1.0 / np.sqrt(3.0)
_GAUSS = [(-_GP, -_GP), (_GP, -_GP), (_GP, _GP), (-_GP, _GP)]
_CORNERS = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]


def q4_B(xy, xi, eta):
    """Strain-displacement matrix (3x8) and Jacobian determinant at (xi, eta)."""
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
    J = np.array([dN_dxi @ xy, dN_deta @ xy])
    detJ = np.linalg.det(J)
    dN = np.linalg.solve(J, np.array([dN_dxi, dN_deta]))
    B = np.zeros((3, 8))
    B[0, 0::2] = dN[0]
    B[1, 1::2] = dN[1]
    B[2, 0::2] = dN[1]
    B[2, 1::2] = dN[0]
    return B, detJ


class PlaneStressQ4:
    """Plane stress / plane strain with bilinear quadrilaterals (full 2x2 Gauss integration).

    incompatible=True adds Wilson-Taylor incompatible modes (Q6 / QM6 style), which removes
    shear locking in bending-dominated problems at almost no extra cost.
    """

    def __init__(self, nodes, elems, E, nu, t, plane="stress", incompatible=False):
        self.nodes = np.asarray(nodes, float)
        self.elems = np.asarray(elems, int)
        self.E, self.nu, self.t = E, nu, t
        self.plane = plane
        self.incompatible = incompatible      # Wilson-Taylor incompatible modes (cures shear locking)
        self._alpha_map = {}
        if plane == "stress":
            self.D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        elif plane == "strain":   # t is the unit thickness (use t=1 for per-metre results)
            c = E / ((1 + nu) * (1 - 2 * nu))
            self.D = c * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])
        else:
            raise ValueError("plane must be 'stress' or 'strain'")
        self.ndof = 2 * len(self.nodes)

    def _edofs(self, conn):
        return np.array([[2 * n, 2 * n + 1] for n in conn]).ravel()

    @staticmethod
    def _binc(xy, xi, eta, detJ0, J0inv, detJ):
        """Strain matrix (3x4) of the incompatible modes N5 = 1 - xi^2, N6 = 1 - eta^2."""
        g5 = J0inv @ np.array([-2 * xi, 0.0])
        g6 = J0inv @ np.array([0.0, -2 * eta])
        Bi = np.zeros((3, 4))
        Bi[0, 0], Bi[0, 2] = g5[0], g6[0]
        Bi[1, 1], Bi[1, 3] = g5[1], g6[1]
        Bi[2, 0], Bi[2, 1], Bi[2, 2], Bi[2, 3] = g5[1], g5[0], g6[1], g6[0]
        return Bi * (detJ0 / detJ)

    def _element_k(self, e):
        xy = self.nodes[self.elems[e]]
        ke = np.zeros((8, 8))
        if not self.incompatible:
            for xi, eta in _GAUSS:
                B, detJ = q4_B(xy, xi, eta)
                ke += B.T @ self.D @ B * detJ * self.t
            return ke
        _, detJ0 = q4_B(xy, 0.0, 0.0)
        dN0 = 0.25 * np.array([[-1, 1, 1, -1], [-1, -1, 1, 1]], float)
        J0inv = np.linalg.inv(dN0 @ xy)
        kua, kaa = np.zeros((8, 4)), np.zeros((4, 4))
        for xi, eta in _GAUSS:
            B, detJ = q4_B(xy, xi, eta)
            Bi = self._binc(xy, xi, eta, detJ0, J0inv, detJ)
            w = detJ * self.t
            ke += B.T @ self.D @ B * w
            kua += B.T @ self.D @ Bi * w
            kaa += Bi.T @ self.D @ Bi * w
        G = np.linalg.solve(kaa, kua.T)                  # alpha = -G u
        self._alpha_map[e] = (-G, J0inv, detJ0)
        return ke - kua @ G

    def stiffness(self):
        rows, cols, vals = [], [], []
        for e, conn in enumerate(self.elems):
            ke = self._element_k(e)
            d = self._edofs(conn)
            rows.extend(np.repeat(d, 8))
            cols.extend(np.tile(d, 8))
            vals.extend(ke.ravel())
        return sp.coo_matrix((vals, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()

    def solve(self, F, fixed_dofs):
        K = self.stiffness()
        u, R = solve_constrained(K, F, fixed_dofs)
        self.u, self.R = u, R
        return u

    def nodal_stress(self, u=None):
        """Stresses [sx, sy, txy] evaluated at element corners and averaged at nodes."""
        u = self.u if u is None else u
        acc = np.zeros((len(self.nodes), 3))
        cnt = np.zeros(len(self.nodes))
        for e, conn in enumerate(self.elems):
            xy = self.nodes[conn]
            ue = u[self._edofs(conn)]
            if self.incompatible:
                Ga, J0inv, detJ0 = self._alpha_map[e]
                alpha = Ga @ ue
            for k, (xi, eta) in enumerate(_CORNERS):
                B, detJ = q4_B(xy, xi, eta)
                eps = B @ ue
                if self.incompatible:
                    eps = eps + self._binc(xy, xi, eta, detJ0, J0inv, detJ) @ alpha
                acc[conn[k]] += self.D @ eps
                cnt[conn[k]] += 1
        return acc / cnt[:, None]

    @staticmethod
    def von_mises(s, nu=None):
        """Von Mises stress from [sx, sy, txy]. Pass nu for plane strain (adds sz = nu (sx + sy))."""
        sx, sy, txy = s[:, 0], s[:, 1], s[:, 2]
        if nu is None:
            return np.sqrt(sx**2 - sx * sy + sy**2 + 3 * txy**2)
        sz = nu * (sx + sy)
        return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * txy**2)

    @staticmethod
    def principal(s):
        """In-plane principal stresses (s1 >= s2) and the angle of s1 in radians."""
        sx, sy, txy = s[:, 0], s[:, 1], s[:, 2]
        avg, r = 0.5 * (sx + sy), np.hypot(0.5 * (sx - sy), txy)
        return avg + r, avg - r, 0.5 * np.arctan2(2 * txy, sx - sy)

    @staticmethod
    def safety_factor(vm, yield_strength):
        """Factor of safety against yield (von Mises criterion), per node."""
        return yield_strength / np.maximum(vm, 1e-30)


# ----------------------------------------------------------------------------
# Mesh generators and load helpers
# ----------------------------------------------------------------------------
def rect_mesh(Lx, Ly, nx, ny):
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    nodes = np.array([[x, y] for y in ys for x in xs])
    elems = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            elems.append([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1])
    return nodes, np.array(elems)


def quarter_hole_mesh(a, L, nr, nt, grade=1.3):
    """Quarter of a square plate (side L) with a circular hole of radius a at the origin."""
    nodes = np.zeros(((nr + 1) * (nt + 1), 2))
    for j in range(nt + 1):
        th = (np.pi / 2) * j / nt
        inner = np.array([a * np.cos(th), a * np.sin(th)])
        if j == nt:
            outer = np.array([0.0, L])
        elif th <= np.pi / 4 + 1e-12:
            outer = np.array([L, L * np.tan(th)])
        else:
            outer = np.array([L / np.tan(th), L])
        for i in range(nr + 1):
            s = (i / nr) ** grade
            nodes[j * (nr + 1) + i] = (1 - s) * inner + s * outer
    elems = []
    for j in range(nt):
        for i in range(nr):
            n0 = j * (nr + 1) + i
            elems.append([n0, n0 + 1, n0 + nr + 2, n0 + nr + 1])
    return nodes, np.array(elems)


def add_edge_load(F, nodes, ids, comp, traction, t):
    """Add a uniform traction (force per area) along a straight edge made of node ids.

    comp = 0 -> force in x, comp = 1 -> force in y.
    """
    ids = np.asarray(ids, int)
    pts = nodes[ids]
    axis = 0 if np.ptp(pts[:, 0]) > np.ptp(pts[:, 1]) else 1
    order = np.argsort(pts[:, axis])
    ids = ids[order]
    pts = pts[order]
    for k in range(len(ids) - 1):
        length = np.hypot(*(pts[k + 1] - pts[k]))
        f = traction * t * length / 2.0
        F[2 * ids[k] + comp] += f
        F[2 * ids[k + 1] + comp] += f
    return F


# ----------------------------------------------------------------------------
# 2D frame (beam-column)
# ----------------------------------------------------------------------------
class Frame2D:
    """2D rigid-jointed frame: Euler-Bernoulli beam-column, 3 dofs per node (ux, uy, rz).

    members : list of (node_i, node_j);  E, A, I may be scalars or per-member arrays.
    Axial force N is positive in tension; end moments follow the local-axis convention.
    """

    def __init__(self, nodes, members, E, A, I):
        self.nodes = np.asarray(nodes, float)
        self.members = np.asarray(members, int)
        m = len(self.members)
        self.E, self.A, self.I = (np.broadcast_to(np.asarray(v, float), (m,)).copy() for v in (E, A, I))

    def geometry(self, k):
        i, j = self.members[k]
        d = self.nodes[j] - self.nodes[i]
        L = float(np.hypot(*d))
        return i, j, L, d[0] / L, d[1] / L

    def local_k(self, k):
        _, _, L, _, _ = self.geometry(k)
        EA, EI = self.E[k] * self.A[k], self.E[k] * self.I[k]
        a, b = EA / L, EI / L**3
        return np.array([
            [a, 0, 0, -a, 0, 0],
            [0, 12 * b, 6 * L * b, 0, -12 * b, 6 * L * b],
            [0, 6 * L * b, 4 * L**2 * b, 0, -6 * L * b, 2 * L**2 * b],
            [-a, 0, 0, a, 0, 0],
            [0, -12 * b, -6 * L * b, 0, 12 * b, -6 * L * b],
            [0, 6 * L * b, 2 * L**2 * b, 0, -6 * L * b, 4 * L**2 * b]])

    def rotation(self, k):
        _, _, _, c, s = self.geometry(k)
        T = np.zeros((6, 6))
        R = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        T[:3, :3] = R
        T[3:, 3:] = R
        return T

    def solve(self, fixed, nodal_loads=None, member_udl=None):
        """fixed: [(node, dof)] with dof 0=ux, 1=uy, 2=rz.
        nodal_loads: {node: (Fx, Fy, Mz)}.
        member_udl: {member: w} uniform load per length in the member's local +y direction.
        """
        n = len(self.nodes)
        K = np.zeros((3 * n, 3 * n))
        F = np.zeros(3 * n)
        fe_local = {}
        for k in range(len(self.members)):
            i, j, L, _, _ = self.geometry(k)
            T = self.rotation(k)
            d = np.r_[3 * i:3 * i + 3, 3 * j:3 * j + 3]
            K[np.ix_(d, d)] += T.T @ self.local_k(k) @ T
            w = (member_udl or {}).get(k, 0.0)
            f0 = np.array([0, w * L / 2, w * L**2 / 12, 0, w * L / 2, -w * L**2 / 12])
            fe_local[k] = f0
            F[d] += T.T @ f0
        for node, (fx, fy, mz) in (nodal_loads or {}).items():
            F[3 * node:3 * node + 3] += [fx, fy, mz]
        fixed_dofs = [3 * nd + d for nd, d in fixed]
        u, R = solve_constrained(K, F, fixed_dofs)
        ends = np.zeros((len(self.members), 6))
        for k in range(len(self.members)):
            i, j, *_ = self.geometry(k)
            d = np.r_[3 * i:3 * i + 3, 3 * j:3 * j + 3]
            ends[k] = self.local_k(k) @ (self.rotation(k) @ u[d]) - fe_local[k]
        self.u, self.end_forces = u, ends
        reactions = {(nd, d): R[3 * nd + d] for nd, d in fixed}
        return dict(u=u.reshape(-1, 3), end_forces=ends, reactions=reactions)

    def local_m(self, k, rho):
        """Consistent mass matrix of a beam-column element (local axes)."""
        _, _, L, _, _ = self.geometry(k)
        m = rho * self.A[k] * L
        a = m / 6.0
        b = m / 420.0
        return np.array([
            [2 * a, 0, 0, a, 0, 0],
            [0, 156 * b, 22 * L * b, 0, 54 * b, -13 * L * b],
            [0, 22 * L * b, 4 * L**2 * b, 0, 13 * L * b, -3 * L**2 * b],
            [a, 0, 0, 2 * a, 0, 0],
            [0, 54 * b, 13 * L * b, 0, 156 * b, -22 * L * b],
            [0, -13 * L * b, -3 * L**2 * b, 0, -22 * L * b, 4 * L**2 * b]])

    def modal(self, rho, fixed, n_modes=4):
        """Natural frequencies (Hz) and mode shapes of the frame.

        rho : density (scalar or per-member). fixed : [(node, dof)] as in solve().
        Returns (freqs, modes) where modes has shape (3 * n_nodes, n_modes), mass-normalised.
        """
        n = len(self.nodes)
        rho = np.broadcast_to(np.asarray(rho, float), (len(self.members),))
        K = np.zeros((3 * n, 3 * n))
        M = np.zeros((3 * n, 3 * n))
        for k in range(len(self.members)):
            i, j, *_ = self.geometry(k)
            T = self.rotation(k)
            d = np.r_[3 * i:3 * i + 3, 3 * j:3 * j + 3]
            K[np.ix_(d, d)] += T.T @ self.local_k(k) @ T
            M[np.ix_(d, d)] += T.T @ self.local_m(k, rho[k]) @ T
        fixed_dofs = np.unique([3 * nd + d for nd, d in fixed])
        free = np.setdiff1d(np.arange(3 * n), fixed_dofs)
        w2, vec = sla.eigh(K[np.ix_(free, free)], M[np.ix_(free, free)])
        n_modes = min(n_modes, len(w2))
        modes = np.zeros((3 * n, n_modes))
        modes[free] = vec[:, :n_modes]
        return np.sqrt(np.abs(w2[:n_modes])) / (2 * np.pi), modes


# ----------------------------------------------------------------------------
# Member buckling check
# ----------------------------------------------------------------------------
def euler_buckling(N, L, E, I, K=1.0):
    """Euler buckling check for axial members.

    N : axial forces (tension positive), L : lengths, E, I : modulus and second moment of area,
    K : effective length factor (1.0 pinned-pinned, 0.7 fixed-pinned, 0.5 fixed-fixed, 2.0 cantilever).
    Returns (Pcr, utilisation, factor). utilisation = |N| / Pcr for compression and 0 for tension;
    factor = Pcr / |N| for compression and inf for tension.
    """
    N, L = np.asarray(N, float), np.asarray(L, float)
    Pcr = np.pi**2 * E * I / (K * L) ** 2
    comp = N < 0
    util = np.where(comp, np.abs(N) / Pcr, 0.0)
    with np.errstate(divide="ignore"):
        factor = np.where(comp, Pcr / np.maximum(np.abs(N), 1e-30), np.inf)
    return Pcr, util, factor


# ----------------------------------------------------------------------------
# 3D truss
# ----------------------------------------------------------------------------
class Truss3D:
    """3D pin-jointed truss (axial bars, 3 dofs per node). Axial force N is positive in tension."""

    def __init__(self, nodes, members, E, A):
        self.nodes = np.asarray(nodes, float)
        if self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (n, 3)")
        self.members = np.asarray(members, int)
        m = len(self.members)
        self.E = np.broadcast_to(np.asarray(E, float), (m,)).copy()
        self.A = np.broadcast_to(np.asarray(A, float), (m,)).copy()

    def geometry(self, k):
        i, j = self.members[k]
        d = self.nodes[j] - self.nodes[i]
        L = float(np.linalg.norm(d))
        return i, j, L, d / L

    def solve(self, loads, fixed):
        """loads: {node: (Fx, Fy, Fz)};  fixed: [(node, 0|1|2), ...] (0 = x, 1 = y, 2 = z)."""
        n = len(self.nodes)
        K = np.zeros((3 * n, 3 * n))
        F = np.zeros(3 * n)
        for k in range(len(self.members)):
            i, j, L, c = self.geometry(k)
            kk = self.E[k] * self.A[k] / L * np.outer(c, c)
            d = np.r_[3 * i:3 * i + 3, 3 * j:3 * j + 3]
            K[np.ix_(d[:3], d[:3])] += kk
            K[np.ix_(d[3:], d[3:])] += kk
            K[np.ix_(d[:3], d[3:])] -= kk
            K[np.ix_(d[3:], d[:3])] -= kk
        for node, f in loads.items():
            F[3 * node:3 * node + 3] += f
        fixed_dofs = [3 * nd + d for nd, d in fixed]
        u, R = solve_constrained(K, F, fixed_dofs)
        N = np.zeros(len(self.members))
        for k in range(len(self.members)):
            i, j, L, c = self.geometry(k)
            N[k] = self.E[k] * self.A[k] / L * c @ (u[3 * j:3 * j + 3] - u[3 * i:3 * i + 3])
        reactions = {(nd, d): R[3 * nd + d] for nd, d in fixed}
        return dict(u=u.reshape(-1, 3), N=N, stress=N / self.A, reactions=reactions)
