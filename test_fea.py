"""Validation tests for fea_core.py. Run with `python test_fea.py` or `pytest`."""
import numpy as np
from fea_core import (Truss2D, Beam, PlaneStressQ4, Frame2D, Truss3D, euler_buckling, rect_mesh, quarter_hole_mesh, add_edge_load)

E, I, L = 210e9, 8e-6, 2.0


def rel(a, b):
    return abs(a - b) / abs(b)


def test_cantilever_tip_load():
    b = Beam(np.linspace(0, L, 5), E, I)
    b.solve([0, 1], point_loads={4: -1000.0})
    assert rel(abs(b.u[8]), 1000 * L**3 / (3 * E * I)) < 1e-9


def test_cantilever_udl():
    b = Beam(np.linspace(0, L, 5), E, I)
    b.solve([0, 1], udl=-1000.0)
    assert rel(abs(b.u[8]), 1000 * L**4 / (8 * E * I)) < 1e-9


def test_simply_supported_cases():
    b = Beam(np.linspace(0, L, 5), E, I)
    b.solve([0, 8], point_loads={2: -1000.0})
    assert rel(abs(b.u[4]), 1000 * L**3 / (48 * E * I)) < 1e-9
    b.solve([0, 8], udl=-1000.0)
    assert rel(abs(b.u[4]), 5 * 1000 * L**4 / (384 * E * I)) < 1e-9
    x, v, V, M = b.diagram()
    assert rel(M.max(), 1000 * L**2 / 8) < 1e-9


def test_beam_reactions_balance_load():
    b = Beam(np.linspace(0, L, 5), E, I)
    b.solve([0, 8], udl=-1000.0)
    R = b.reactions()
    assert rel(R[0] + R[8], 1000 * L) < 1e-9


def test_fixed_fixed_udl():
    b = Beam(np.linspace(0, L, 5), E, I)
    b.solve([0, 1, 8, 9], udl=-1000.0)
    assert rel(abs(b.u[4]), 1000 * L**4 / (384 * E * I)) < 1e-9


def test_truss_method_of_joints():
    P = 10e3
    t = Truss2D([(0, 0), (4, 0), (2, 3)], [(0, 1), (0, 2), (1, 2)], E, 1e-3)
    r = t.solve({2: (0, -P)}, [(0, 0), (0, 1), (1, 1)])
    assert rel(r["N"][0], P / 3) < 1e-9
    assert rel(r["N"][1], -P * np.sqrt(13) / 6) < 1e-9
    assert rel(r["N"][2], -P * np.sqrt(13) / 6) < 1e-9


def test_q4_patch_test():
    nodes, elems = rect_mesh(2.0, 1.0, 4, 2)
    m = PlaneStressQ4(nodes, elems, E, 0.3, 0.01)
    F = np.zeros(m.ndof)
    right = np.where(np.isclose(nodes[:, 0], 2.0))[0]
    left = np.where(np.isclose(nodes[:, 0], 0.0))[0]
    bot = np.where(np.isclose(nodes[:, 1], 0.0))[0]
    add_edge_load(F, nodes, right, 0, 100e6, 0.01)
    u = m.solve(F, [2 * i for i in left] + [2 * i + 1 for i in bot])
    s = m.nodal_stress()
    assert np.allclose(s[:, 0], 100e6, rtol=1e-9)
    assert rel(u[2 * right[0]], 100e6 * 2.0 / E) < 1e-9


def test_hole_stress_concentration_close_to_kirsch():
    a, Lq, s0, nr, nt = 0.05, 0.5, 50e6, 16, 32
    nodes, elems = quarter_hole_mesh(a, Lq, nr, nt, 2.0)
    m = PlaneStressQ4(nodes, elems, E, 0.3, 0.01)
    F = np.zeros(m.ndof)
    right = np.where(np.isclose(nodes[:, 0], Lq))[0]
    add_edge_load(F, nodes, right, 0, s0, 0.01)
    sy = np.where(np.isclose(nodes[:, 1], 0))[0]
    sx = np.where(np.isclose(nodes[:, 0], 0))[0]
    m.solve(F, [2 * i + 1 for i in sy] + [2 * i for i in sx])
    Kt = m.nodal_stress()[nt * (nr + 1), 0] / s0
    assert 2.9 < Kt < 3.2


def test_cantilever_frequencies():
    side = 0.04
    A, Ib, rho = side**2, side**4 / 12, 7850.0
    b = Beam(np.linspace(0, 1.0, 21), E, Ib)
    f, _ = b.modal(rho, A, [0, 1], 3)
    bl = np.array([1.8751, 4.6941, 7.8548])
    fa = bl**2 / (2 * np.pi) * np.sqrt(E * Ib / (rho * A))
    assert np.all(np.abs(f / fa - 1) < 1e-3)


def test_frame_cantilever_matches_beam():
    A = 0.01
    fr = Frame2D([[0, 0], [L, 0]], [[0, 1]], E, A, I)
    r = fr.solve(fixed=[(0, 0), (0, 1), (0, 2)], nodal_loads={1: (0, -1000.0, 0)})
    assert rel(abs(r["u"][1, 1]), 1000 * L**3 / (3 * E * I)) < 1e-9


def test_frame_rotated_member_and_axial():
    # vertical column in tension: u = P L / (E A), independent of the member rotation
    A = 0.01
    fr = Frame2D([[0, 0], [0, L]], [[0, 1]], E, A, I)
    r = fr.solve(fixed=[(0, 0), (0, 1), (0, 2)], nodal_loads={1: (0, 5000.0, 0)})
    assert rel(r["u"][1, 1], 5000 * L / (E * A)) < 1e-9
    assert rel(r["end_forces"][0, 3], 5000.0) < 1e-9 or rel(abs(r["end_forces"][0, 0]), 5000.0) < 1e-9


def test_frame_fixed_fixed_udl_and_equilibrium():
    A = 0.01
    nodes = [[0, 0], [L / 2, 0], [L, 0]]
    fr = Frame2D(nodes, [[0, 1], [1, 2]], E, A, I)
    w = -2000.0
    r = fr.solve(fixed=[(0, 0), (0, 1), (0, 2), (2, 0), (2, 1), (2, 2)], member_udl={0: w, 1: w})
    assert rel(abs(r["u"][1, 1]), abs(w) * L**4 / (384 * E * I)) < 1e-9
    assert rel(abs(r["reactions"][(0, 2)]), abs(w) * L**2 / 12) < 1e-9
    assert rel(r["reactions"][(0, 1)] + r["reactions"][(2, 1)], abs(w) * L) < 1e-9


def test_plane_strain_patch_and_principal():
    nodes, elems = rect_mesh(1.0, 1.0, 3, 3)
    m = PlaneStressQ4(nodes, elems, 200e9, 0.3, 1.0, plane="strain")
    F = np.zeros(m.ndof)
    right = np.where(np.isclose(nodes[:, 0], 1.0))[0]
    add_edge_load(F, nodes, right, 0, 100e6, 1.0)
    fixed = [2 * i for i in np.where(np.isclose(nodes[:, 0], 0))[0]] + [1]
    m.solve(F, fixed)
    s = m.nodal_stress()
    assert np.allclose(s[:, 0], 100e6, rtol=1e-6)
    s1, s2, _ = m.principal(s)
    assert np.allclose(s1, 100e6, rtol=1e-6) and np.allclose(s2, 0, atol=1.0)
    vm = m.von_mises(s, nu=0.3)
    assert np.allclose(vm, 100e6 * np.sqrt(1 - 0.3 + 0.09), rtol=1e-3)
    sf = m.safety_factor(vm, 250e6)
    assert np.all(sf > 1.0)


def test_frame_modal_cantilever_bending_and_axial():
    side = 0.04
    A, Ib, rho, Lc = side**2, side**4 / 12, 7850.0, 1.0
    n = 20
    pts = [[x, 0.0] for x in np.linspace(0, Lc, n + 1)]
    fr = Frame2D(pts, [[i, i + 1] for i in range(n)], E, A, Ib)
    f, _ = fr.modal(rho, [(0, 0), (0, 1), (0, 2)], 6)
    bl = np.array([1.8751, 4.6941])
    fa = bl**2 / (2 * np.pi) * np.sqrt(E * Ib / (rho * A * Lc**4))
    assert np.all(np.abs(f[:2] / fa - 1) < 1e-3)
    f_ax = np.sqrt(E / rho) / (4 * Lc)          # first axial mode of a fixed-free bar
    assert np.min(np.abs(f / f_ax - 1)) < 1e-3


def test_frame_modal_rotation_invariant():
    A, rho = 0.01, 7850.0
    def first(angle):
        c, s = np.cos(angle), np.sin(angle)
        pts = [[x * c, x * s] for x in np.linspace(0, 2.0, 9)]
        fr = Frame2D(pts, [[i, i + 1] for i in range(8)], E, A, I)
        return fr.modal(rho, [(0, 0), (0, 1), (0, 2)], 3)[0]
    assert np.allclose(first(0.0), first(0.7), rtol=1e-8)


def test_truss3d_tripod_and_2d_consistency():
    # three legs meeting at an apex, vertical load P: N = -P / (3 sin(theta))
    h, r, P, A = 2.0, 1.5, 9000.0, 1e-3
    base = [[r * np.cos(a), r * np.sin(a), 0.0] for a in (0, 2 * np.pi / 3, 4 * np.pi / 3)]
    t3 = Truss3D(base + [[0, 0, h]], [[0, 3], [1, 3], [2, 3]], E, A)
    fixed = [(i, d) for i in range(3) for d in range(3)]
    r3 = t3.solve({3: (0, 0, -P)}, fixed)
    sin_t = h / np.hypot(h, r)
    assert np.allclose(r3["N"], -P / (3 * sin_t), rtol=1e-9)
    assert rel(sum(v for (n_, d), v in r3["reactions"].items() if d == 2), P) < 1e-9
    # planar truss in the x-z plane must match Truss2D
    nodes2 = [(0, 0), (4, 0), (2, 3)]
    mem = [(0, 1), (0, 2), (1, 2)]
    a = Truss2D(nodes2, mem, E, 1e-3).solve({2: (10e3, -20e3)}, [(0, 0), (0, 1), (1, 1)])
    b = Truss3D([(x, 0, z) for x, z in nodes2], mem, E, 1e-3).solve(
        {2: (10e3, 0, -20e3)}, [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1)])   # (2, 1): block out-of-plane motion
    assert np.allclose(a["N"], b["N"], rtol=1e-9)


def _cantilever_plate(nx, ny, incompatible):
    Ec, nuc, tc, Lx, Ly, Pc = 210e9, 0.30, 0.1, 1.0, 0.2, 1000.0
    nd, el = rect_mesh(Lx, Ly, nx, ny)
    m = PlaneStressQ4(nd, el, Ec, nuc, tc, incompatible=incompatible)
    F = np.zeros(m.ndof)
    rt = np.where(np.isclose(nd[:, 0], Lx))[0]
    lf = np.where(np.isclose(nd[:, 0], 0))[0]
    add_edge_load(F, nd, rt, 1, -Pc / (tc * Ly), tc)
    u = m.solve(F, [2 * i for i in lf] + [2 * i + 1 for i in lf])
    Ic = tc * Ly**3 / 12
    d_eb = Pc * Lx**3 / (3 * Ec * Ic)
    return -np.mean(u[2 * rt + 1]) / d_eb, m


def test_incompatible_modes_remove_locking():
    r_std, _ = _cantilever_plate(5, 1, False)
    r_inc, m = _cantilever_plate(5, 1, True)
    assert r_std < 0.75 and r_inc > 0.95


def test_incompatible_modes_patch_test():
    nodes, elems = rect_mesh(1.0, 1.0, 3, 3)
    nodes = nodes + 0.03 * np.sin(7 * nodes[:, ::-1] + 1) * (~np.isclose(nodes, 0) & ~np.isclose(nodes, 1))
    m = PlaneStressQ4(nodes, elems, 200e9, 0.3, 1.0, incompatible=True)
    F = np.zeros(m.ndof)
    right = np.where(np.isclose(nodes[:, 0], 1.0))[0]
    add_edge_load(F, nodes, right, 0, 100e6, 1.0)
    fixed = [2 * i for i in np.where(np.isclose(nodes[:, 0], 0))[0]] + [1]
    m.solve(F, fixed)
    s = m.nodal_stress()
    assert np.allclose(s[:, 0], 100e6, rtol=1e-6) and np.abs(s[:, 1]).max() < 1e-3 * 100e6


def test_euler_buckling_pinned_column():
    Pcr, util, factor = euler_buckling([-1000.0, 500.0], [2.0, 2.0], E, I)
    assert rel(Pcr[0], np.pi**2 * E * I / 4.0) < 1e-12
    assert rel(util[0], 1000.0 / Pcr[0]) < 1e-12 and util[1] == 0.0 and np.isinf(factor[1])
    assert rel(euler_buckling(-1.0, 2.0, E, I, K=0.5)[0], 4 * Pcr[0]) < 1e-12


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"All {len(tests)} tests passed")
