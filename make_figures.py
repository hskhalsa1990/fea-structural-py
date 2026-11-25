"""Generate the extra 2D / 3D charts and comparison tables.

Run:  python make_figures.py
Writes fig_*.png in this folder (flat project) and comparison_tables.md.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from fea_core import (MATERIALS, Beam, Truss3D, PlaneStressQ4, euler_buckling,
                      rect_mesh, quarter_hole_mesh, add_edge_load)

plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#7f7f7f"
tables = []


def table(title, header, rows):
    out = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    tables.append("\n".join(out) + "\n")


# ---------------------------------------------------------------- 1. beam validation (2D bar chart + table)
E_s, I_s, L_s, P, q = 210e9, 8e-6, 2.0, 1000.0, 1000.0
x5 = np.linspace(0, L_s, 5)
cases = []
b = Beam(x5, E_s, I_s); b.solve([0, 1], point_loads={4: -P})
cases.append(("Cantilever, tip load", abs(b.u[8]), P * L_s**3 / (3 * E_s * I_s)))
b.solve([0, 1], udl=-q)
cases.append(("Cantilever, UDL", abs(b.u[8]), q * L_s**4 / (8 * E_s * I_s)))
b.solve([0, 8], point_loads={2: -P})
cases.append(("Simply supported, mid load", abs(b.u[4]), P * L_s**3 / (48 * E_s * I_s)))
b.solve([0, 8], udl=-q)
cases.append(("Simply supported, UDL", abs(b.u[4]), 5 * q * L_s**4 / (384 * E_s * I_s)))
fig, ax = plt.subplots(figsize=(9, 4))
xs = np.arange(len(cases)); w = 0.38
ax.bar(xs - w / 2, [c[1] * 1e3 for c in cases], w, label="FEA", color=BLUE)
ax.bar(xs + w / 2, [c[2] * 1e3 for c in cases], w, label="Analytical", color=ORANGE)
ax.set_xticks(xs); ax.set_xticklabels([c[0] for c in cases], fontsize=8, rotation=12)
ax.set_ylabel("Deflection (mm)"); ax.set_title("Beam solver vs closed-form results"); ax.legend()
plt.tight_layout(); plt.savefig("fig_validation_bars.png", dpi=150); plt.close()
table("Beam solver vs analytical (E = 210 GPa, I = 8e-6 m4, L = 2 m, 1 kN or 1 kN/m)",
      ["Case", "FEA (mm)", "Analytical (mm)", "Relative error"],
      [(n, f"{a*1e3:.5f}", f"{e*1e3:.5f}", f"{abs(a-e)/e:.1e}") for n, a, e in cases])

# ---------------------------------------------------------------- 2. element comparison (2D line chart + table)
Ec, nuc, tc, Lx, Ly, Pc = 210e9, 0.30, 0.1, 1.0, 0.2, 1000.0
Ic = tc * Ly**3 / 12
d_eb = Pc * Lx**3 / (3 * Ec * Ic)
d_tim = d_eb + Pc * Lx / ((5 / 6) * (Ec / (2 * (1 + nuc))) * tc * Ly)


def tip(nx, ny, inc):
    nd, el = rect_mesh(Lx, Ly, nx, ny)
    m = PlaneStressQ4(nd, el, Ec, nuc, tc, incompatible=inc)
    F = np.zeros(m.ndof)
    rt = np.where(np.isclose(nd[:, 0], Lx))[0]; lf = np.where(np.isclose(nd[:, 0], 0))[0]
    add_edge_load(F, nd, rt, 1, -Pc / (tc * Ly), tc)
    u = m.solve(F, [2 * i for i in lf] + [2 * i + 1 for i in lf])
    return m.ndof, -np.mean(u[2 * rt + 1])


meshes = [(5, 1), (10, 2), (20, 4), (40, 8), (80, 16)]
rows, std, inc = [], [], []
for nx, ny in meshes:
    d, a = tip(nx, ny, False); _, c = tip(nx, ny, True)
    std.append(a / d_tim); inc.append(c / d_tim)
    rows.append((f"{nx}x{ny}", d, f"{a/d_tim:.4f}", f"{c/d_tim:.4f}"))
dofs = [r[1] for r in rows]
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(dofs, std, "s-", color=RED, lw=2, label="Standard Q4")
ax.semilogx(dofs, inc, "o-", color=GREEN, lw=2, label="Q4 + incompatible modes")
ax.axhline(1, color=GREY, ls="--"); ax.set_xlabel("Degrees of freedom"); ax.set_ylabel("Tip deflection / Timoshenko")
ax.set_title("Element comparison, slender cantilever plate"); ax.legend()
plt.tight_layout(); plt.savefig("fig_element_comparison.png", dpi=150); plt.close()
table("Q4 element comparison (cantilever plate, tip deflection / Timoshenko)",
      ["Mesh", "DOFs", "Standard Q4", "Q4 + incompatible modes"], rows)

# ---------------------------------------------------------------- 3. material comparison (2D charts + table)
side, Lb, Pb = 0.04, 1.0, 1000.0
Ab, Ibb = side**2, side**4 / 12
names, defl, mass, sf = [], [], [], []
for n, p in MATERIALS.items():
    bm = Beam(np.linspace(0, Lb, 11), p["E"], Ibb); bm.solve([0, 1], point_loads={10: -Pb})
    sig = Pb * Lb * (side / 2) / Ibb
    names.append(n); defl.append(abs(bm.u[20]) * 1e3); mass.append(p["rho"] * Ab * Lb); sf.append(p["sy"] / sig)
fig, axs = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, vals, lab, col in zip(axs, (defl, mass, sf), ("Tip deflection (mm)", "Mass (kg)", "Safety factor (yield)"), (BLUE, ORANGE, GREEN)):
    ax.bar([n.split()[0] for n in names], vals, color=col); ax.set_title(lab)
    for i, v in enumerate(vals): ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
plt.suptitle("Cantilever 40 mm square, 1 m, 1 kN tip load"); plt.tight_layout()
plt.savefig("fig_material_charts.png", dpi=150); plt.close()
table("Material comparison (40 mm square cantilever, 1 m, 1 kN)",
      ["Material", "E (GPa)", "Tip deflection (mm)", "Mass (kg)", "Stiffness / mass", "Safety factor"],
      [(n, f"{MATERIALS[n]['E']/1e9:.1f}", f"{d:.3f}", f"{m:.2f}", f"{(Pb/(d*1e-3))/m:.0f}", f"{s:.2f}")
       for n, d, m, s in zip(names, defl, mass, sf)])

# ---------------------------------------------------------------- 4. plate with hole: 3D surface + 2D contour
a_, Lq, t, s0 = 0.05, 0.5, 0.01, 50e6
nr, nt = 24, 48
nd, el = quarter_hole_mesh(a_, Lq, nr, nt)
m = PlaneStressQ4(nd, el, 210e9, 0.30, t, incompatible=True)
F = np.zeros(m.ndof)
add_edge_load(F, nd, np.where(np.isclose(nd[:, 0], Lq))[0], 0, s0, t)
fixed = [2 * i + 1 for i in np.where(np.isclose(nd[:, 1], 0))[0]] + [2 * i for i in np.where(np.isclose(nd[:, 0], 0))[0]]
m.solve(F, fixed)
sh = m.nodal_stress(); vm = m.von_mises(sh) / s0
tris = np.vstack([el[:, [0, 1, 2]], el[:, [0, 2, 3]]])
n = len(nd)
X = np.concatenate([nd[:, 0], -nd[:, 0], -nd[:, 0], nd[:, 0]])
Y = np.concatenate([nd[:, 1], nd[:, 1], -nd[:, 1], -nd[:, 1]])
Z = np.tile(vm, 4)
T = np.vstack([tris + k * n for k in range(4)])
fig = plt.figure(figsize=(11, 4.8))
ax = fig.add_subplot(121, projection="3d")
surf = ax.plot_trisurf(Triangulation(X, Y, T), Z, cmap="inferno", linewidth=0, antialiased=True)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("von Mises / nominal")
ax.set_title("3D stress surface"); ax.view_init(35, -60)
ax2 = fig.add_subplot(122)
tc2 = ax2.tricontourf(Triangulation(X, Y, T), Z, 20, cmap="inferno"); ax2.set_aspect("equal")
fig.colorbar(tc2, ax=ax2, label="von Mises / nominal"); ax2.set_title("2D contour")
plt.tight_layout(); plt.savefig("fig_3d_stress_surface.png", dpi=150); plt.close()
Kt = sh[nt * (nr + 1), 0] / s0

# ---------------------------------------------------------------- 5. 3D design surface: cantilever deflection vs L and h
Ls = np.linspace(0.5, 3.0, 26); hs = np.linspace(0.03, 0.12, 26)
steel = MATERIALS["Steel (structural)"]; bw = 0.04
Zd = np.zeros((len(hs), len(Ls)))
for i, h in enumerate(hs):
    for j, Lj in enumerate(Ls):
        bm = Beam(np.linspace(0, Lj, 5), steel["E"], bw * h**3 / 12); bm.solve([0, 1], point_loads={4: -Pb})
        Zd[i, j] = abs(bm.u[8]) * 1e3
LL, HH = np.meshgrid(Ls, hs)
fig = plt.figure(figsize=(7.5, 5.5)); ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(LL, HH * 1e3, Zd, cmap="viridis", edgecolor="none")
ax.set_xlabel("Length (m)"); ax.set_ylabel("Height (mm)"); ax.set_zlabel("Tip deflection (mm)")
ax.set_title("Steel cantilever, 40 mm wide, 1 kN tip load"); ax.view_init(25, -125)
plt.tight_layout(); plt.savefig("fig_3d_design_surface.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 6. 3D tower buckling chart + table
ang = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]], float)
nodes3 = np.vstack([np.c_[ang * 1.0, np.zeros(4)], np.c_[ang * 0.5, np.full(4, 3.0)]])
legs = [(i, 4 + i) for i in range(4)]; ring = [(4 + i, 4 + (i + 1) % 4) for i in range(4)]; diag = [(i, 4 + (i + 1) % 4) for i in range(4)]
members = legs + ring + diag
A_t = 8e-4; OD = 0.05; ID = np.sqrt(OD**2 - 4 * A_t / np.pi); I_t = np.pi / 64 * (OD**4 - ID**4)
tower = Truss3D(nodes3, members, 210e9, A_t)
res = tower.solve({4: (10e3, 0, -20e3)}, [(i, d) for i in range(4) for d in range(3)])
Lm = np.array([tower.geometry(k)[2] for k in range(len(members))])
Pcr, util, factor = euler_buckling(res["N"], Lm, 210e9, I_t)
labels = [f"L{i+1}" for i in range(4)] + [f"R{i+1}" for i in range(4)] + [f"D{i+1}" for i in range(4)]
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cols = [RED if v > 0.5 else (ORANGE if v > 0.25 else BLUE) for v in util]
axs[0].bar(labels, res["N"] / 1e3, color=[BLUE if v >= 0 else RED for v in res["N"]])
axs[0].set_ylabel("Axial force (kN)"); axs[0].set_title("Tower member forces (+ tension, - compression)")
axs[1].bar(labels, util, color=cols); axs[1].axhline(1, color="k", ls="--"); axs[1].set_ylabel("|N| / Pcr")
axs[1].set_title("Euler buckling utilisation (K = 1)")
plt.tight_layout(); plt.savefig("fig_buckling_chart.png", dpi=150); plt.close()
table(f"3D tower: member forces and Euler buckling (tube OD {OD*1e3:.0f} mm, A = {A_t*1e4:.1f} cm2, K = 1)",
      ["Member", "Length (m)", "Force (kN)", "Pcr (kN)", "Utilisation"],
      [(lab, f"{Lm[k]:.3f}", f"{(0.0 if abs(res['N'][k]) < 5 else res['N'][k])/1e3:.2f}", f"{Pcr[k]/1e3:.1f}", f"{util[k]:.2f}") for k, lab in enumerate(labels)])

# ---------------------------------------------------------------- 7. solver overview table
table("Solver overview", ["Class", "Element", "DOF per node", "Static", "Modal", "Notes"], [
    ("Truss2D", "2-node bar", 2, "yes", "no", "Member forces, reactions"),
    ("Truss3D", "2-node bar", 3, "yes", "no", "Space trusses, use with euler_buckling"),
    ("Beam", "Hermite cubic", 2, "yes", "yes", "Shear and moment diagrams"),
    ("Frame2D", "Beam-column", 3, "yes", "yes", "Rigid joints, member UDL, end forces"),
    ("PlaneStressQ4", "4-node quad", 2, "yes", "no", "Plane stress or strain, optional incompatible modes")])
table("Plate with hole (nominal 50 MPa)", ["Quantity", "Value"],
      [("Kt (Q4 + incompatible modes, 24x48)", f"{Kt:.3f}"), ("Kirsch (infinite plate)", "3.000"),
       ("Peak von Mises / nominal", f"{vm.max():.3f}"),
       ("Safety factor vs yield (steel, 250 MPa)", f"{250e6/(vm.max()*s0):.2f}")])
open("comparison_tables.md", "w").write("# Comparison tables\n\nGenerated by make_figures.py\n\n" + "\n".join(tables))
print("done")
