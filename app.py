"""Streamlit dashboard for fea-structural-py.  Run:  streamlit run app.py"""
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from fea_core import (Beam, PlaneStressQ4, Frame2D, MATERIALS, quarter_hole_mesh, add_edge_load)

st.set_page_config(page_title="FEA Structural", layout="wide")
st.title("fea-structural-py")
tab_beam, tab_plate, tab_frame = st.tabs(["Beam", "Plate with hole", "Portal frame"])

with tab_beam:
    c1, c2, c3 = st.columns(3)
    mat = c1.selectbox("Material", list(MATERIALS), key="bm")
    L = c1.number_input("Length (m)", 0.5, 20.0, 2.0)
    b = c2.number_input("Width (mm)", 5.0, 500.0, 40.0) / 1000
    h = c2.number_input("Height (mm)", 5.0, 500.0, 40.0) / 1000
    P = c3.number_input("Tip load (N, down)", 1.0, 1e6, 1000.0)
    supp = c3.radio("Support", ["Cantilever", "Simply supported (load at mid)"])
    I = b * h**3 / 12
    n = 20
    beam = Beam(np.linspace(0, L, n + 1), MATERIALS[mat]["E"], I)
    if supp == "Cantilever":
        beam.solve([0, 1], point_loads={n: -P})
    else:
        beam.solve([0, 2 * n], point_loads={n // 2: -P})
    x, v, V, M = beam.diagram()
    sig = np.abs(M).max() * (h / 2) / I
    st.metric("Max bending stress (MPa)", f"{sig/1e6:.1f}")
    st.metric("Safety factor (yield)", f"{MATERIALS[mat]['sy']/sig:.2f}")
    fig, ax = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
    for a, y, lab in zip(ax, (v * 1e3, V, M), ("Deflection (mm)", "Shear (N)", "Moment (N m)")):
        a.plot(x, y); a.set_ylabel(lab); a.grid(alpha=.3)
    ax[-1].set_xlabel("x (m)")
    st.pyplot(fig)

with tab_plate:
    mat = st.selectbox("Material", list(MATERIALS), key="pm")
    s0 = st.slider("Nominal stress (MPa)", 10, 200, 50) * 1e6
    nr, nt = 16, 16
    a_, Lq, t = 0.01, 0.1, 0.005
    nodes, elems = quarter_hole_mesh(a_, Lq, nr, nt)
    m = PlaneStressQ4(nodes, elems, MATERIALS[mat]["E"], MATERIALS[mat]["nu"], t)
    F = np.zeros(m.ndof)
    add_edge_load(F, nodes, np.where(np.isclose(nodes[:, 0], Lq))[0], 0, s0, t)
    fixed = [2 * i + 1 for i in np.where(np.isclose(nodes[:, 1], 0))[0]] + \
            [2 * i for i in np.where(np.isclose(nodes[:, 0], 0))[0]]
    m.solve(F, fixed)
    vm = m.von_mises(m.nodal_stress())
    st.metric("Peak von Mises (MPa)", f"{vm.max()/1e6:.1f}")
    st.metric("Safety factor", f"{m.safety_factor(vm, MATERIALS[mat]['sy']).min():.2f}")
    fig, ax = plt.subplots(figsize=(6, 5))
    tc = ax.tricontourf(nodes[:, 0], nodes[:, 1], vm / 1e6, 20, cmap="inferno")
    ax.set_aspect("equal"); fig.colorbar(tc, label="von Mises (MPa)")
    st.pyplot(fig)

with tab_frame:
    W = st.number_input("Span (m)", 1.0, 20.0, 4.0)
    H = st.number_input("Height (m)", 1.0, 10.0, 3.0)
    Fx = st.number_input("Lateral load at beam level (N)", 0.0, 1e6, 10000.0)
    nodes = [[0, 0], [0, H], [W, H], [W, 0]]
    fr = Frame2D(nodes, [[0, 1], [1, 2], [3, 2]], 210e9, 8e-3, 1e-4)
    r = fr.solve([(0, 0), (0, 1), (0, 2), (3, 0), (3, 1), (3, 2)], nodal_loads={1: (Fx, 0, 0)})
    fig, ax = plt.subplots(figsize=(6, 4))
    xy = np.array(nodes, float)
    sc = 0.2 * W / max(np.abs(r["u"][:, :2]).max(), 1e-12)
    for i, j in fr.members:
        ax.plot(*xy[[i, j]].T, "k--", lw=1)
        ax.plot(*(xy[[i, j]] + sc * r["u"][[i, j], :2]).T, "r-", lw=2)
    ax.set_aspect("equal"); ax.set_title(f"Deformed shape (x{sc:.0f})")
    st.pyplot(fig)
    st.metric("Sway (mm)", f"{r['u'][1,0]*1e3:.2f}")
