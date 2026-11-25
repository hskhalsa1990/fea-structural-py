# Summary: fea-structural-py (with 3D diagrams)

- **Repo name:** `fea-structural-py`
- **Description (GitHub, under 350 characters):** Finite Element Analysis in Python (NumPy and SciPy only): 2D/3D truss, beam, frame and plate solvers with static, modal and buckling analysis, stress and deflection plots, comparison tables, and validation against closed-form results. Notebook, tests and Streamlit app included.
- **Topics:** finite-element-analysis, fea, structural-analysis, mechanical-engineering, python, numpy, scipy, matplotlib, jupyter-notebook, engineering-simulation
- **Flat zip:** fea_core.py, FEA_Structural_Analysis.ipynb, test_fea.py, app.py, make_figures.py, comparison_tables.md, 20 fig_*.png, 10 3d_*.jpg, README.md, requirements.txt, LICENSE
- **v1.1 additions:** Frame2D (+ modal), Truss3D, plane strain, principal stresses, safety factor, incompatible-modes Q4, app.py (Streamlit), notebook sections 10 and 11.
- **v1.2 additions:** euler_buckling, make_figures.py (6 new 2D/3D charts) and comparison_tables.md, 19 tests.
- **3D images included (10):** stiffness_matrix_assembly (regenerated with correct 4x4 layout), plate_hole_stress_concentration (physics correct: max at top/bottom of hole), plus fea_workflow, mesh_types, cantilever_beam_deflection (formula fixed to delta = P L^3 / (3 E I)), simply_supported_beam, truss_bridge, mesh_convergence, von_mises_failure, vibration_mode_shapes
- **Notes on included images:** numbers inside them are illustrative (truss 50 kN Pratt span 24 m vs notebook 20 kN Warren; mesh counts and errors in the refinement picture; vibration picture shows a supported beam with sin(n pi x / L), not the cantilever used in the notebook).
- **Validated results:** 19 tests pass; beam and truss exact; cantilever f1 = 33.421 Hz; plate-with-hole Kt = 3.09 (Kirsch 3.00, handbook 3.02).
