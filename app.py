import os, io, json, shutil, subprocess, tempfile, base64
import streamlit as st
import pandas as pd
from continuous_beam_userinput_v2 import BeamModel
from pathlib import Path
import numpy as np
import bleach
from jsonschema import Draft7Validator, ValidationError
import numpy.linalg as npl

st.set_page_config(layout="wide")

# Add CSS for center alignment of all table texts and light grey table headers
st.markdown("""
    <style>
    table th, table td {
        text-align: center !important;
    }
    table th {
        background-color: #f0f0f0 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ───────────────── Config / Security ─────────────────
APP_DIR = Path(__file__).parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB app-enforced

ALLOWED_TAGS = ["b","strong","i","em","u","br","p","ul","ol","li","a"]
ALLOWED_ATTRS = {"a": ["href", "title", "target", "rel"]}

def sanitize_html(s: str) -> str:
    return bleach.clean(s or "", tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRS, strip=True)

def details_html_from_state() -> str:
    raw = st.session_state.get("project_details", "") or ""
    safe = sanitize_html(raw).replace("\n", "<br>")
    return (f'<div class="report-details" '
            f'style="text-align:left; font-size:12pt; white-space:normal;">{safe}</div>') if safe else ""

def load_logo_data_uri() -> str | None:
    for fname in ["logo.png", "static/logo.png", "assets/logo.png"]:
        p = APP_DIR / fname
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return f"data:image/png;base64,{b64}"
    return None

LOGO_DATA_URI = load_logo_data_uri()

SAVED_SCHEMA = {
  "type": "object",
  "properties": {
    "inputs": {
      "type": "object",
      "properties": {
        "L": {"type": "number", "minimum": 0.1, "maximum": 1e4},
        "E_Nmm2": {"type": "number", "minimum": 0.1, "maximum": 1e9},
        "I_cm4": {"type": "number", "minimum": 0.1, "maximum": 1e12},
        "perm_shear": {"type": ["number","null"]},
        "perm_moment": {"type": ["number","null"]},
        "grade": {"type": "string"},
        "supports": {"type": "array", "maxItems": 40},
        "point_loads_kN": {"type":"array", "maxItems": 200},
        "udls_kNpm": {"type":"array", "maxItems": 200},
        "springs_kNpm": {"type":"array", "maxItems": 80},
        "use_conc": {"type":"boolean"},
        "conc": {"type":["object","null"]},
        "use_conc_hydro": {"type":"boolean"},
        "hydro": {
          "type": ["object","null"],
          "properties": {
            "pressure_kNpm2": {"type":"number", "minimum":0},
            "influence_m": {"type":"number", "minimum":0},
            "x_start": {"type":"number", "minimum":0},
            "x_end": {"type":"number", "minimum":0}
          }
        },
        "SectionName": {"type": ["string","null"]}
      },
      "required": ["L","E_Nmm2","I_cm4"]
    },
    "project_details": {"type":["string","null"]},
    "results": {"type":"object"}
  },
  "required": ["inputs"],
  "additionalProperties": True
}

def validate_json_schema(data: dict) -> None:
    errors = sorted(Draft7Validator(SAVED_SCHEMA).iter_errors(data), key=lambda e: e.path)
    if errors:
        raise ValidationError("; ".join([e.message for e in errors]))

def scan_with_clamav_bytes(file_bytes: bytes):
    if shutil.which("clamscan") is None:
        return (None, "clamscan not available")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    try:
        proc = subprocess.run(["clamscan", "--no-summary", tmp_path], capture_output=True, text=True)
        if proc.returncode == 0: return (True, "Clean")
        if proc.returncode == 1: return (False, proc.stdout.strip())
        return (None, proc.stderr or proc.stdout)
    finally:
        os.remove(tmp_path)

st.markdown("""
<style>
/* Top logo (screen only, above tabs) */
.app-top-logo { display:flex; align-items:center; justify-content:center; margin:6px 0 6px; }
.app-top-logo img { height:52px; }

/* 75% Zoom wrapper for Analysis tab (screen only) */
.zoom-75 {
  transform: scale(0.5);
  transform-origin: top center;
  width: 2%;
  margin: 0 auto;
}

/* Screen: hide print-only blocks */
@media screen {
  .print-section, .print-graphs, .print-only { display:none !important; }
}

/* Print: general layout & typography */
@media print {
  @page { size: A4 portrait; margin: 1mm 12mm 12mm 12mm; }
  .block-container { padding-top: 0 !important; }
  header, footer,
  section[data-testid="stSidebar"],
  div[role="tablist"], .stTabs,
  .stFileUploader, .stDownloadButton,
  button, .stButton,
  .print-hide,
  .on-screen-results,
  .app-top-logo { display: none !important; }
  [data-testid="stHorizontalBlock"] > div:first-child { display: none !important; }
  [data-testid="stHorizontalBlock"] > div:last-child {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 auto !important;
  }
  body {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 12pt;
    line-height: 1.3;
    background: white;
  }
  .print-section h1 {
    text-align: center;
    font-size: 15pt;
    margin: 2mm 0 4mm 0;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 1px;
  }
  .print-section h2 {
    text-align: center;
    font-size: 15pt;
    margin: 6mm 0 3mm 0;
    text-transform: uppercase;
    border-bottom: 1px solid #333;
    padding-bottom: 2mm;
  }
  .print-section table {
    border-collapse: collapse;
    width: 90%;
    margin: 0 auto 8mm auto;
    table-layout: fixed;
  }
  .print-section th,
  .print-section td {
    border: 1px solid #666;
    padding: 4px 6px;
    text-align: center;
    font-size: 11pt;
    word-wrap: break-word;
  }
  .print-section th {
    background-color: #f0f0f0;
    font-weight: 600;
  }
}

/* Footer markers */
@media print {
  @page {
    @bottom-left {
      content: "★ Hut Beam Analysis ★";
      font-size: 11pt;
      color: #2c3e50;
      font-weight: 600;
    }
    @bottom-right {
      content: "Sheet " counter(page);
      font-size: 11pt;
      color: #2c3e50;
      font-weight: 600;
    }
  }
}

/* --- PRINT SETTINGS (UPDATED) --- */

/* Tables section: let it grow freely. No forced break BEFORE or AFTER. */
.print-section {
    /* no page-break rules, so if it grows, it will flow naturally to page 2 */
}

/* Graphs section: always start on a fresh page, and never split */
.print-graphs {
    page-break-before: always;   /* graphs start on a new page */
    page-break-inside: avoid;    /* graphs won't be split across pages */
    text-align: center;
}

/* Each graph image/canvas: keep each intact on one page */
.print-graphs img,
.print-graphs canvas {
    page-break-inside: avoid;
    break-inside: avoid;         /* modern browsers */
    display: block;
    margin: 10mm auto;
    max-width: 95%;
    height: auto;
}

/* keep print-only blocks visible during print */
.print-only, .print-section, .print-graphs { display: block !important; }

/* existing scale helper (unchanged) */
.print-scale-75 {
    transform: scale(0.75);
    transform-origin: top center;
    width: 75%;
    margin: 0 auto;
}

html, body { height: auto !important; }
.block-container { margin-bottom: 0 !important; padding-bottom: 0 !important; }
.custom-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 2px solid #2c3e50;
    margin-top: 10mm;
    padding-top: 4mm;
    font-size: 11pt;
    color: #2c3e50;
    font-weight: 600;
}
.custom-footer .footer-text { flex: 1; text-align: center; }
</style>
""", unsafe_allow_html=True)

if LOGO_DATA_URI:
    st.markdown(
        f'<div class="app-top-logo print-hide"><img src="{LOGO_DATA_URI}" alt="Logo"></div>',
        unsafe_allow_html=True
    )

sections_path = APP_DIR / "sections.csv"
required_cols = ["SectionName","E_Nmm2","I_cm4","PermissibleShear_kN","PermissibleMoment_kNm","Grade"]
if sections_path.exists():
    sections_df = pd.read_csv(sections_path)
    missing = [c for c in required_cols if c not in sections_df.columns]
    if missing:
        st.error(f"sections.csv is missing columns: {missing}")
        sections_df = pd.DataFrame(columns=required_cols)
else:
    st.warning("sections.csv not found. Using manual defaults.")
    sections_df = pd.DataFrame(columns=required_cols)

def safe_status(val, limit):
    if limit is None:
        return ""
    return "✅ SAFE" if abs(val) <= float(limit) else "❌ NOT SAFE"

if "saved_inputs" not in st.session_state:
    st.session_state["saved_inputs"] = {}
if "project_details" not in st.session_state:
    st.session_state["project_details"] = ""
if "last_ctx" not in st.session_state:
    st.session_state["last_ctx"] = None
if "last_res" not in st.session_state:
    st.session_state["last_res"] = None

saved_inputs = st.session_state["saved_inputs"]

tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Form", "Files", "Contact Us"])

# -------------------- Files TAB --------------------
with tab3:
    st.markdown("### Files")
    st.markdown('<div class="print-hide">[➕ New Calculation (new tab)](./)</div>', unsafe_allow_html=True)
    st.caption("Note: App-enforced upload limit 10 MB (JSON only).")
    uploaded = st.file_uploader("Load saved calculation (.json)", type=["json"])
    if uploaded is not None:
        content = uploaded.getvalue()
        if len(content) > MAX_UPLOAD_BYTES:
            st.error("File too large. App limit is 10 MB.")
        else:
            state, msg = scan_with_clamav_bytes(content)
            if state is False:
                st.error(f"Upload blocked by antivirus: {msg}")
            else:
                try:
                    payload = json.loads(content.decode("utf-8"))
                    validate_json_schema(payload)
                    st.session_state["saved_inputs"] = payload.get("inputs", {}) or {}
                    if "project_details" in payload:
                        st.session_state["project_details"] = sanitize_html(payload.get("project_details") or "")
                    st.success("Saved file loaded. Switch to the Analysis tab to see values prefilled.")
                    if state is None:
                        st.info("Antivirus not available on this host; using schema + sanitization.")
                except ValidationError as ve:
                    st.error(f"Invalid file structure: {ve}")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

# -------------------- Form TAB --------------------
with tab2:
    st.markdown("### Project Details")
    default_details = st.session_state.get("project_details", "")
    details = st.text_area("Enter client/project info", height=200, value=default_details)
    if st.button("Save details", type="primary"):
        st.session_state["project_details"] = details
        st.success("Details saved to session.")

# -------------------- Analysis TAB --------------------
with tab1:
    st.markdown('<div class="zoom-75">', unsafe_allow_html=True)
    left_col, right_col = st.columns([0.8,1.2])

    with left_col:
        st.markdown('<div class="print-hide">', unsafe_allow_html=True)
        st.title("Hut Beam Analysis")
        st.subheader("Member Section (from sections.csv)")

        # ─────────────── REPLACED BLOCK STARTS HERE ───────────────
        section_list = sections_df["SectionName"].dropna().tolist() if not sections_df.empty else []
        selected_section_default = saved_inputs.get("SectionName", "--Manual--")
        if selected_section_default not in (["--Manual--"] + section_list):
            selected_section_default = "--Manual--"

        # ➊ Dropdown to either choose or go manual
        selected_section = st.selectbox(
            "Select a Section",
            ["--Manual--"] + section_list,
            index=(["--Manual--"] + section_list).index(selected_section_default)
        )

        # ➋ If manual, show a text box so the user can type a custom name
        if selected_section == "--Manual--":
            manual_name = st.text_input("Type custom section name",
                                        value=saved_inputs.get("ManualSectionName",""))
            # add ** to indicate user typed it
            if manual_name.strip():
                selected_section = manual_name.strip() + "**"
        # ─────────────── REPLACED BLOCK ENDS HERE ───────────────

        if selected_section != "--Manual--" and not sections_df.empty:
            r = sections_df.loc[sections_df["SectionName"] == selected_section].iloc[0] if selected_section in sections_df["SectionName"].values else None
            if r is not None:
                default_E = float(r["E_Nmm2"]) if pd.notna(r["E_Nmm2"]) else 210000.0
                default_I = float(r["I_cm4"]) if pd.notna(r["I_cm4"]) else 1000.0
                perm_shear_default = float(r["PermissibleShear_kN"]) if pd.notna(r["PermissibleShear_kN"]) else 50.0
                perm_moment_default = float(r["PermissibleMoment_kNm"]) if pd.notna(r["PermissibleMoment_kNm"]) else 20.0
                grade_default = str(r["Grade"]) if pd.notna(r["Grade"]) else "Custom"
            else:
                # Selected was a manual custom name with **; fall back to defaults
                default_E = 210000.0; default_I = 1000.0
                perm_shear_default = 50.0; perm_moment_default = 20.0
                grade_default = "Custom"
        else:
            default_E = 210000.0; default_I = 1000.0
            perm_shear_default = 50.0; perm_moment_default = 20.0
            grade_default = "Custom"

        st.subheader("Beam Parameters")
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            L = st.number_input("Total beam length (m)", min_value=0.1,
                                value=float(saved_inputs.get("L", 4.0)))
        with col2:
            E_Nmm2 = st.number_input("E (N/mm²)", min_value=0.1,
                                     value=float(saved_inputs.get("E_Nmm2", default_E)))
        with col3:
            I_cm4 = st.number_input("I (cm⁴)", min_value=0.1,
                                    value=float(saved_inputs.get("I_cm4", default_I)))

        col4, col5, col6 = st.columns([1,1,1])
        with col4:
            perm_shear = st.number_input("Permissible Shear (kN)",
                                         value=float(saved_inputs.get("perm_shear", perm_shear_default)))
        with col5:
            perm_moment = st.number_input("Permissible Moment (kNm)",
                                          value=float(saved_inputs.get("perm_moment", perm_moment_default)))
        with col6:
            grade = st.text_input("Grade",
                                  value=str(saved_inputs.get("grade", grade_default)))

        E = E_Nmm2 * 1e6
        I = I_cm4 * 1e-8

        saved_supports = saved_inputs.get("supports", [])
        n_supp = st.number_input("Number of supports", min_value=0, step=1, value=int(len(saved_supports)))
        supports = []
        for i in range(int(n_supp)):
            default_pos = float(saved_supports[i][0]) if i < len(saved_supports) else 0.0
            default_kind = str(saved_supports[i][1]) if i < len(saved_supports) else "pin"
            c1, c2 = st.columns([1,1])
            with c1:
                pos = st.number_input(f"Support {i+1} position (m)", min_value=0.0, max_value=L,
                                      value=default_pos, key=f"supp_pos_{i}")
            with c2:
                kind = st.selectbox(f"Support {i+1} type",
                                    ["pin","roller","fixed","hinge"],
                                    index=["pin","roller","fixed","hinge"].index(
                                        default_kind if default_kind in ["pin","roller","fixed","hinge"] else "pin"),
                                    key=f"supp_kind_{i}")
            supports.append((pos, kind))

        saved_point_loads = saved_inputs.get("point_loads_kN", [])
        n_pl = st.number_input("Number of point loads", min_value=0, step=1, value=int(len(saved_point_loads)))
        point_loads = []
        for i in range(int(n_pl)):
            default_pos = float(saved_point_loads[i][0]) if i < len(saved_point_loads) else 0.0
            default_mag_kN = float(saved_point_loads[i][1]) if i < len(saved_point_loads) else 10.0
            c1, c2 = st.columns([1,1])
            with c1:
                pos = st.number_input(f"Point load {i+1} position (m)", min_value=0.0, max_value=L,
                                      value=default_pos, key=f"pl_pos_{i}")
            with c2:
                mag = st.number_input(f"Magnitude kN (+down)", value=default_mag_kN, key=f"pl_mag_{i}")
            point_loads.append((pos, mag*1e3))

        saved_udls = st.session_state["saved_inputs"].get("udls_kNpm", [])
        n_udl = st.number_input("Number of UDL segments", min_value=0, step=1, value=int(len(saved_udls)))
        udls = []
        for i in range(int(n_udl)):
            a_def = float(saved_udls[i][0]) if i < len(saved_udls) else 0.0
            b_def = float(saved_udls[i][1]) if i < len(saved_udls) else 0.0
            w1_def = float(saved_udls[i][2]) if i < len(saved_udls) else 5.0
            w2_def = float(saved_udls[i][3]) if i < len(saved_udls) else 5.0
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            with c1:
                a = st.number_input(f"UDL {i+1} start (m)", min_value=0.0, max_value=L, value=a_def, key=f"udl_a_{i}")
            with c2:
                b = st.number_input("End (m)", min_value=0.0, max_value=L, value=b_def, key=f"udl_b_{i}")
            with c3:
                w1 = st.number_input("Start intensity (kN/m)", value=w1_def, key=f"udl_w1_{i}")
            with c4:
                w2 = st.number_input("End intensity (kN/m)", value=w2_def, key=f"udl_w2_{i}")
            udls.append((a, b, w1*1e3, w2*1e3))

        saved_springs = saved_inputs.get("springs_kNpm", [])
        n_spring = st.number_input("Number of spring supports", min_value=0, step=1, value=int(len(saved_springs)))
        springs = []
        for i in range(int(n_spring)):
            pos_def = float(saved_springs[i][0]) if i < len(saved_springs) else 0.0
            k_def = float(saved_springs[i][1]) if i < len(saved_springs) else 100.0
            c1, c2 = st.columns([1,1])
            with c1:
                pos = st.number_input(f"Spring {i+1} position (m)", min_value=0.0, value=pos_def, key=f"spring_pos_{i}")
            with c2:
                k = st.number_input("Stiffness k (kN/m)", value=k_def, key=f"spring_k_{i}")
            springs.append((pos, k*1e3))

        saved_use_hydro = bool(saved_inputs.get("use_conc_hydro", False))
        hydro_defaults = saved_inputs.get("hydro", {}) if saved_use_hydro else {}
        use_conc_hydro = st.checkbox("Add Concrete Pressure (hydrostatic triangular)", value=saved_use_hydro)
        if use_conc_hydro:
            colh1, colh2 = st.columns([1,1])
            with colh1:
                pressure_kNpm2 = st.number_input("Concrete Pressure (kN/m²)", min_value=0.0,
                                                 value=float(hydro_defaults.get("pressure_kNpm2", 50.0)))
            with colh2:
                influence_m = st.number_input("Influence (m)", min_value=0.0,
                                              value=float(hydro_defaults.get("influence_m", 0.25)))
            colh3, colh4 = st.columns([1,1])
            with colh3:
                x_start = st.number_input("Start distance (m)", min_value=0.0, max_value=L,
                                          value=float(hydro_defaults.get("x_start", 0.0)))
            with colh4:
                x_end = st.number_input("End distance (m)", min_value=0.0, max_value=L,
                                        value=float(hydro_defaults.get("x_end", L)))
            Lh = pressure_kNpm2 / 25.0
            a_eff = max(x_start, x_end - Lh)
            b_eff = x_end
            a_clip = max(0.0, min(a_eff, L))
            b_clip = max(0.0, min(b_eff, L))
            L_eff = max(0.0, b_clip - a_clip)
            w_max_kNpm = pressure_kNpm2 * influence_m
            resultant_kN = 0.5 * w_max_kNpm * L_eff
            st.info(f"Lh = {Lh:.3f} m | effective span = {L_eff:.3f} m | max intensity = {w_max_kNpm:.3f} kN/m | Q = {resultant_kN:.3f} kN")

        solve_btn = st.button("Solve", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)  # end print-hide

    with right_col:
        if solve_btn:
            beam = BeamModel(L, E, I, perm_shear=perm_shear, perm_moment=perm_moment, grade=grade)
            for pos, kind in supports: beam.add_support(pos, kind)
            for pos, mag in point_loads: beam.add_point_load(pos, mag)
            for a, b, w1, w2 in udls: beam.add_udl(a, b, w1, w2)
            for pos, k in springs: beam.add_spring_support(pos, k)
            if use_conc_hydro:
                beam.add_concrete_pressure_hydro(
                    pressure_kNpm2=pressure_kNpm2, influence_m=influence_m, x_start=x_start, x_end=x_end
                )

            try:
                res = beam.solve(max_elem_len=0.001)
            except ValueError as e:
                st.error(str(e))
                res = None
            except npl.LinAlgError:
                st.error("Model is unstable/singular. Please add sufficient supports or springs.")
                res = None

            if res is not None:
                st.session_state["last_res"] = res
                st.session_state["last_ctx"] = {
                    "L": L, "E": E, "I": I, "perm_shear": perm_shear, "perm_moment": perm_moment, "grade": grade,
                    "supports": supports, "point_loads": point_loads, "udls": udls, "springs": springs,
                    "use_conc_hydro": bool(use_conc_hydro),
                    "SectionName": selected_section,
                    **({"hydro": {"pressure_kNpm2": pressure_kNpm2, "influence_m": influence_m, "x_start": x_start, "x_end": x_end}} if use_conc_hydro else {})
                }

                st.markdown('<div class="on-screen-results">', unsafe_allow_html=True)

                det_html = details_html_from_state()
                if det_html:
                    st.markdown(det_html, unsafe_allow_html=True)

                if LOGO_DATA_URI:
                    st.markdown(
                        f'<div style="text-align:center; margin-bottom:6px;">'
                        f'<img src="{LOGO_DATA_URI}" style="display:block;margin:0 auto 10px auto;width:120px;">'
                        f'</div>', unsafe_allow_html=True
                    )
                st.markdown('<h1 style="font-size:15px;text-align:center;">BEAM ANALYSIS REPORT</h1>', unsafe_allow_html=True)

                st.markdown('<h2 style="font-size:15px;">ANALYSIS RESULTS</h2>', unsafe_allow_html=True)
                st.subheader("Reactions")

                # ---------- UPDATED REACTIONS TABLE (on-screen) ----------
                if res["reactions_kN"]:
                    reac_df = pd.DataFrame(res["reactions_kN"], columns=["Support at (m)", "Ry (kN)"])
                    if not reac_df.empty:
                        reac_df = reac_df.sort_values(by="Support at (m)").reset_index(drop=True)
                        reac_df.insert(1, "Rx (kN)", 0.0)
                        reac_df["Mx (kN-m)"] = 0.0
                        reac_df.index = np.arange(1, len(reac_df)+1)
                        # ✅ Ry sign flip karke text format karo
                        reac_df["Ry (kN)"] = reac_df["Ry (kN)"].apply(lambda v: f"{v * -1:.2f} kN")
                    st.table(reac_df)
                else:
                    st.write("No supports / reactions.")

                x = res["x"]; M = res["M_kNm"]; V = res["V_kN"]; w = res["w_mm"]
                M_max = float(M.max()) if M.size else 0.0
                M_min = float(M.min()) if M.size else 0.0
                V_max = float(V.max()) if V.size else 0.0
                V_min = float(V.min()) if V.size else 0.0
                w_max = float(w.max()) if w.size else 0.0
                w_min = float(w.min()) if w.size else 0.0

                M_gov = max(abs(M_max), abs(M_min))
                V_gov = max(abs(V_max), abs(V_min))

                force_ext = pd.DataFrame([
                    ["Bending Moment", f"{M_max:.3f}", f"{M_min:.3f}",
                     f"{perm_moment:.3f}" if perm_moment is not None else "",
                     safe_status(M_gov, perm_moment)],
                    ["Shear", f"{V_max:.3f}", f"{V_min:.3f}",
                     f"{perm_shear:.3f}" if perm_shear is not None else "",
                     safe_status(V_gov, perm_shear)],
                    ["Deformation", f"{w_max:.3f}", f"{w_min:.3f}", "", ""]
                ], columns=["Result", "Max", "Min", "Permissible", "Status"])
                force_ext.index = np.arange(1, len(force_ext)+1)
                st.markdown("**Force Extremes**", unsafe_allow_html=True)
                st.table(force_ext)

                st.pyplot(beam.plot_FBD(res))
                st.pyplot(beam.plot_SFD(res))
                st.pyplot(beam.plot_BMD(res))

                # ✅ Deflection plot without re-setting grid/locators
                fig = beam.plot_deflection(res)
                ax = fig.gca()
                ax.invert_yaxis()
                st.pyplot(fig)

                save_inputs = {
                    "L": L, "E_Nmm2": E_Nmm2, "I_cm4": I_cm4,
                    "perm_shear": perm_shear, "perm_moment": perm_moment, "grade": grade,
                    "supports": supports,
                    "point_loads_kN": [(p[0], p[1]/1e3) for p in point_loads],
                    "udls_kNpm": [(a,b,w1/1e3,w2/1e3) for (a,b,w1,w2) in udls],
                    "springs_kNpm": [(p[0], p[1]/1e3) for p in springs],
                    "use_conc_hydro": bool(use_conc_hydro)
                }
                if use_conc_hydro:
                    save_inputs["hydro"] = {
                        "pressure_kNpm2": float(pressure_kNpm2),
                        "influence_m": float(influence_m),
                        "x_start": float(x_start),
                        "x_end": float(x_end)
                    }

                save_payload = {
                    "inputs": save_inputs,
                    "project_details": st.session_state.get("project_details", ""),
                    "results": {
                        "x_nodes": res["x_nodes"].tolist(),
                        "deflection_mm": res["deflection_mm"].tolist(),
                        "x": x.tolist(), "M_kNm": M.tolist(), "V_kN": V.tolist(), "w_mm": w.tolist(),
                        "reactions_kN": res["reactions_kN"],
                    }
                }

                st.markdown('<div class="print-hide">', unsafe_allow_html=True)
                st.download_button("💾 Save Calculation",
                    data=json.dumps(save_payload, indent=2),
                    file_name="beam_calculation.json",
                    mime="application/json",
                    use_container_width=True
                )
                st.markdown("""
                    <div style="margin-top: 12px; margin-bottom: 4px; text-align:center;">
                        <button onclick="window.top.print()" style="font-size:16px; padding:6px 12px;">
                            🖨️ Print Full Report
                        </button>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Contact Us TAB --------------------
with tab4:
    st.markdown("""
    <hr>
    ### About This Software
    This project is <b>open source</b> and provided <b>free of charge</b> to help make your work easier and support educational goals.
    <ul>
      <li><b>Free &amp; Open Source:</b> You are welcome to use, modify, and share this software in accordance with the applicable open-source license.</li>
      <li><b>Community Sharing:</b> Feel free to share it with others who might benefit.</li>
      <li><b>User Responsibility:</b> All analyses and results generated using this software are performed at your own discretion, and <b>you are solely responsible</b> for any outcomes or decisions based on its use.</li>
    </ul>
    <hr>
    <p><b>Developer Contact:</b> <a href="mailto:kaiumattar@gmail.com">kaiumattar@gmail.com</a></p>
    <hr>
    """, unsafe_allow_html=True)

# -------- PRINT-ONLY RENDER (any tab) ----------
def render_print_only():
    res = st.session_state.get("last_res")
    ctx = st.session_state.get("last_ctx")
    if not res or not ctx:
        return
    beam_p = BeamModel(ctx["L"], ctx["E"], ctx["I"],
                       perm_shear=ctx["perm_shear"], perm_moment=ctx["perm_moment"], grade=ctx["grade"])
    for pos, kind in ctx["supports"]: beam_p.add_support(pos, kind)
    for pos, mag in ctx["point_loads"]: beam_p.add_point_load(pos, mag)
    for a,b,w1,w2 in ctx["udls"]: beam_p.add_udl(a,b,w1,w2)
    for pos,k in ctx["springs"]: beam_p.add_spring_support(pos,k)
    if ctx.get("use_conc_hydro") and "hydro" in ctx:
        h = ctx["hydro"]
        beam_p.add_concrete_pressure_hydro(h["pressure_kNpm2"], h["influence_m"], h["x_start"], h["x_end"])

    st.markdown('<div class="print-section print-only print-scale-75">', unsafe_allow_html=True)

    if LOGO_DATA_URI:
        st.markdown(
            f'<div style="text-align:center; margin:0 0 4mm 0;">'
            f'<img src="{LOGO_DATA_URI}" style="display:block;margin:0 auto;width:120px;">'
            f'</div>', unsafe_allow_html=True
        )
    det_html = details_html_from_state()
    if det_html:
        st.markdown(det_html, unsafe_allow_html=True)

    st.markdown('<h1 style="font-size:15px;">BEAM ANALYSIS REPORT</h1>', unsafe_allow_html=True)

    section_name = ctx.get("SectionName", "--Manual--")

    input_summary = pd.DataFrame([
        [f"{section_name} Length", f"{ctx['L']} m"],
        ["Moment of Inertia (I)", f"{(ctx['I']*1e8):.0f} cm⁴"],
        ["Young's Modulus (E)", f"{(ctx['E']/1e6):.0f} MPa"],
        ["Grade", ctx["grade"]]
    ], columns=["General Info", "Value"])
    input_summary.index = np.arange(1, len(input_summary)+1)
    st.markdown('<h2 style="font-size:15px;">INPUT SUMMARY</h2>', unsafe_allow_html=True)
    st.table(input_summary)

    # ---------- UPDATED REACTIONS TABLE (print-only) ----------
    reac_df = pd.DataFrame(res["reactions_kN"], columns=["Support at (m)", "Ry (kN)"])
    if not reac_df.empty:
        reac_df = reac_df.sort_values(by="Support at (m)").reset_index(drop=True)
        reac_df.insert(1, "Rx (kN)", 0.0)
        reac_df["Mx (kN-m)"] = 0.0
        reac_df.index = np.arange(1, len(reac_df)+1)
        # ✅ Ry sign flip and text formatting
        reac_df["Ry (kN)"] = reac_df["Ry (kN)"].apply(lambda v: f"{v * -1:.2f} kN")
    st.markdown('<h2 style="font-size:15px;">ANALYSIS RESULTS</h2>', unsafe_allow_html=True)
    st.markdown("**Reactions**", unsafe_allow_html=True)
    st.table(reac_df)

    x = res["x"]; M = res["M_kNm"]; V = res["V_kN"]; w = res["w_mm"]
    M_max = float(M.max()) if len(M) else 0.0
    M_min = float(M.min()) if len(M) else 0.0
    V_max = float(V.max()) if len(V) else 0.0
    V_min = float(V.min()) if len(V) else 0.0
    w_max = float(w.max()) if len(w) else 0.0
    w_min = float(w.min()) if len(w) else 0.0

    M_gov = max(abs(M_max), abs(M_min))
    V_gov = max(abs(V_max), abs(V_min))

    force_ext = pd.DataFrame([
        ["Bending Moment", f"{M_max:.3f}", f"{M_min:.3f}",
         f"{ctx['perm_moment']:.3f}" if ctx["perm_moment"] is not None else "",
         safe_status(M_gov, ctx["perm_moment"])],
        ["Shear", f"{V_max:.3f}", f"{V_min:.3f}",
         f"{ctx['perm_shear']:.3f}" if ctx["perm_shear"] is not None else "",
         safe_status(V_gov, ctx["perm_shear"])],
        ["Deformation", f"{w_max:.3f}", f"{w_min:.3f}", "", ""]
    ], columns=["Result", "Max", "Min", "Permissible", "Status"])
    force_ext.index = np.arange(1, len(force_ext)+1)
    st.markdown("**Force Extremes**", unsafe_allow_html=True)
    st.table(force_ext)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="print-graphs print-only">', unsafe_allow_html=True)
    st.pyplot(beam_p.plot_FBD(res))
    st.pyplot(beam_p.plot_SFD(res))
    st.pyplot(beam_p.plot_BMD(res))

    # ✅ Deflection plot without re-setting grid/locators
    fig = beam_p.plot_deflection(res)
    ax = fig.gca()
    ax.invert_yaxis()
    st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)
    # temp commit check


render_print_only()
