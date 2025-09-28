import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, MultipleLocator

class BeamModel:
    """Finite-element (Euler–Bernoulli) continuous beam model.
    All internal forces/loads in N or N/m; user-facing outputs in kN or kNm.
    """
    def __init__(self, L, E, I, perm_shear=None, perm_moment=None, grade=""):
        self.L = float(L)
        self.E = float(E)  # N/m²
        self.I = float(I)  # m⁴
        self.supports = []
        self.point_loads = []
        self.udls = []  # tuples (a, b, w1, w2) in N/m
        self.extra_nodes = {0.0, float(L)}
        self.spring_supports = []
        self.perm_shear = perm_shear  # kN
        self.perm_moment = perm_moment  # kNm
        self.grade = grade

        # Hydro (concrete pressure) visual parts
        self._hydro_parts = []
        self._hydro_udl_tuples = set()

    def add_support(self, x, kind='pin'):
        self.supports.append((float(x), kind.lower()))
        self.extra_nodes.add(float(x))

    def add_point_load(self, x, P):
        self.point_loads.append((float(x), float(P)))
        self.extra_nodes.add(float(x))

    def add_udl(self, a, b, w1, w2=None):
        a, b = float(a), float(b)
        if b < a:
            a, b = b, a
        if w2 is None:
            w2 = w1
        tup = (a, b, float(w1), float(w2))
        self.udls.append(tup)
        self.extra_nodes.update([a, b])
        return tup

    def add_concrete_pressure_hydro(self, pressure_kNpm2, influence_m, x_start, x_end):
        """Create a two-part UDL for concrete (hydrostatic) pressure."""
        L = self.L
        x_start = max(0.0, min(float(x_start), L))
        x_end = max(0.0, min(float(x_end), L))
        if x_end <= x_start:
            return

        Lh = float(pressure_kNpm2) / 25.0  # 25 kN/m³ = unit weight of concrete
        w_max_kNpm = float(pressure_kNpm2) * float(influence_m)  # kN/m line load

        flat_end = max(x_start, x_end - Lh)

        # 1️⃣ flat part
        if flat_end - x_start > 1e-12:
            t = self.add_udl(
                x_start, flat_end,
                w_max_kNpm * 1e3,  # to N/m
                w_max_kNpm * 1e3
            )
            self._hydro_udl_tuples.add(t)

        # 2️⃣ triangular tail
        if x_end - flat_end > 1e-12:
            t = self.add_udl(flat_end, x_end, w_max_kNpm * 1e3, 0.0)
            self._hydro_udl_tuples.add(t)

        self._hydro_parts.append({
            "start": x_start,
            "end": x_end,
            "Lh": Lh,
            "wmax_kNpm": w_max_kNpm,
            "flat": (x_start, flat_end),
            "tri": (flat_end, x_end, w_max_kNpm)
        })
        self.extra_nodes.update([x_start, x_end, flat_end])

    def add_spring_support(self, x, k):
        self.spring_supports.append((float(x), float(k)))
        self.extra_nodes.add(float(x))

    def _build_mesh(self, max_elem_len=0.25):
        pts = sorted(self.extra_nodes)
        refined = [pts[0]]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            nseg = max(1, int(np.ceil((b - a) / max_elem_len)))
            for k in range(1, nseg + 1):
                refined.append(a + (b - a) * k / nseg)
        xs = np.array(sorted({round(x, 9) for x in refined}), dtype=float)
        return xs

    @staticmethod
    def _consistent_load_vec_udl(w, L):
        return np.array(
            [w * L / 2, w * L ** 2 / 12, w * L / 2, -w * L ** 2 / 12],
            dtype=float
        )

    def solve(self, max_elem_len=0.25):
        x_nodes = self._build_mesh(max_elem_len=max_elem_len)
        n = len(x_nodes)
        dof = 2 * n

        K = np.zeros((dof, dof))
        F = np.zeros(dof)

        for e in range(n - 1):
            xa, xb = x_nodes[e], x_nodes[e + 1]
            Le = xb - xa

            ke = (self.E * self.I / Le ** 3) * np.array(
                [
                    [12, 6 * Le, -12, 6 * Le],
                    [6 * Le, 4 * Le ** 2, -6 * Le, 2 * Le ** 2],
                    [-12, -6 * Le, 12, -6 * Le],
                    [6 * Le, 2 * Le ** 2, -6 * Le, 4 * Le ** 2],
                ],
                float,
            )
            idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
            K[np.ix_(idx, idx)] += ke

            # UDL contributions
            for (a, b, w1, w2) in self.udls:
                ov_a = max(xa, a)
                ov_b = min(xb, b)
                overlap = max(0.0, ov_b - ov_a)
                if overlap > 1e-12:
                    if b - a > 1e-12:
                        t1 = (ov_a - a) / (b - a)
                        t2 = (ov_b - a) / (b - a)
                        w_a = w1 + (w2 - w1) * max(0.0, min(1.0, t1))
                        w_b = w1 + (w2 - w1) * max(0.0, min(1.0, t2))
                        w_avg_local = 0.5 * (w_a + w_b)
                    else:
                        w_avg_local = 0.5 * (w1 + w2)
                    frac = overlap / Le
                    F[idx] += frac * self._consistent_load_vec_udl(w_avg_local, Le)

        for (xp, P) in self.point_loads:
            j = int(np.argmin(np.abs(x_nodes - xp)))
            F[2 * j] += P

        for (xs, k) in self.spring_supports:
            j = int(np.argmin(np.abs(x_nodes - xs)))
            K[2 * j, 2 * j] += k

        bc = set()
        for (xs, kind) in self.supports:
            j = int(np.argmin(np.abs(x_nodes - xs)))
            bc.add(2 * j)
            if kind == 'fixed':
                bc.add(2 * j + 1)
        bc = sorted(bc)
        free = np.setdiff1d(np.arange(dof), bc)
        if free.size == dof:
            raise ValueError("Model is under-restrained: add at least one support/spring.")

        U = np.zeros(dof)
        U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        R = K @ U - F

        reactions = [(x_nodes[b // 2], R[b] / 1e3) for b in bc]  # kN

        v = U[0::2]
        th = U[1::2]

        xe, Me, Ve, we = [], [], [], []
        for e in range(n - 1):
            xa, xb = x_nodes[e], x_nodes[e + 1]
            Le = xb - xa
            v1, t1, v2, t2 = v[e], th[e], v[e + 1], th[e + 1]

            s = np.linspace(0, 1, 20)
            xloc = xa + s * Le

            N1 = 1 - 3 * s ** 2 + 2 * s ** 3
            N2 = Le * (s - 2 * s ** 2 + s ** 3)
            N3 = 3 * s ** 2 - 2 * s ** 3
            N4 = Le * (-s ** 2 + s ** 3)
            wvals = N1 * v1 + N2 * t1 + N3 * v2 + N4 * t2

            d2N1 = (-6 + 12 * s) / Le ** 2
            d2N2 = (-4 + 6 * s) / Le
            d2N3 = (6 - 12 * s) / Le ** 2
            d2N4 = (-2 + 6 * s) / Le
            curv = d2N1 * v1 + d2N2 * t1 + d2N3 * v2 + d2N4 * t2
            M = self.E * self.I * curv
            V = -np.gradient(M, xloc, edge_order=2)

            xe.append(xloc)
            Me.append(M)
            Ve.append(V)
            we.append(wvals)

        x_plot = np.concatenate(xe)

        return {
            "x_nodes": x_nodes,
            "deflection_mm": v * 1e3,
            "x": x_plot,
            "M_kNm": -np.concatenate(Me) / 1e3,
            "V_kN": np.concatenate(Ve) / 1e3,
            "w_mm": np.concatenate(we) * 1e3,
            "reactions_kN": reactions
        }

    def plot_FBD(self, results):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(10, 3))

        # Dense locators (effectively hide ticks; we turn ticks off below)
        ax.xaxis.get_major_locator().set_params(nbins=1000)
        ax.yaxis.get_major_locator().set_params(nbins=1000)

        fig.suptitle("FREE BODY DIAGRAM", fontsize=11, fontweight='bold', y=1.02)

        L = self.L
        tri_base_y = -0.2
        dim_line_y = -0.55
        tri_half_w = 0.02 * L

        # Clean axes
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Beam line + end labels
        ax.plot([0, L], [0, 0], color="black", linewidth=2)
        ax.text(0.0, -0.06, "0", ha="center", va="top", fontsize=9)
        ax.text(L, -0.06, f"{L:.2f}", ha="center", va="top", fontsize=9)

        # --- Support symbols ---
        def draw_pin(x):
            tri = mpatches.Polygon(
                [[x, 0.0], [x - tri_half_w, tri_base_y], [x + tri_half_w, tri_base_y]],
                closed=True, fill=False, edgecolor="black", linewidth=1.5
            )
            ax.add_patch(tri)
            return tri_base_y

        def draw_roller(x):
            tri = mpatches.Polygon(
                [[x, 0.0], [x - tri_half_w, tri_base_y], [x + tri_half_w, tri_base_y]],
                closed=True, fill=False, edgecolor="black", linewidth=1.5
            )
            ax.add_patch(tri)
            r = tri_half_w / 2.0
            y_c = tri_base_y - r
            circ_left = plt.Circle((x - r, y_c), r, fill=False, edgecolor="black", linewidth=1.3)
            circ_right = plt.Circle((x + r, y_c), r, fill=False, edgecolor="black", linewidth=1.3)
            ax.add_patch(circ_left); ax.add_patch(circ_right)
            return tri_base_y

        def draw_fixed(x):
            rect = plt.Rectangle(
                (x - 0.02 * L, tri_base_y), 0.04 * L, abs(tri_base_y),
                fill=True, alpha=0.4, edgecolor="black",
            )
            ax.add_patch(rect)
            return tri_base_y

        def draw_hinge(x):
            circ = plt.Circle((x, -0.02), 0.02, fill=False, color="black", linewidth=1.3)
            ax.add_patch(circ)
            ax.vlines(x, -0.05, tri_base_y, colors="black", linewidth=1.0)
            return tri_base_y

        support_bottom = {}
        for (x, kind) in self.supports:
            kind_lower = kind.lower()
            if kind_lower == "fixed":
                bottom = draw_fixed(x)
            elif kind_lower == "roller":
                bottom = draw_roller(x)
            elif kind_lower == "hinge":
                bottom = draw_hinge(x)
            else:
                bottom = draw_pin(x)
            support_bottom[float(x)] = bottom

        # ✅ Draw reaction arrows + values **below** supports from results["reactions_kN"]
        # results["reactions_kN"] = [(x_support, Ry_kN), ...]
        # --- FLIPPED VERSION: positive R draws arrow DOWN; negative R draws arrow UP ---
        if "reactions_kN" in results and results["reactions_kN"]:
            for x_supp, R in results["reactions_kN"]:
                bottom_y = support_bottom.get(float(x_supp), tri_base_y)

                # Arrow geometry below the support
                y_top = bottom_y - 0.1
                y_bot = bottom_y - 0.5

                if R >= 0:
                    # originally upward ⇒ now draw arrow DOWN
                    ax.annotate(
                        "",
                        xy=(x_supp, y_bot),
                        xytext=(x_supp, y_top),
                        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.6)
                    )
                else:
                    # originally downward ⇒ now draw arrow UP
                    ax.annotate(
                        "",
                        xy=(x_supp, y_top),
                        xytext=(x_supp, y_bot),
                        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.6)
                    )

                # Value label stays the same
                ax.text(
                    x_supp, bottom_y - 0.7, f"{abs(R):.3f} kN",
                    color="black", fontsize=9, ha="center", va="top", fontweight="bold",
                )

        # Distance dims between supports
        xs = sorted(x for x, _ in self.supports)
        xs_full = [0.0] + xs + [self.L]

        def bottom_at(x_val):
            for xs_ in support_bottom.keys():
                if abs(xs_ - x_val) < 1e-9:
                    return support_bottom[xs_]
            return tri_base_y

        for i in range(len(xs_full) - 1):
            a = xs_full[i]; b = xs_full[i + 1]; xm = 0.5 * (a + b)
            ax.annotate("", xy=(b, dim_line_y), xytext=(a, dim_line_y),
                        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2))
            ax.vlines(a, dim_line_y, bottom_at(a), colors="black", linewidth=1.0)
            ax.vlines(b, dim_line_y, bottom_at(b), colors="black", linewidth=1.0)
            ax.text(xm, dim_line_y - 0.045, f"{(b - a):.2f} m", ha="center", va="top", fontsize=9)

        ax.set_xlim(-0.05 * L, 1.05 * L)
        ax.set_ylim(-0.9, 0.65)

        # --- UDLs (blue) ---
        for tup in self.udls:
            if tup in self._hydro_udl_tuples:
                continue
            a, b, w1, w2 = tup
            y_ref = 0.35; y_tail = 0.05
            denom = max(abs(w1), abs(w2), 1.0)
            slope = 0.12 * np.tanh((w2 - w1) / denom)
            ax.plot([a, b], [y_ref, y_ref + slope], color="blue", linewidth=1.5)
            for x_i in np.linspace(a, b, 8):
                ax.annotate("", xy=(x_i, y_tail),
                            xytext=(x_i, y_ref + slope * (x_i - a) / max(b - a, 1e-9)),
                            arrowprops=dict(arrowstyle="->", color="blue", linewidth=1.5))
            ax.text((a + b) / 2, 0.48, f"{w1 / 1e3:.2f}→{w2 / 1e3:.2f} kN/m",
                    ha="center", va="bottom", fontsize=9, color="blue")

        # --- Hydro parts ---
        for hp in self._hydro_parts:
            start = hp["start"]; end = hp["end"]
            flat_a, flat_b = hp["flat"]; tri_a, tri_b, _ = hp["tri"]
            wmax_kNpm = hp["wmax_kNpm"]
            y_top = 0.35; y_tail = 0.05

            if flat_b - flat_a > 1e-12:
                ax.hlines(y_top, flat_a, flat_b, colors="blue", linewidth=1.8)
            if tri_b - tri_a > 1e-12:
                ax.plot([tri_a, tri_b], [y_top, y_tail], color="blue", linewidth=1.8)

            xs_ = np.linspace(start, end, 10)
            L_tri = max(1e-9, tri_b - tri_a)
            for x_i in xs_:
                if x_i <= tri_a:
                    y_head = y_top
                else:
                    lam = max(0.0, min(1.0, (tri_b - x_i) / L_tri))
                    y_head = y_tail + lam * (y_top - y_tail)
                ax.annotate("", xy=(x_i, y_tail), xytext=(x_i, y_head),
                            arrowprops=dict(arrowstyle="->", color="blue", linewidth=1.5))
            ax.text(0.5 * (start + end), 0.52, f"{wmax_kNpm:.2f} kN/m",
                    ha="center", va="bottom", fontsize=9, color="blue")

        # --- Point loads (red) ---
        for (xp, P) in self.point_loads:
            if P >= 0:
                ax.annotate("", xy=(xp, 0.06), xytext=(xp, 0.55),
                            arrowprops=dict(arrowstyle="->", color="red", linewidth=2))
                text_y = 0.58
            else:
                ax.annotate("", xy=(xp, 0.55), xytext=(xp, 0.06),
                            arrowprops=dict(arrowstyle="->", color="red", linewidth=2))
                text_y = 0.03
            ax.text(xp, text_y, f"{abs(P) / 1e3:.2f} kN",
                    ha="center", va="bottom", fontsize=9, color="red")

        plt.tight_layout()
        return fig

    def plot_SFD(self, results):
        fig, ax = plt.subplots(figsize=(10, 2.5))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axhline(0, color='grey', linewidth=1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=6, integer=True, steps=[1, 2, 5, 10])
        )
        ax.yaxis.set_major_formatter('{x:.0f}')
        ax.minorticks_off()
        ax.grid(True, which='major', axis='both', color='grey', linewidth=0.5, alpha=0.35)
        fig.suptitle("SHEAR FORCE DIAGRAM", fontsize=9, fontweight='bold', y=1.02)

        x, V = results["x"], results["V_kN"]
        if self.perm_shear is not None and np.isfinite(self.perm_shear):
            exceed_mask = np.abs(V) > float(self.perm_shear)
            V_safe = np.where(exceed_mask, np.nan, V)
            V_ex = np.where(exceed_mask, V, np.nan)
            ax.plot(x, V_safe, color="blue", linewidth=2)
            ax.fill_between(x, 0, V_safe, alpha=0.2, color="blue")
            ax.plot(x, V_ex, color="red", linewidth=2)
            ax.fill_between(x, 0, V_ex, alpha=0.25, color="red")
        else:
            ax.plot(x, V, color="blue", linewidth=2)
            ax.fill_between(x, 0, V, alpha=0.2, color="blue")

        ax.set_ylabel("Shear (kN) (upward +)")
        ax.set_xlabel("x (m)")
        self._annotate_extrema(ax, x, V, "kN", check=(self.perm_shear, self.grade))
        return fig

    def plot_BMD(self, results):
        fig, ax = plt.subplots(figsize=(10, 2.5))
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axhline(0, color='grey', linewidth=1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=6, integer=True, steps=[1, 2, 5, 10])
        )
        ax.yaxis.set_major_formatter('{x:.0f}')
        ax.minorticks_off()
        ax.grid(True, which='major', axis='both', color='grey', linewidth=0.5, alpha=0.35)
        fig.suptitle("BENDING MOMENT DIAGRAM", fontsize=9, fontweight='bold', y=1.02)

        x, M = results["x"], results["M_kNm"]
        if self.perm_moment is not None and np.isfinite(self.perm_moment):
            exceed_mask = np.abs(M) > float(self.perm_moment)
            M_safe = np.where(exceed_mask, np.nan, M)
            M_ex = np.where(exceed_mask, M, np.nan)
            ax.plot(x, M_safe, color="blue")
            ax.fill_between(x, 0, M_safe, alpha=0.2, color="blue")
            ax.plot(x, M_ex, color="red", linewidth=2)
            ax.fill_between(x, 0, M_ex, alpha=0.25, color="red")
        else:
            ax.plot(x, M, color="blue")
            ax.fill_between(x, 0, M, alpha=0.2, color="blue")

        ax.set_ylabel("Moment (kNm)")
        ax.set_xlabel("x (m)")
        self._annotate_extrema(ax, x, M, "kNm", check=(self.perm_moment, self.grade))
        ax.invert_yaxis()
        return fig

    def plot_deflection(self, results):
        """Plot deflection diagram with BMD-style axes (spines off, zero line, grid & ticks on)."""
        fig, ax = plt.subplots(figsize=(10, 2.5))

        # 1️⃣ Spines off (keep like BMD)
        for spine in ax.spines.values():
            spine.set_visible(False)

        # 2️⃣ Grid lines ON like BMD
        ax.axhline(0, color='grey', linewidth=1.0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, steps=[1, 2, 5, 10]))
        ax.yaxis.set_major_formatter('{x:.0f}')
        ax.minorticks_off()
        ax.grid(True, which='major', axis='both', color='grey', linewidth=0.5, alpha=0.35)

        # 3️⃣ Keep normal ticks/labels (so NO need to remove them)

        x, y = results["x"], results["w_mm"]
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        pad = 0.05 * max(1e-9, ymax - ymin)
        ymin -= pad
        ymax += pad
        ax.set_ylim(ymin, ymax)

        fig.suptitle("DEFLECTION", fontsize=9, fontweight='bold', y=1.02)
        ax.plot(x, y, color="blue", linewidth=2)
        ax.fill_between(x, 0, y, alpha=0.2, color="blue")

        ax.set_ylabel("Deflection (mm)")
        ax.set_xlabel("x (m)")
        self._annotate_extrema(ax, x, y, "mm")
        return fig

    def _annotate_extrema(self, ax, x, y, units, check=None):
        y_max = y[np.argmax(y)]
        x_max = x[np.argmax(y)]
        y_min = y[np.argmin(y)]
        x_min = x[np.argmin(y)]

        msg1 = f"Max = {y_max:.2f} {units} at x = {x_max:.2f} m"
        msg2 = f"Min = {y_min:.2f} {units} at x = {x_min:.2f} m"
    # temp commit check


        if check and check[0] is not None:
            limit, grade = check
            msg1 += f" | Permissible = {limit:.2f} {units} ({grade})"

        ax.text(0.5, -0.25, msg1, transform=ax.transAxes, ha="center", va="top")
        ax.text(0.5, -0.37, msg2, transform=ax.transAxes, ha="center", va="top")
