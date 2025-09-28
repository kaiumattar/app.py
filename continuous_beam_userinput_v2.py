import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, NullLocator


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
        self.grade = grade  # Hydro (concrete pressure) visual parts

        self._hydro_parts = []
        self._hydro_udl_tuples = set()

    def add_support(self, x, kind="pin"):
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

        # 1) flat part
        if flat_end - x_start > 1e-12:
            t = self.add_udl(
                x_start,
                flat_end,
                w_max_kNpm * 1e3,  # to N/m
                w_max_kNpm * 1e3,
            )
            self._hydro_udl_tuples.add(t)

        # 2) triangular tail
        if x_end - flat_end > 1e-12:
            t = self.add_udl(flat_end, x_end, w_max_kNpm * 1e3, 0.0)
            self._hydro_udl_tuples.add(t)

        self._hydro_parts.append(
            {
                "start": x_start,
                "end": x_end,
                "Lh": Lh,
                "wmax_kNpm": w_max_kNpm,
                "flat": (x_start, flat_end),
                "tri": (flat_end, x_end, w_max_kNpm),
            }
        )
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
            [w * L / 2, w * L**2 / 12, w * L / 2, -w * L**2 / 12],
            dtype=float,
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

            ke = (self.E * self.I / Le**3) * np.array(
                [
                    [12, 6 * Le, -12, 6 * Le],
                    [6 * Le, 4 * Le**2, -6 * Le, 2 * Le**2],
                    [-12, -6 * Le, 12, -6 * Le],
                    [6 * Le, 2 * Le**2, -6 * Le, 4 * Le**2],
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

        # Boundary conditions
        bc = set()
        for (xs, kind) in self.supports:
            j = int(np.argmin(np.abs(x_nodes - xs)))
            bc.add(2 * j)
            if kind == "fixed":
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

            N1 = 1 - 3 * s**2 + 2 * s**3
            N2 = Le * (s - 2 * s**2 + s**3)
            N3 = 3 * s**2 - 2 * s**3
            N4 = Le * (-s**2 + s**3)

            wvals = N1 * v1 + N2 * t1 + N3 * v2 + N4 * t2

            d2N1 = (-6 + 12 * s) / Le**2
            d2N2 = (-4 + 6 * s) / Le
            d2N3 = (6 - 12 * s) / Le**2
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
            "reactions_kN": reactions,
        }

    def plot_deflection(self, results):
        """Plot deflection diagram with max 7 horizontal grid lines."""
        fig, ax = plt.subplots(figsize=(10, 2.5))

        # Hide spines
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Turn off default grid
        ax.grid(False)

        # Forcefully turn off minor ticks on y-axis
        ax.minorticks_off()
        ax.yaxis.set_minor_locator(NullLocator())

        # Tick marks off for x-axis only
        ax.tick_params(left=True, bottom=False)

        x, y = results["x"], results["w_mm"]
        ymin, ymax = float(np.nanmin(y)), float(np.nanmax(y))
        pad = 0.05 * max(1e-9, ymax - ymin)
        ymin -= pad
        ymax += pad
        ax.set_ylim(ymin, ymax)

        # ✅ Y-axis: max 7 major horizontal grid lines
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax.yaxis.set_major_formatter("{x:.1f}")
        ax.yaxis.grid(True, which="major")   # horizontal grid lines ON (major only)

        # ✅ Vertical grid lines
        ax.xaxis.grid(True, which="major")   # vertical grid lines ON (major only)

        fig.suptitle("DEFLECTION", fontsize=9, fontweight="bold", y=1.02)
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

        if check and check[0] is not None:
            limit, grade = check
            msg1 += f" | Permissible = {limit:.2f} {units} ({grade})"

        ax.text(0.5, -0.25, msg1, transform=ax.transAxes, ha="center", va="top")
        ax.text(0.5, -0.37, msg2, transform=ax.transAxes, ha="center", va="top")
