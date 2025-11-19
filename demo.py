#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Martha Elias
from __future__ import annotations


"""
CAUTION
Deterministic modeling is vulnerable to unnatural distortions and algorithmically triggered reactions. Independent safety and risk management strategies are essential.

DISCLAIMER (Research Only)
This repository contains a research prototype. It is provided for educational and research purposes only. It does NOT constitute financial, investment, legal, medical, or any other professional advice. No warranty is given. Use at your own risk. Before using any outputs to inform real-world decisions, obtain advice from qualified professionals and perform independent verification.

Copyright (c) 2025 Martha Elias
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.


I’d be happy if you like my work: https://buymeacoffee.com/marthafay
Author: Martha Elias
DOI: 10.5281/zenodo.17475177
Version: v1.0 (October 2025)
marthaelias [at] protonmail [dot] com
"""

"""
demo.py — Pseudoskalar (I) & „10 Ways to i“ — Visual + Numeric Checks
Research-only • Apache-2.0 • keine Gewähr

Panels (2×3):
  [1] Unit circle & Euler  e^{iθ}
  [2] Pseudoskalar/Rotator (J) acting on a vector (90° rotation)
  [3] exp(θJ) vs cosθ·I + sinθ·J (series vs closed form; max error)

  [4] Hilbert transform (90° phase shift)  cos → H{cos} ≈ sin
  [5] Power series of e^{ix} (truncated) vs cos/sin
  [6] Quaternions: pure units squared → −1 (scalar), zero vector part

CLI:
  python demo.py --plot
"""

import argparse
import math
import numpy as np

# Matplotlib is optional; we guard plotting.
HAS_MPL = True
try:
    import matplotlib.pyplot as plt
except Exception:
    HAS_MPL = False


# ---------------------------
# Pseudoskalar / Rotations-ops
# ---------------------------

def J_rot() -> np.ndarray:
    """J is the 90°-rotation operator in R^2: J^2 = -I."""
    return np.array([[0.0, -1.0],
                     [1.0,  0.0]], dtype=float)

def expm_series_thetaJ(theta: float, K: int = 40) -> np.ndarray:
    """
    exp(theta * J) via power series (no SciPy):
      exp(A) = sum_{k=0}^K A^k / k!
    Powers of J cycle every 4: J^0=I, J^1=J, J^2=-I, J^3=-J, J^4=I, ...
    """
    I = np.eye(2)
    J = J_rot()
    out = np.zeros((2, 2), float)
    fact = 1.0
    # We exploit the cycle to avoid repeated matmuls.
    for k in range(K + 1):
        if k == 0:
            Ak = I
        elif k % 4 == 1:
            Ak = J
        elif k % 4 == 2:
            Ak = -I
        elif k % 4 == 3:
            Ak = -J
        else:
            Ak = I
        if k > 0:
            fact *= k
        out += (theta ** k / fact) * Ak
    return out

def closed_form_thetaJ(theta: float) -> np.ndarray:
    """exp(θJ) = cosθ·I + sinθ·J."""
    I = np.eye(2)
    J = J_rot()
    return math.cos(theta) * I + math.sin(theta) * J


# ---------------------------
# Hilbert transform (FFT, no SciPy)
# ---------------------------

def hilbert_fft(x: np.ndarray) -> np.ndarray:
    """
    90°-Phasenverschiebung via FFT-basierter Hilbert-Transformation.
    Rückgabe: y = H{x} (Imaginärteil des analytischen Signals).
    """
    x = np.asarray(x, float)
    N = x.size
    X = np.fft.fft(x)

    h = np.zeros(N, dtype=float)
    if N % 2 == 0:
        # even: DC=1, Nyquist=1, positives doubled
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        # odd: DC=1, positives doubled
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0

    z = np.fft.ifft(X * h)  # analytic signal
    y = np.imag(z)
    return y.real


# ---------------------------
# Power series e^{ix}
# ---------------------------

def exp_ix_series(x: np.ndarray, K: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """
    Truncated series for e^{ix} = Σ (i x)^k / k!  → (Re≈cos, Im≈sin)
    Returns (Re_series, Im_series).
    """
    x = np.asarray(x, float)
    re = np.zeros_like(x, float)
    im = np.zeros_like(x, float)
    fact = 1.0
    i_pow_re, i_pow_im = 1.0, 0.0  # i^0 = 1 + 0i

    for k in range(K + 1):
        term_re = (i_pow_re * (x ** k)) / fact
        term_im = (i_pow_im * (x ** k)) / fact
        re += term_re
        im += term_im

        # Update (i)^(k+1) using multiplication by i → rotate (re,im)
        i_pow_re, i_pow_im = -i_pow_im, i_pow_re  # multiply by i
        fact *= (k + 1) if (k + 1) > 0 else 1.0

    return re, im


# ---------------------------
# Quaternion helpers
# ---------------------------

def q_mul(a: tuple[float, float, float, float],
          b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Hamilton product (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    w = aw*bw - ax*bx - ay*by - az*bz
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    return (w, x, y, z)

def q_square(u: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return q_mul(u, u)


# ---------------------------
# Plot figure
# ---------------------------

def build_figure():
    if not HAS_MPL:
        print("[warn] matplotlib not available; skipping plot.")
        return

    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axA, axB, axC, axD, axE, axF = axs.ravel()

    # [1] Unit circle & Euler e^{iθ}, mark θ=π/2
    th = np.linspace(0, 2*np.pi, 600)
    axA.plot(np.cos(th), np.sin(th), color="#333", lw=1.5)
    axA.scatter([0], [1], color="#d62728", s=40)
    axA.text(0.02, 0.02, r"$e^{i\pi/2} = i$", fontsize=10,
             transform=axA.transAxes, bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    axA.set_aspect("equal", "box")
    axA.set_xlim(-1.1, 1.1); axA.set_ylim(-1.1, 1.1)
    axA.set_title("Einheitskreis & Euler")

    # [2] Pseudoskalar/Rotation: v → Jv → J^2 v
    J = J_rot()
    v = np.array([1.0, 0.2])
    Jv = J @ v
    J2v = J @ Jv
    axB.arrow(0, 0, v[0], v[1], color="#1f77b4", width=0.01, length_includes_head=True)
    axB.arrow(0, 0, Jv[0], Jv[1], color="#ff7f0e", width=0.01, length_includes_head=True)
    axB.arrow(0, 0, J2v[0], J2v[1], color="#2ca02c", width=0.01, length_includes_head=True)
    axB.set_aspect("equal", "box")
    axB.set_xlim(-1.5, 1.5); axB.set_ylim(-1.5, 1.5)
    axB.set_title("Pseudoskalar als 90°-Rotation (J)")

    # [3] exp(θJ) series vs closed form → sup error vs θ
    thetas = np.linspace(0, np.pi, 128)
    errs = []
    for t in thetas:
        S = expm_series_thetaJ(t, K=24)
        C = closed_form_thetaJ(t)
        errs.append(np.max(np.abs(S - C)))
    axC.plot(thetas, errs, color="#1f77b4")
    axC.set_title(r"$\|e^{\theta J} - (\cos\theta\,I + \sin\theta\,J)\|_\infty$")
    axC.set_xlabel(r"$\theta$ (rad)")
    axC.set_ylabel("Fehler")

    # [4] Hilbert transform: cos -> ~sin
    t = np.linspace(0, 1.0, 1000, endpoint=False)
    x = np.cos(2*np.pi*5*t)  # 5 Hz cosine
    y = hilbert_fft(x)       # quadrature
    corr = np.corrcoef(y, np.sin(2*np.pi*5*t))[0, 1]
    axD.plot(t, x, label="cos", color="#1f77b4")
    axD.plot(t, y, label="Hilbert{cos} ≈ sin", color="#ff7f0e", alpha=0.9)
    axD.set_title(f"Hilbert-Transformation (90°-Phasenverschiebung)\n"
                  f"corr(H{{cos}}, sin) ≈ {corr:+.3f}")
    axD.legend(loc="upper right")

    # [5] Power series e^{ix} vs cos/sin
    xs = np.linspace(0, 2*np.pi, 400)
    re_ser, im_ser = exp_ix_series(xs, K=8)
    axE.plot(xs, re_ser, "--", label="Re series", color="#1f77b4")
    axE.plot(xs, np.cos(xs), "-", label="cos x", color="#1f77b4", alpha=0.4)
    axE.plot(xs, im_ser, "--", label="Im series", color="#d62728")
    axE.plot(xs, np.sin(xs), "-", label="sin x", color="#d62728", alpha=0.4)
    axE.set_title(r"Potenzreihe $e^{ix}$, $K=8$")
    axE.legend()

    # [6] Quaternions: i^2, j^2, k^2  → −1 (scalar part)
    q_i = (0.0, 1.0, 0.0, 0.0)
    q_j = (0.0, 0.0, 1.0, 0.0)
    q_k = (0.0, 0.0, 0.0, 1.0)
    i2 = q_square(q_i); j2 = q_square(q_j); k2 = q_square(q_k)
    scalars = [i2[0], j2[0], k2[0]]
    axF.bar(["i²", "j²", "k²"], scalars, color=["#9467bd", "#8c564b", "#2ca02c"])
    axF.set_ylim(-1.2, 0.2)
    axF.set_title("Quaternionen: jede reine Einheit hat Quadrat = −1")

    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.show()


# ---------------------------
# Console sanity checks
# ---------------------------

def print_checks():
    # I ≙ J acting on v
    J = J_rot()
    v = np.array([1.0, 0.0])
    Jv = J @ v
    J2v = J @ Jv
    print("\n== Pseudoskalar (I ≙ 90°-Rotator) ==")
    print({
        "v_x": v[0], "v_y": v[1],
        "Jv_x": Jv[0], "Jv_y": Jv[1],
        "J2v_x": J2v[0], "J2v_y": J2v[1],
    })

    # 1) Algebraic “i”
    i2 = -1.0
    z = 5 + 2j
    print("\n== 1) Algebraische Erweiterung ==")
    print({"i^2": i2, "i^2_b": (1j**2).real, "z_abs2": (z.real**2 + z.imag**2)})

    # 2) J^2
    J2 = J @ J
    print("\n== 2) 90°-Rotationsmatrix ==")
    print({"J2_00": J2[0, 0], "J2_11": J2[1, 1], "J2_01": J2[0, 1], "J2_10": J2[1, 0]})

    # 3) exp(θJ) series vs closed
    theta = math.pi/2
    S = expm_series_thetaJ(theta, K=30)
    C = closed_form_thetaJ(theta)
    err = np.max(np.abs(S - C))
    print("\n== 3) Matrix-Exponential ==")
    print({"||exp(pi/2 J) - J||_max": float(err)})

    # 4) Eigenvalues of rotation R(π/2)
    R = closed_form_thetaJ(theta)
    eig = np.linalg.eigvals(R)
    print("\n== 4) Eigenwerte R(π/2) ==")
    print({"eig1": eig[0], "eig2": eig[1]})

    # 5) Euler
    print("\n== 5) Euler auf Unit Circle ==")
    print({"e^{i*pi/2}": np.exp(1j * math.pi/2)})

    # 6) ODE roots (u''+u=0 → z=±i)
    print("\n== 6) ODE u''+u=0 ==")
    print({"root1_im": +1.0, "root2_im": -1.0})

    # 7) Power series quality at x=1
    re_ser, im_ser = exp_ix_series(np.array([1.0]), K=12)
    err_series = abs((re_ser[0] + 1j*im_ser[0]) - np.exp(1j*1.0))
    print("\n== 7) Potenzreihe e^{ix} ==")
    print({"series_err_at_x": 1.0, "K": 12, "|approx-exact|": float(err_series)})

    # 8) GA feeling: I^2=-1 via J
    I2 = J @ J
    print("\n== 8) GA-Feeling (I^2=-1) ==")
    val = np.array([0.6, -1.1])
    Jval = J @ val
    print({"I2_00": I2[0, 0], "I2_11": I2[1, 1], "|v|": float(np.linalg.norm(val)),
           "|Iv|": float(np.linalg.norm(Jval))})

    # 9) Hilbert transform correlation
    t = np.linspace(0, 1.0, 2000, endpoint=False)
    x = np.cos(2*np.pi*7*t)
    y = hilbert_fft(x)
    corr = float(np.corrcoef(y, np.sin(2*np.pi*7*t))[0, 1])
    print("\n== 9) Hilbert-Transform ==")
    print({"corr(H{cos}, sin)": corr})

    # 10) Quaternions squared
    i2q = q_square((0.0, 1.0, 0.0, 0.0))
    print("\n== 10) Quaternionen ==")
    print({"q_i_sq_w": i2q[0], "q_i_sq_xyz_norm": float(np.linalg.norm(i2q[1:]))})


# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Pseudoskalar & 10 Ways to i — Demo")
    ap.add_argument("--plot", action="store_true", help="Render figure")
    args = ap.parse_args()

    print_checks()
    if args.plot:
        build_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())