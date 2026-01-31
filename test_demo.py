#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Martha Elias

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

Version: v1.0 (October 2025)
I want to get hired! Contact: marthaelias [at] protonmail [dot] com
"""
# test_demo.py
import importlib
import numpy as np

# ------------------------------------------------------------
# Helpers (nur für Tests, unabhängig von demo.py implementiert)
# ------------------------------------------------------------

def rot90_matrix():
    """J: 90°-Rotator (Pseudoskalar-Wirkung in 2D)."""
    return np.array([[0.0, -1.0],
                     [1.0,  0.0]])

def exp_theta_J(theta):
    """exp(θ J) = cosθ I + sinθ J  (geschlossene Form)."""
    I = np.eye(2)
    J = rot90_matrix()
    return np.cos(theta)*I + np.sin(theta)*J

def hilbert_discrete(x):
    """
    Diskrete Hilbert-Transform via FFT (ohne SciPy).
    Liefert y = H{x}. Quelle: übliche Analytic-Signal-Maske.
    """
    N = len(x)
    X = np.fft.fft(x)
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1.0
        H[N//2] = 1.0
        H[1:N//2] = 2.0
    else:
        H[0] = 1.0
        H[1:(N+1)//2] = 2.0
    z = np.fft.ifft(X * H)
    return np.imag(z)

def exp_ix_series(x, K=16):
    """Teilsumme der Potenzreihe e^{ix} bis Ordnung K-1."""
    x = np.asarray(x, float)
    s = np.zeros_like(x, dtype=np.complex128)
    term = np.ones_like(x, dtype=np.complex128)
    for k in range(K):
        s += term
        term *= (1j * x) / (k + 1)
    return s

def qmul(a, b):
    """Quaternionprodukt. a,b als (w,x,y,z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw
    ], dtype=float)


# ------------------------------------------------------------
# Smoke: Modul lädt
# ------------------------------------------------------------

def test_demo_module_imports():
    mod = importlib.import_module("demo")  # nur verifizieren, dass's importierbar ist
    assert mod is not None


# ------------------------------------------------------------
# 1/2) Pseudoskalar/Rotation
# ------------------------------------------------------------

def test_J_squared_is_minus_I():
    J = rot90_matrix()
    J2 = J @ J
    assert np.allclose(J2, -np.eye(2), atol=1e-12)

def test_J_rotation_norm_preserved_and_J2v_eq_minus_v():
    rng = np.random.default_rng(7)
    for _ in range(5):
        v = rng.standard_normal(2)
        J = rot90_matrix()
        Jv = J @ v
        assert np.allclose(np.linalg.norm(Jv), np.linalg.norm(v), atol=1e-12)
        assert np.allclose(J @ (J @ v), -v, atol=1e-12)

def test_expm_equals_closed_form_at_pi_over_2():
    # exp(π/2 J) = J
    J = rot90_matrix()
    A = exp_theta_J(np.pi/2.0)
    assert np.allclose(A, J, atol=1e-12)


# ------------------------------------------------------------
# 5) Euler: e^{iθ} = cosθ + i sinθ
# ------------------------------------------------------------

def test_euler_identity_uniform_grid():
    theta = np.linspace(0, 2*np.pi, 721)  # feines Gitter
    lhs = np.exp(1j * theta)
    rhs = np.cos(theta) + 1j*np.sin(theta)
    err = np.max(np.abs(lhs - rhs))
    assert err < 1e-12


# ------------------------------------------------------------
# 6) ODE u'' + u = 0  ⇒  z^2 + 1 = 0
# ------------------------------------------------------------

def test_ode_roots_are_pm_i():
    # Charakteristisches Polynom s^2 + 1
    coeff = [1.0, 0.0, 1.0]
    r = np.roots(coeff)
    # sortiere nach Imaginärteil
    r = r[np.argsort(r.imag)]
    assert np.allclose(r[0].real, 0.0, atol=1e-12)
    assert np.allclose(r[1].real, 0.0, atol=1e-12)
    assert np.allclose(r[0].imag, -1.0, atol=1e-12)
    assert np.allclose(r[1].imag, +1.0, atol=1e-12)


# ------------------------------------------------------------
# 7) Potenzreihe e^{ix}
# ------------------------------------------------------------

def test_series_exp_ix_accuracy():
    x = 1.2345
    K = 20
    approx = exp_ix_series(x, K=K)
    exact = np.exp(1j*x)
    assert np.allclose(approx, exact, atol=1e-10)


# ------------------------------------------------------------
# 9) Hilbert-Transform: H{cos} ≈ sin
# ------------------------------------------------------------

def test_hilbert_cos_to_sin_shift():
    N = 4096
    t = np.arange(N) / N
    # mehrere Frequenzen (Bandbegrenzung hilft der Diskretisierung)
    f = 7.0
    x = np.cos(2*np.pi*f*t)
    y = hilbert_discrete(x)
    # Korrelation mit +sin
    corr = np.corrcoef(y, np.sin(2*np.pi*f*t))[0,1]
    assert corr > 0.995


# ------------------------------------------------------------
# 10) Quaternionen: i^2 = j^2 = k^2 = -1
# ------------------------------------------------------------

def test_quaternion_pure_units_square_to_minus_one():
    one = np.array([1.0, 0.0, 0.0, 0.0])
    qi  = np.array([0.0, 1.0, 0.0, 0.0])
    qj  = np.array([0.0, 0.0, 1.0, 0.0])
    qk  = np.array([0.0, 0.0, 0.0, 1.0])

    for q in (qi, qj, qk):
        q2 = qmul(q, q)
        # Erwartet: (-1, 0, 0, 0)
        assert np.allclose(q2[0], -1.0, atol=1e-12)
        assert np.allclose(q2[1:], [0.0, 0.0, 0.0], atol=1e-12)
