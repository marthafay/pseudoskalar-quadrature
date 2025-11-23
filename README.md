# pseudoskalar-quadrature
Pseudoscalar Quadrature Representation for Real-Valued Signals

Instead of modeling the imaginary part of a “complex” signal using the pure number i (with i² = −1 implicitly), we propose using the pseudoscalar I of planar geometry. In 2D geometric algebra, I² = −1 arises from oriented area rather than from numerical definition. Thus, the “imaginary part” becomes an oriented geometric quantity rather than an abstract scalar.

 

We construct a window-based measure

H = H⋆ + I κP†

from a normalized entropy H⋆ = Hreal / log K ∈ [0,1] and a pseudoscalar phase term P† ∈ [−1,1] that combines oddness, chirality, and statistical evidence.

Decisions are made on the polar mapping (H⋆, κP†) → (r, θ) instead of on an unscaled linear sum. This approach replaces the numeric imaginary unit by a geometric one, giving i a physical interpretation as an oriented area element.

 

A companion script (demo.py) demonstrates the ten realizations of i across algebra, rotation, differential equations, and geometric algebra, and verifies equivalence by numeric and visual checks.

 

Tested with Python 3.11 and NumPy ≥ 1.26. License: Apache-2.0, research-only, no warranty.

