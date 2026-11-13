#!/usr/bin/env python3
# SPDX-License-Identifier: CC-BY-4.0
"""Exact symbolic verification of the manuscript's explicit examples.

Requires Python 3.10 or later and SymPy. Run:
    python verify_examples.py
    python verify_examples.py --json

The script checks displayed algebraic identities and counterexamples. It does
not prove a classification theorem, construct a quantum field theory, or
determine which source perturbations are physically admissible.
No numerical sampling, floating-point approximation, or network access is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from typing import Any

import sympy as sp


class VerificationError(AssertionError):
    """An expected exact identity did not reduce to zero."""


THETA, PHI, HALF_ANGLE = sp.symbols("theta phi half_angle", real=True)


def reduce_exact(expression: sp.Expr) -> sp.Expr:
    """Normalize rational, exponential, and half-angle trigonometric forms.

    Replacing theta by twice a fresh real variable is an invertible change of
    variable, not a numerical specialization. It makes half-angle identities
    explicit before simplification.
    """
    expression = sp.simplify(expression)
    expression = expression.subs(THETA, 2 * HALF_ANGLE)
    return sp.simplify(sp.trigsimp(sp.expand_trig(expression)))


def check_equal(actual: Any, expected: Any, description: str) -> None:
    """Require exact equality, component by component for matrices."""
    difference = actual - expected
    entries = list(difference) if isinstance(
        difference, sp.MatrixBase
    ) else [sp.sympify(difference)]
    residuals = [reduce_exact(entry) for entry in entries]
    if any(residual != 0 for residual in residuals):
        nonzero = [str(residual) for residual in residuals if residual != 0]
        raise VerificationError(f"{description}: residuals {nonzero}")


def hopf_data() -> tuple[sp.Matrix, sp.Matrix]:
    """Local unit frames of the tautological line over CP^1."""
    north = sp.Matrix([
        sp.cos(THETA / 2),
        sp.exp(sp.I * PHI) * sp.sin(THETA / 2),
    ])
    south = sp.Matrix([
        sp.exp(-sp.I * PHI) * sp.cos(THETA / 2),
        sp.sin(THETA / 2),
    ])
    return north, south


def verify_hopf_connection() -> dict[str, Any]:
    """Compute the Berry forms and oriented first Chern number."""
    north, south = hopf_data()
    check_equal(
        south, sp.exp(-sp.I * PHI) * north, "North/south transition"
    )
    check_equal((north.H * north)[0], 1, "North frame norm")
    check_equal((south.H * south)[0], 1, "South frame norm")

    def connection(frame: sp.Matrix) -> tuple[sp.Expr, sp.Expr]:
        # Convention: A = i u^dagger du and nabla = d - i A.
        return tuple(
            sp.simplify(sp.I * (frame.H * frame.diff(variable))[0])
            for variable in (THETA, PHI)
        )

    north_theta, north_phi = connection(north)
    south_theta, south_phi = connection(south)
    check_equal(north_theta, 0, "North polar connection component")
    check_equal(south_theta, 0, "South polar connection component")
    check_equal(
        north_phi, -sp.sin(THETA / 2) ** 2,
        "North azimuthal connection component",
    )
    check_equal(
        south_phi, sp.cos(THETA / 2) ** 2,
        "South azimuthal connection component",
    )
    check_equal(south_phi - north_phi, 1, "Transition A_S = A_N + dphi")

    curvature_north = sp.simplify(
        sp.diff(north_phi, THETA) - sp.diff(north_theta, PHI)
    )
    curvature_south = sp.simplify(
        sp.diff(south_phi, THETA) - sp.diff(south_theta, PHI)
    )
    check_equal(curvature_north, curvature_south, "Curvature patch agreement")
    check_equal(
        curvature_north, -sp.sin(THETA) / 2, "Berry curvature coefficient"
    )

    # Orientation: sin(theta) dtheta wedge dphi.
    # F/(2*pi) represents the real image of the integral class c_1.
    chern_number = sp.simplify(
        sp.integrate(
            curvature_north,
            (THETA, 0, sp.pi),
            (PHI, 0, 2 * sp.pi),
        ) / (2 * sp.pi)
    )
    check_equal(chern_number, -1, "Oriented first Chern number")
    return {
        "name": "Hopf connection and first Chern number",
        "north_A_phi": str(north_phi),
        "south_A_phi": str(south_phi),
        "F_theta_phi": str(curvature_north),
        "c1_integral": str(chern_number),
    }


def verify_projectors() -> dict[str, Any]:
    """Check the global phase-independent density projector."""
    north, south = hopf_data()
    rho_north, rho_south = north * north.H, south * south.H
    expected = sp.Rational(1, 2) * sp.Matrix([
        [1 + sp.cos(THETA), sp.sin(THETA) * sp.exp(-sp.I * PHI)],
        [sp.sin(THETA) * sp.exp(sp.I * PHI), 1 - sp.cos(THETA)],
    ])
    check_equal(rho_north, expected, "Bloch-projector formula")
    check_equal(rho_north, rho_south, "Projector patch agreement")
    check_equal(rho_north.H, rho_north, "Projector Hermiticity")
    check_equal(rho_north * rho_north, rho_north, "Projector idempotence")
    check_equal(sp.trace(rho_north), 1, "Projector trace")
    check_equal(rho_north.det(), 0, "Projector determinant")
    check_equal(
        rho_north.subs(THETA, 0), sp.diag(1, 0), "Projector at north pole"
    )
    check_equal(
        rho_north.subs(THETA, sp.pi), sp.diag(0, 1), "Projector at south pole"
    )
    phase = sp.symbols("chi", real=True)
    rephased = sp.exp(sp.I * phase) * north
    check_equal(
        rephased * rephased.H, rho_north, "Global phase independence"
    )
    return {
        "name": "Projectors and global phase independence",
        "rank": 1,
        "trace": 1,
        "patches_agree": True,
    }


def verify_pauli_cocycle() -> dict[str, Any]:
    """Check the multiplier on Z_2^2 and its tensor-square cancellation."""
    pauli_x = sp.Matrix([[0, 1], [1, 0]])
    pauli_z = sp.diag(1, -1)
    group = tuple(product(range(2), repeat=2))

    def add(g: tuple[int, int], h: tuple[int, int]) -> tuple[int, int]:
        return ((g[0] + h[0]) % 2, (g[1] + h[1]) % 2)

    def operator(g: tuple[int, int]) -> sp.Matrix:
        return pauli_x ** g[0] * pauli_z ** g[1]

    def multiplier(g: tuple[int, int], h: tuple[int, int]) -> sp.Integer:
        return sp.Integer(-1) ** (g[1] * h[0])

    for g, h in product(group, repeat=2):
        check_equal(
            operator(g) * operator(h),
            multiplier(g, h) * operator(add(g, h)),
            f"Projective product for {g}, {h}",
        )
        vg = sp.kronecker_product(operator(g), operator(g))
        vh = sp.kronecker_product(operator(h), operator(h))
        vgh = sp.kronecker_product(
            operator(add(g, h)), operator(add(g, h))
        )
        check_equal(vg * vh, vgh, f"Tensor-square product for {g}, {h}")

    for g, h, k in product(group, repeat=3):
        check_equal(
            multiplier(g, h) * multiplier(add(g, h), k),
            multiplier(h, k) * multiplier(g, add(h, k)),
            f"Cocycle identity for {g}, {h}, {k}",
        )

    generator_x, generator_z = (1, 0), (0, 1)
    commutator_phase = sp.simplify(
        multiplier(generator_x, generator_z)
        / multiplier(generator_z, generator_x)
    )
    check_equal(commutator_phase, -1, "Projective commutator phase")
    xx, zz = (
        sp.kronecker_product(pauli_x, pauli_x),
        sp.kronecker_product(pauli_z, pauli_z),
    )
    check_equal(xx * zz, zz * xx, "Commuting tensor-square generators")

    # Conjugation is an ordinary group action even before tensor squaring.
    r0, r1, real_part, imaginary_part = sp.symbols("r0 r1 x y", real=True)
    rho = sp.Matrix([
        [r0, real_part + sp.I * imaginary_part],
        [real_part - sp.I * imaginary_part, r1],
    ])
    for g, h in product(group, repeat=2):
        ug, uh, ugh = operator(g), operator(h), operator(add(g, h))
        check_equal(
            ug * (uh * rho * uh.H) * ug.H,
            ugh * rho * ugh.H,
            f"Density-operator composition for {g}, {h}",
        )
    return {
        "name": "Pauli cocycle and tensor-square cancellation",
        "group_order": len(group),
        "commutator_phase": str(commutator_phase),
        "tensor_square_multiplier": "1",
        "density_action_composes": True,
    }


def powered_weights(
    rho: sp.Matrix, projectors: tuple[sp.Matrix, ...]
) -> tuple[sp.Expr, ...]:
    """The counterexample rule q_i^2 / sum_j q_j^2."""
    weights = tuple(sp.simplify(sp.trace(rho * p)) for p in projectors)
    denominator = sum(weight ** 2 for weight in weights)
    return tuple(sp.simplify(weight ** 2 / denominator) for weight in weights)


def verify_powered_weights() -> dict[str, Any]:
    """Check non-Born weights and failure of measurement-context independence."""
    p0, p1 = sp.diag(1, 0), sp.diag(0, 1)
    rho = sp.diag(sp.Rational(1, 3), sp.Rational(2, 3))
    probabilities = powered_weights(rho, (p0, p1))
    check_equal(
        sp.Matrix(probabilities),
        sp.Matrix([sp.Rational(1, 5), sp.Rational(4, 5)]),
        "Non-Born powered weights",
    )
    check_equal(sum(probabilities), 1, "Powered-weight normalization")
    check_equal(
        sp.Matrix(powered_weights(p0, (p0, p1))), sp.Matrix([1, 0]),
        "Powered-weight certainty",
    )

    # Simultaneous unitary change of state and measurement preserves the rule.
    unitary = sp.Matrix([[1, 1], [1, -1]]) / sp.sqrt(2)
    check_equal(unitary * unitary.H, sp.eye(2), "Covariance-example unitarity")
    transformed = powered_weights(
        unitary * rho * unitary.H,
        (unitary * p0 * unitary.H, unitary * p1 * unitary.H),
    )
    check_equal(
        sp.Matrix(transformed), sp.Matrix(probabilities),
        "Simultaneous unitary covariance",
    )

    # Keep the first event fixed while coarsening the other two.
    basis = (sp.diag(1, 0, 0), sp.diag(0, 1, 0), sp.diag(0, 0, 1))
    state = sp.Matrix([1 / sp.sqrt(2), sp.Rational(1, 2), sp.Rational(1, 2)])
    rho_three = state * state.H
    fine = powered_weights(rho_three, basis)
    coarse = powered_weights(rho_three, (basis[0], basis[1] + basis[2]))
    check_equal(
        sp.Matrix(fine),
        sp.Matrix([sp.Rational(2, 3), sp.Rational(1, 6), sp.Rational(1, 6)]),
        "Fine-measurement probabilities",
    )
    check_equal(
        sp.Matrix(coarse),
        sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2)]),
        "Coarse-measurement probabilities",
    )
    check_equal(fine[0] - coarse[0], sp.Rational(1, 6), "Context dependence")

    # The failure also occurs between two maximal rank-one measurements.
    complement_rotation = sp.Matrix([
        [1, 0, 0],
        [0, 1 / sp.sqrt(2), 1 / sp.sqrt(2)],
        [0, 1 / sp.sqrt(2), -1 / sp.sqrt(2)],
    ])
    rotated_basis = tuple(
        complement_rotation * p * complement_rotation.H for p in basis
    )
    check_equal(rotated_basis[0], basis[0], "Unchanged first projector")
    rotated = powered_weights(rho_three, rotated_basis)
    check_equal(
        sp.Matrix(rotated),
        sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2), 0]),
        "Changed complementary rank-one basis",
    )
    return {
        "name": "Powered-weight counterexample",
        "q_1_3_2_3": [str(p) for p in probabilities],
        "fine_measurement": [str(p) for p in fine],
        "coarse_measurement": [str(p) for p in coarse],
        "rotated_complement": [str(p) for p in rotated],
        "same_event_different_probability": True,
    }


def verify_finite_jets() -> dict[str, Any]:
    """Verify local vanishing for an unspecified smooth multiplier h."""
    x, w, a, b = sp.symbols("x w a b", real=True)
    h = sp.Function("h")
    jet_order = 2
    perturbation = h(x, w) * (w - a) ** (jet_order + 1) * (
        w - b
    ) ** (jet_order + 1)

    # Compact support and source admissibility are separate assumptions.
    for boundary in (a, b):
        for normal_order in range(jet_order + 1):
            derivative = sp.diff(perturbation, w, normal_order)
            check_equal(
                derivative.subs(w, boundary), 0,
                f"Normal derivative {normal_order} at w={boundary}",
            )
            check_equal(
                sp.diff(derivative, x).subs(w, boundary), 0,
                f"Mixed derivative x,w^{normal_order} at w={boundary}",
            )

    # The next derivative need not vanish: this is not the zero perturbation.
    next_derivative = sp.diff(perturbation, w, jet_order + 1)
    expected_a = sp.factorial(jet_order + 1) * (a - b) ** (
        jet_order + 1
    ) * h(x, a)
    expected_b = sp.factorial(jet_order + 1) * (b - a) ** (
        jet_order + 1
    ) * h(x, b)
    check_equal(next_derivative.subs(w, a), expected_a, "First available jet at a")
    check_equal(next_derivative.subs(w, b), expected_b, "First available jet at b")
    explicit = w ** 3 * (w - 1) ** 3
    check_equal(sp.diff(explicit, w, 3).subs(w, 0), -6, "Nonzero next jet")
    return {
        "name": "Finite normal-jet vanishing",
        "normal_orders_vanishing": [0, 1, 2],
        "arbitrary_smooth_multiplier": True,
        "third_derivative_at_a": str(expected_a),
        "nonzero_example_third_derivative_at_0": "-6",
    }


def verify_positive_kernel() -> dict[str, Any]:
    """Check an exact symbolic two-mode, three-sample Gram identity."""
    energies = sp.symbols("E0:2", nonnegative=True)
    weights = sp.symbols("w0:2", nonnegative=True)
    times = sp.symbols("t0:3", nonnegative=True)
    real_coefficients = sp.symbols("x0:3", real=True)
    imaginary_coefficients = sp.symbols("y0:3", real=True)
    coefficients = sp.Matrix([
        x + sp.I * y for x, y in zip(real_coefficients, imaginary_coefficients)
    ])

    def correlator(time: sp.Expr) -> sp.Expr:
        return sum(
            weight * sp.exp(-energy * time)
            for energy, weight in zip(energies, weights)
        )

    kernel = sp.Matrix([
        [correlator(tk + tl) for tl in times] for tk in times
    ])
    evaluation = sp.Matrix([
        [sp.exp(-energy * time) for time in times] for energy in energies
    ])
    gram = evaluation.T * sp.diag(*weights) * evaluation
    check_equal(kernel, gram, "Reflection-kernel Gram factorization")
    left = sp.expand((coefficients.H * kernel * coefficients)[0])
    right = 0
    for j, weight in enumerate(weights):
        real_amplitude = sum(
            real_coefficients[l] * evaluation[j, l] for l in range(len(times))
        )
        imaginary_amplitude = sum(
            imaginary_coefficients[l] * evaluation[j, l] for l in range(len(times))
        )
        right += weight * (real_amplitude ** 2 + imaginary_amplitude ** 2)
    check_equal(left, sp.expand(right), "Positive-kernel quadratic-form identity")
    if right.is_nonnegative is not True:
        raise VerificationError("Sum of nonnegative weighted real squares")
    return {
        "name": "Positive reflection-kernel Gram identity",
        "spectral_modes": len(energies),
        "sample_times": len(times),
        "weighted_sum_of_real_squares_nonnegative": True,
        "assumptions": "E_j, w_j, t_k >= 0; coefficients arbitrary complex",
    }


def verify_finite_index() -> dict[str, Any]:
    """Check the index of diag(s,1), including its kernel jump at s=0."""
    parameter = sp.symbols("s", real=True, nonzero=True)
    regular = sp.diag(parameter, 1)
    singular = sp.diag(0, 1)
    check_equal(
        regular * sp.diag(1 / parameter, 1), sp.eye(2),
        "Inverse for nonzero parameter",
    )

    def dimensions(matrix: sp.Matrix) -> dict[str, int]:
        rank = int(matrix.rank())
        kernel_dimension = matrix.cols - rank
        cokernel_dimension = matrix.rows - rank
        check_equal(
            len(matrix.nullspace()), kernel_dimension,
            "Nullspace dimension",
        )
        check_equal(
            len(matrix.H.nullspace()), cokernel_dimension,
            "Adjoint nullspace and cokernel dimension",
        )
        return {
            "kernel": kernel_dimension,
            "cokernel": cokernel_dimension,
            "index": kernel_dimension - cokernel_dimension,
        }

    regular_dimensions = dimensions(regular)
    singular_dimensions = dimensions(singular)
    expected_regular = {"kernel": 0, "cokernel": 0, "index": 0}
    expected_singular = {"kernel": 1, "cokernel": 1, "index": 0}
    if regular_dimensions != expected_regular:
        raise VerificationError("Index data for s != 0")
    if singular_dimensions != expected_singular:
        raise VerificationError("Index data for s = 0")
    return {
        "name": "Finite-dimensional index and kernel jump",
        "s_nonzero": regular_dimensions,
        "s_zero": singular_dimensions,
    }


def verify_source_selector_profile() -> dict[str, Any]:
    """Check a weighted pullback instance and the source-conditioned qubit profile.

    The cube calculation instantiates the manuscript's declared scalar
    functional. It does not establish that a particular physical source
    equation admits this scalar field. Alpha is held constant while the
    selector angles vary, exactly as required for the fixed-source Chern
    calculation.
    """
    x1, x2, x3, w = sp.symbols("x1 x2 x3 w", real=True)
    beta, shift, alpha = sp.symbols("beta source_shift alpha", real=True)
    source_scalar = beta + x1 + 2 * x2 + 3 * x3 + w
    weight = 2 * x1
    cube = ((x1, 0, 1), (x2, 0, 1), (x3, 0, 1))
    check_equal(sp.integrate(weight, *cube), 1, "Cube weight normalization")
    pulled_back = source_scalar.subs(w, 0)
    source_alpha = sp.integrate(weight * pulled_back, *cube)
    check_equal(source_alpha, beta + sp.Rational(19, 6), "Weighted source functional")
    check_equal(
        sp.diff(source_alpha, beta), 1, "Explicit dependence on the source scalar"
    )
    # Under F(x,w)=(x,w+c), transform the scalar and immersion together.
    transformed_scalar = source_scalar.subs(w, w - shift)
    check_equal(
        transformed_scalar.subs(w, shift), pulled_back,
        "Pullback invariance under simultaneous source reparametrization",
    )

    north, south = hopf_data()
    unitary = sp.diag(sp.exp(-sp.I * alpha / 2), sp.exp(sp.I * alpha / 2))
    check_equal(unitary.H * unitary, sp.eye(2), "Source-rotation unitarity")
    projector = north * north.H
    rotated_north, rotated_south = unitary * north, unitary * south
    rotated_projector = unitary * projector * unitary.H
    check_equal(
        rotated_projector, projector.subs(PHI, PHI + alpha),
        "Source-rotation selector map phi -> phi + alpha",
    )
    check_equal((rotated_north.H * rotated_north)[0], 1, "Rotated north norm")
    check_equal((rotated_south.H * rotated_south)[0], 1, "Rotated south norm")
    check_equal(rotated_projector.H, rotated_projector, "Rotated projector Hermiticity")
    check_equal(
        rotated_projector * rotated_projector, rotated_projector,
        "Rotated projector idempotence",
    )
    check_equal(sp.trace(rotated_projector), 1, "Rotated projector trace")
    check_equal(
        rotated_south, sp.exp(-sp.I * PHI) * rotated_north,
        "Fixed-source eigenline transition",
    )
    for original, rotated, patch in (
        (north, rotated_north, "north"),
        (south, rotated_south, "south"),
    ):
        for angle in (THETA, PHI):
            check_equal(
                sp.I * (rotated.H * rotated.diff(angle))[0],
                sp.I * (original.H * original.diff(angle))[0],
                f"Fixed-source {patch} connection in {angle}",
            )
    a_theta = sp.simplify(sp.I * (rotated_north.H * rotated_north.diff(THETA))[0])
    a_phi = sp.simplify(sp.I * (rotated_north.H * rotated_north.diff(PHI))[0])
    curvature = sp.simplify(sp.diff(a_phi, THETA) - sp.diff(a_theta, PHI))
    chern_number = sp.simplify(
        sp.integrate(curvature, (THETA, 0, sp.pi), (PHI, 0, 2 * sp.pi))
        / (2 * sp.pi)
    )
    check_equal(chern_number, -1, "Fixed-source induced eigenline Chern number")
    return {
        "name": "Source functional and selector-indexed qubit profile",
        "cube_weight": "2*x1 on [0,1]^3",
        "cube_source_functional": str(source_alpha),
        "simultaneous_source_reparametrization_invariant": True,
        "selector_rotation": "phi -> phi + alpha(Xi)",
        "density_fibers_fixed": "D(C^2)",
        "fixed_source_eigenline_c1_integral": str(chern_number),
        "scope": "Explicit conditional scalar/readout realization; no source-field admissibility inferred",
    }


def verify_qubit_effect_transfer() -> dict[str, Any]:
    """Check the qubit transfer counterexample and a tomographic identification.

    The last calculation verifies that four calibrated effects distinguish
    Hermitian qubit descriptors. The general effect-functional representation
    theorem and its hypotheses are proved analytically in the manuscript.
    """
    identity = sp.eye(2)
    projector = sp.diag(1, 0)
    state = identity / 2
    fine = (projector, (identity - projector) / 2, (identity - projector) / 2)
    coarse = (projector, identity - projector)
    for effects, label in ((fine, "fine"), (coarse, "coarse")):
        check_equal(sum(effects, sp.zeros(2)), identity, f"{label} POVM normalization")
        for i, effect in enumerate(effects):
            check_equal(effect.H, effect, f"{label} effect {i} Hermiticity")
            if any(value.is_nonnegative is not True for value in effect.eigenvals()):
                raise VerificationError(f"{label} effect {i} positivity")
    fine_born = tuple(sp.trace(state * effect) for effect in fine)
    coarse_born = tuple(sp.trace(state * effect) for effect in coarse)
    fine_powered = powered_weights(state, fine)
    coarse_powered = powered_weights(state, coarse)
    check_equal(
        sp.Matrix(fine_born),
        sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(1, 4)]),
        "Qubit fine POVM Born weights",
    )
    check_equal(
        sp.Matrix(fine_powered),
        sp.Matrix([sp.Rational(2, 3), sp.Rational(1, 6), sp.Rational(1, 6)]),
        "Qubit fine POVM powered weights",
    )
    check_equal(
        sp.Matrix(coarse_powered),
        sp.Matrix([sp.Rational(1, 2), sp.Rational(1, 2)]),
        "Qubit coarse POVM powered weights",
    )
    check_equal(fine_born[0], coarse_born[0], "Born transfer for unchanged effect")
    check_equal(
        fine_powered[0] - coarse_powered[0], sp.Rational(1, 6),
        "Powered-rule failure across qubit measurement selectors",
    )

    pauli_x = sp.Matrix([[0, 1], [1, 0]])
    pauli_y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    pauli_z = sp.diag(1, -1)
    hermitian_basis = (identity, pauli_x, pauli_y, pauli_z)
    tomography_effects = (
        identity, (identity + pauli_x) / 2,
        (identity + pauli_y) / 2, (identity + pauli_z) / 2,
    )
    for i, effect in enumerate(tomography_effects):
        check_equal(effect.H, effect, f"Tomographic effect {i} Hermiticity")
        for value in effect.eigenvals():
            if value.is_nonnegative is not True or (1 - value).is_nonnegative is not True:
                raise VerificationError(f"Tomographic effect {i} belongs to [0,I]")
    coefficient_map = sp.Matrix([
        [sp.trace(effect * basis_element) for basis_element in hermitian_basis]
        for effect in tomography_effects
    ])
    check_equal(coefficient_map.det(), 2, "Tomographically complete effect span")
    coefficients = sp.Matrix(sp.symbols("d0 dx dy dz", real=True))
    difference = sum(
        (coefficient * basis_element for coefficient, basis_element
         in zip(coefficients, hermitian_basis)), sp.zeros(2)
    )
    observations = sp.Matrix([sp.trace(difference * effect) for effect in tomography_effects])
    check_equal(
        coefficient_map.inv() * observations, coefficients,
        "Exact identification of independently introduced Hermitian descriptors",
    )
    return {
        "name": "Qubit POVM transfer failure and tomographic identification",
        "fine_born": [str(value) for value in fine_born],
        "fine_powered": [str(value) for value in fine_powered],
        "coarse_powered": [str(value) for value in coarse_powered],
        "unchanged_effect_powered_probability_difference": "1/6",
        "tomographic_effects": ["I", "(I+sigma_x)/2", "(I+sigma_y)/2", "(I+sigma_z)/2"],
        "effect_span_determinant": "2",
        "scope": "Exact finite counterexample and descriptor identification; transfer theorem proved in text",
    }


def verify_harmonic_reconstruction() -> dict[str, Any]:
    """Check harmonic Cauchy traces and the fixed-order exponential bound.

    At fixed derivative orders, the trace norm is bounded by n^m/sinh(na).
    The exponential identity and the positive derivative of its denominator
    establish the displayed bound for integers n>=1. A generic symbolic
    logarithmic limit proves decay for every fixed finite m.
    """
    x1, x2, x3, w = sp.symbols("hx1 hx2 hx3 hw", real=True)
    a = sp.symbols("ha", positive=True)
    n = sp.symbols("hn", positive=True, integer=True)
    order = sp.symbols("hm", nonnegative=True, integer=True)
    field = sp.sinh(n * w) * sp.sin(n * x1) / sp.sinh(n * a)
    laplacian = sum(sp.diff(field, variable, 2) for variable in (x1, x2, x3, w))
    check_equal(laplacian, 0, "Four-dimensional harmonic field")
    check_equal(field.subs(w, 0), 0, "Harmonic Dirichlet trace")
    check_equal(
        sp.diff(field, w).subs(w, 0), n * sp.sin(n * x1) / sp.sinh(n * a),
        "Harmonic normal Cauchy trace",
    )
    check_equal(field.subs(w, a), sp.sin(n * x1), "Remote-boundary trace")
    check_equal(
        field.subs({w: a, x1: sp.pi / (2 * n)}), 1,
        "Remote-boundary uniform norm attained",
    )

    # Verify the general derivative recurrences, as well as representative
    # finite jets. The recurrences start at the exact order-zero formulas.
    tangential_template = n ** order * sp.sin(n * x1 + order * sp.pi / 2)
    check_equal(
        sp.diff(tangential_template, x1), tangential_template.subs(order, order + 1),
        "General tangential derivative recurrence",
    )
    normal_template = n ** order * (
        sp.exp(n * w) - (-1) ** order * sp.exp(-n * w)
    ) / 2
    check_equal(normal_template.subs(order, 0), sp.sinh(n * w), "Normal derivative base")
    check_equal(
        sp.diff(normal_template, w), normal_template.subs(order, order + 1),
        "General normal derivative recurrence",
    )
    for normal_order in range(6):
        for tangential_order in range(5):
            trace = sp.diff(field, w, normal_order, x1, tangential_order).subs(w, 0)
            expected = (
                n ** (normal_order + tangential_order)
                * (1 - (-1) ** normal_order)
                * sp.sin(n * x1 + tangential_order * sp.pi / 2)
                / (2 * sp.sinh(n * a))
            )
            check_equal(
                trace, expected,
                f"Mixed harmonic trace with normal/tangential orders {normal_order}/{tangential_order}",
            )

    check_equal(
        (1 / sp.sinh(n * a)).rewrite(sp.exp),
        2 * sp.exp(-n * a) / (1 - sp.exp(-2 * n * a)),
        "Exponential trace-denominator identity",
    )
    denominator = 1 - sp.exp(-2 * n * a)
    if sp.diff(denominator, n).is_positive is not True:
        raise VerificationError("Exponential denominator strictly increases for a,n>0")
    positive_denominator_at_one = 2 * sp.exp(-a) * sp.sinh(a)
    check_equal(
        positive_denominator_at_one.rewrite(sp.exp), 1 - sp.exp(-2 * a),
        "Positive factorization of the denominator lower bound",
    )
    if positive_denominator_at_one.is_positive is not True:
        raise VerificationError("Positive denominator lower bound for integer n>=1")
    logarithmic_limit = sp.limit(order * sp.log(n) - a * n, n, sp.oo)
    if logarithmic_limit != -sp.oo:
        raise VerificationError("Fixed-order polynomial times exponential tends to zero")
    # Therefore n^m/sinh(na) <= 2*n^m*exp(-na)/(1-exp(-2*a)) -> 0.
    return {
        "name": "Harmonic reconstruction and fixed-order trace instability",
        "laplacian": "0",
        "dirichlet_trace": "0",
        "normal_trace": "n*sin(n*x1)/sinh(n*a)",
        "remote_trace": "sin(n*x1)",
        "remote_uniform_norm": "1",
        "verified_mixed_jet_orders": {"normal": [0, 1, 2, 3, 4, 5], "tangential": [0, 1, 2, 3, 4]},
        "general_derivative_recurrences": True,
        "fixed_order_bound": "n^m/sinh(n*a) <= 2*n^m*exp(-n*a)/(1-exp(-2*a)), n>=1, a>0",
        "fixed_order_logarithmic_limit": str(logarithmic_limit),
        "fixed_order_trace_bound_limit": "0",
        "scope": "Declared periodic harmonic scalar class; w is geometric, not temporal",
    }


def verify_source_redundancy_and_nonlinear_readout() -> dict[str, Any]:
    """Check finite witnesses for exact redundancy and second-order readout.

    The rotation acts on a two-dimensional null plane and fixes the observed
    coordinate. The polynomial and circle calculations distinguish a zero
    first derivative, a nonzero Hessian, and exact constancy on a gauge orbit.
    These examples do not prove the general closed-operator theorem and do
    not derive a source action or a physical readout from the TCGS axioms.
    """
    angle = sp.symbols("redundancy_angle", real=True)
    rotation = sp.Matrix([
        [sp.cos(angle), -sp.sin(angle), 0],
        [sp.sin(angle), sp.cos(angle), 0],
        [0, 0, 1],
    ])
    derivative = sp.Matrix([[0, 0, 1]])
    null_projector = sp.diag(1, 1, 0)
    observable_projector = sp.diag(0, 0, 1)
    check_equal(rotation.T * rotation, sp.eye(3), "Source rotation is orthogonal")
    check_equal(derivative * rotation, derivative, "Exact readout redundancy L U = L")
    check_equal(
        derivative * (rotation - sp.eye(3)), sp.zeros(1, 3),
        "Range of U-I is annihilated by L",
    )
    check_equal(
        null_projector * (rotation - sp.eye(3)), rotation - sp.eye(3),
        "Range of U-I lies in the declared null plane",
    )
    check_equal(
        derivative.T * derivative, observable_projector,
        "Positive observability operator is the observed-coordinate projector",
    )
    check_equal(
        rotation * observable_projector, observable_projector,
        "Positive observability sector is fixed pointwise",
    )

    x, y, epsilon = sp.symbols("null_x null_y null_epsilon", real=True)
    vx, vy = sp.symbols("null_vx null_vy", real=True)
    coordinates = (x, y)
    origin = {x: 0, y: 0}
    f2 = x ** 2 + y ** 2
    f4 = f2 ** 2
    gradient2 = sp.Matrix([sp.diff(f2, variable) for variable in coordinates])
    gradient4 = sp.Matrix([sp.diff(f4, variable) for variable in coordinates])
    hessian2 = sp.hessian(f2, coordinates)
    hessian4 = sp.hessian(f4, coordinates)
    check_equal(gradient2.subs(origin), sp.zeros(2, 1), "Quadratic readout has zero first derivative at origin")
    check_equal(gradient4.subs(origin), sp.zeros(2, 1), "Quartic readout has zero first derivative at origin")
    check_equal(hessian2.subs(origin), 2 * sp.eye(2), "Quadratic readout Hessian is twice the identity")
    check_equal(hessian4.subs(origin), sp.zeros(2), "Quartic readout Hessian vanishes at origin")
    rotated = rotation[:2, :2] * sp.Matrix([x, y])
    substitution = {x: rotated[0], y: rotated[1]}
    check_equal(f2.subs(substitution, simultaneous=True), f2, "Quadratic readout is rotation invariant")
    check_equal(f4.subs(substitution, simultaneous=True), f4, "Quartic readout is rotation invariant")
    line_substitution = {x: epsilon * vx, y: epsilon * vy}
    check_equal(
        f2.subs(line_substitution, simultaneous=True), epsilon ** 2 * (vx ** 2 + vy ** 2),
        "Quadratic response along an arbitrary straight variation",
    )
    check_equal(
        f4.subs(line_substitution, simultaneous=True), epsilon ** 4 * (vx ** 2 + vy ** 2) ** 2,
        "Quartic response along an arbitrary straight variation",
    )

    circle = sp.Matrix([sp.cos(epsilon), sp.sin(epsilon)])
    velocity = circle.diff(epsilon).subs(epsilon, 0)
    acceleration = circle.diff(epsilon, 2).subs(epsilon, 0)
    base = {x: 1, y: 0}
    check_equal(velocity, sp.Matrix([0, 1]), "Circle-orbit velocity")
    check_equal(acceleration, sp.Matrix([-1, 0]), "Circle-orbit acceleration")
    gradient_at_base = gradient2.subs(base)
    hessian_at_base = hessian2.subs(base)
    hessian_term = (velocity.T * hessian_at_base * velocity)[0]
    acceleration_term = gradient_at_base.dot(acceleration)
    check_equal(gradient_at_base.dot(velocity), 0, "Gauge-orbit first derivative vanishes")
    check_equal(hessian_term, 2, "Circle Hessian term")
    check_equal(acceleration_term, -2, "Circle acceleration term")
    check_equal(hessian_term + acceleration_term, 0, "Exact second-order gauge-orbit cancellation")
    on_orbit = f2.subs({x: circle[0], y: circle[1]}, simultaneous=True)
    check_equal(on_orbit, 1, "Full readout is constant on the entire circle orbit")
    check_equal(sp.diff(on_orbit, epsilon, 2), 0, "Exact orbit second derivative")
    return {
        "name": "Exact source redundancy and nonlinear readout witnesses",
        "rotation_null_plane": "(x,y)",
        "observed_coordinate": "z",
        "L": [[0, 0, 1]],
        "L_U_equals_L": True,
        "range_U_minus_I_in_kernel_L": True,
        "positive_sector_fixed_pointwise": True,
        "origin_gradients": {"quadratic": [0, 0], "quartic": [0, 0]},
        "origin_hessians": {"quadratic": [[2, 0], [0, 2]], "quartic": [[0, 0], [0, 0]]},
        "circle_second_order_terms": {"hessian": "2", "acceleration": "-2", "sum": "0"},
        "scope": "Exact finite witnesses; general domain-sensitive operator and constitutive claims require the hypotheses and proofs in the text",
    }


def verify_bell_recovery_obstruction() -> dict[str, Any]:
    """Verify an exact noninjective quantum-readout witness.

    Orthogonal Bell projectors give the same marginal I/2. The obstruction to
    a common exact recovery is then single-valuedness: one input cannot be
    mapped to two distinct outputs. The calculation establishes this finite
    witness, not a general reconstruction or source-information theorem.
    """
    bell_plus = sp.Matrix([1, 0, 0, 1]) / sp.sqrt(2)
    bell_minus = sp.Matrix([1, 0, 0, -1]) / sp.sqrt(2)
    rho_plus = bell_plus * bell_plus.H
    rho_minus = bell_minus * bell_minus.H
    for name, vector, rho in (
        ("plus", bell_plus, rho_plus),
        ("minus", bell_minus, rho_minus),
    ):
        check_equal((vector.H * vector)[0], 1, f"Bell {name} norm")
        check_equal(rho.H, rho, f"Bell {name} projector is Hermitian")
        check_equal(rho * rho, rho, f"Bell {name} projector is idempotent")
        check_equal(sp.trace(rho), 1, f"Bell {name} projector has unit trace")
    check_equal((bell_plus.H * bell_minus)[0], 0, "Bell plus/minus vectors are orthogonal")
    check_equal(rho_plus * rho_minus, sp.zeros(4), "Bell projector supports are orthogonal")

    def partial_trace_second(rho: sp.Matrix) -> sp.Matrix:
        # Basis ordering: |00>, |01>, |10>, |11>.
        return sp.Matrix(2, 2, lambda i, j: sum(rho[2 * i + k, 2 * j + k] for k in range(2)))

    marginal_plus = partial_trace_second(rho_plus)
    marginal_minus = partial_trace_second(rho_minus)
    check_equal(marginal_plus, sp.eye(2) / 2, "Bell plus marginal is I/2")
    check_equal(marginal_minus, sp.eye(2) / 2, "Bell minus marginal is I/2")
    check_equal(marginal_plus, marginal_minus, "Bell marginals agree exactly")
    difference = rho_plus - rho_minus
    expected_difference = sp.Matrix([
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ])
    check_equal(difference, expected_difference, "Distinct Bell projectors retain opposite coherence")
    check_equal(sp.trace(difference.H * difference), 2, "Nonzero exact Hilbert-Schmidt separation")
    eigenvalue = sp.symbols("bell_eigenvalue")
    check_equal(
        difference.charpoly(eigenvalue).as_expr(), eigenvalue ** 2 * (eigenvalue ** 2 - 1),
        "Bell-state difference has eigenvalues +1,-1,0,0",
    )
    check_equal(partial_trace_second(difference), sp.zeros(2), "Partial trace erases this coherence difference")
    return {
        "name": "Bell-state obstruction to common exact recovery",
        "states": ["(|00>+|11>)/sqrt(2)", "(|00>-|11>)/sqrt(2)"],
        "inner_product": "0",
        "common_marginal": [["1/2", "0"], ["0", "1/2"]],
        "projector_difference_squared_Hilbert_Schmidt_norm": "2",
        "difference_eigenvalues": ["1", "-1", "0", "0"],
        "scope": "Exact finite noninjectivity witness; no single-valued recovery can return both projectors from their identical marginal",
    }


def verify_invisible_resolvent_sector() -> dict[str, Any]:
    """Check the direct-sum identity in a positive finite-dimensional fixture.

    A=diag(1,4), b=(1,1); its extension has hidden eigenvalue 9 and probe
    (1,1,0). All calculations use (A+q^2 I)^(-1), q>0. The fixture supports
    the block identity proved in the text; it does not establish the general
    infinite-dimensional inverse theorem or classify source geometries.
    """
    q = sp.symbols("response_q", positive=True)
    observed_operator = sp.diag(1, 4)
    observed_probe = sp.Matrix([1, 1])
    extended_operator = sp.diag(1, 4, 9)
    extended_probe = sp.Matrix([1, 1, 0])
    alternative_operator = sp.diag(1, 4, 16)
    check_equal(observed_operator.H, observed_operator, "Observed operator is Hermitian")
    check_equal(extended_operator.H, extended_operator, "Extended operator is Hermitian")
    for name, operator in (("observed", observed_operator), ("extended", extended_operator), ("alternative", alternative_operator)):
        if not all(operator[i, i].is_positive for i in range(operator.rows)):
            raise VerificationError(f"The {name} diagonal fixture must be positive")
    observed_resolvent = (observed_operator + q ** 2 * sp.eye(2)).inv()
    extended_resolvent = (extended_operator + q ** 2 * sp.eye(3)).inv()
    alternative_resolvent = (alternative_operator + q ** 2 * sp.eye(3)).inv()
    response = (observed_probe.H * observed_resolvent * observed_probe)[0]
    extended_response = (extended_probe.H * extended_resolvent * extended_probe)[0]
    alternative_response = (extended_probe.H * alternative_resolvent * extended_probe)[0]
    expected = 1 / (q ** 2 + 1) + 1 / (q ** 2 + 4)
    check_equal(response, expected, "Observed boundary resolvent response")
    check_equal(extended_response, expected, "Added invisible sector leaves boundary response unchanged")
    check_equal(alternative_response, expected, "Changing invisible eigenvalue leaves boundary response unchanged")
    check_equal(
        extended_resolvent,
        sp.diag(1 / (q ** 2 + 1), 1 / (q ** 2 + 4), 1 / (q ** 2 + 9)),
        "Exact direct-sum resolvent matrix",
    )
    check_equal(
        sp.trace(extended_resolvent) - sp.trace(observed_resolvent),
        1 / (q ** 2 + 9),
        "Full resolvent trace retains the hidden sector",
    )
    check_equal(
        sp.trace(extended_resolvent) - sp.trace(alternative_resolvent),
        7 / ((q ** 2 + 9) * (q ** 2 + 16)),
        "Same-dimensional hidden spectra have different full resolvent traces",
    )

    # An exact finite-rank illustration of which subspace is generated by the
    # probe. It is not a substitute for a cyclicity theorem in general.
    observed_krylov = sp.Matrix.hstack(observed_probe, observed_operator * observed_probe)
    extended_krylov = sp.Matrix.hstack(
        extended_probe, extended_operator * extended_probe,
        extended_operator ** 2 * extended_probe,
    )
    check_equal(observed_krylov.det(), 3, "Probe spans the two-dimensional observed sector")
    check_equal(extended_krylov.rank(), 2, "Extended probe-generated subspace has rank two")
    hidden_vector = sp.Matrix([0, 0, 1])
    check_equal(hidden_vector.H * extended_krylov, sp.zeros(1, 3), "Hidden eigenvector is orthogonal to the probe-generated sector")
    check_equal(extended_operator * hidden_vector, 9 * hidden_vector, "Invisible vector is an actual positive-eigenvalue mode")
    return {
        "name": "Invisible direct-sum sector in a compressed resolvent",
        "spectral_parameter": "q^2, q>0",
        "observed_operator": "diag(1,4)",
        "observed_probe": [1, 1],
        "extended_operators": ["diag(1,4,9)", "diag(1,4,16)"],
        "extended_probe": [1, 1, 0],
        "common_response": "1/(q^2+1)+1/(q^2+4)",
        "extended_minus_observed_full_trace": "1/(q^2+9)",
        "hidden_spectrum_full_trace_difference": "7/((q^2+9)*(q^2+16))",
        "probe_generated_ranks": {"observed": 2, "extended": 2},
        "ambient_dimensions": {"observed": 2, "extended": 3},
        "scope": "Exact positive finite fixture for the general block identity; the manuscript supplies the general operator argument",
    }


def verify_common_kernel() -> dict[str, Any]:
    """Check both amplitude rows, transfer, and positivity on 0 <= xi <= 1.

    xi = r**2/(1+r**2), r >= 0, covers exactly [0,1). Its inverse is
    r=sqrt(xi/(1-xi)) there. The endpoint xi=1 is checked separately.
    This avoids silently treating sqrt(1-xi) as real outside its domain.
    """
    xi = sp.symbols("xi_kernel", real=True)
    r = sp.symbols("r_kernel", nonnegative=True)
    theta = sp.symbols("theta_kernel", real=True)
    eta_symbol = sp.symbols("eta_accessibility", positive=True)
    xi_r = r**2 / (1 + r**2)
    a1 = sp.Matrix([
        sp.sqrt(xi), sp.sqrt(xi), sp.sqrt(1 - xi), 0,
    ]) / 2
    a2 = sp.Matrix([
        sp.sqrt(xi), sp.exp(sp.I * theta) * sp.sqrt(xi),
        0, sp.sqrt(1 - xi),
    ]) / 2
    a1_r = a1.subs(xi, xi_r).applyfunc(sp.simplify)
    a2_r = a2.subs(xi, xi_r).applyfunc(sp.simplify)
    denominator = 2 * sp.sqrt(1 + r**2)
    check_equal(
        a1_r, sp.Matrix([r, r, 1, 0]) / denominator,
        "First square-root amplitude row on the complete open-end domain",
    )
    check_equal(
        a2_r, sp.Matrix([r, sp.exp(sp.I * theta) * r, 0, 1]) / denominator,
        "Second square-root amplitude row and selector phase",
    )

    eta_r = (1 + xi_r) / 4
    check_equal((a1_r.H * a1_r)[0], eta_r, "First success norm")
    check_equal((a2_r.H * a2_r)[0], eta_r, "Second success norm")
    raw_overlap = (a2_r.H * a1_r)[0]
    check_equal(
        raw_overlap, xi_r * (1 + sp.exp(-sp.I * theta)) / 4,
        "Conjugate overlap and phase sign",
    )
    gamma_r = sp.simplify(raw_overlap / eta_r)
    expected_gamma = xi_r * (1 + sp.exp(-sp.I * theta)) / (1 + xi_r)
    check_equal(gamma_r, expected_gamma, "Normalized detector overlap")
    check_equal(
        gamma_r,
        (1 - 1 / (4 * eta_r)) * (1 + sp.exp(-sp.I * theta)),
        "Accessibility eliminates the source descriptor",
    )
    check_equal(4 * eta_r - 1, xi_r, "Inverse accessibility calibration")
    check_equal(
        4 * eta_r * gamma_r,
        (4 * eta_r - 1) * (1 + sp.exp(-sp.I * theta)),
        "Joint accessibility-coherence restriction",
    )
    gamma_xi = xi * (1 + sp.exp(-sp.I * theta)) / (1 + xi)
    check_equal(
        gamma_xi.subs(xi, 4 * eta_symbol - 1),
        (1 - 1 / (4 * eta_symbol)) * (1 + sp.exp(-sp.I * theta)),
        "Direct rational elimination of xi",
    )

    # Exact nonnegative factorizations certify the entire parameter domain.
    lower_margin = r**2 / (4 * (1 + r**2))
    upper_margin = 1 / (4 * (1 + r**2))
    failure_weight = (3 + 2 * r**2) / (4 * (1 + r**2))
    check_equal(eta_r - sp.Rational(1, 4), lower_margin, "Lower norm margin")
    check_equal(sp.Rational(1, 2) - eta_r, upper_margin, "Upper norm margin")
    check_equal(1 - eta_r, failure_weight, "Failure weight")
    for expression, description in (
        (xi_r, "Source descriptor nonnegativity"),
        (lower_margin, "Accessibility at least one quarter"),
    ):
        if expression.is_nonnegative is not True:
            raise VerificationError(description)
    for expression, description in (
        (1 / (1 + r**2), "Source descriptor strictly below one"),
        (eta_r, "Strictly positive success norm"),
        (upper_margin, "Accessibility below one half before endpoint"),
        (failure_weight, "Strictly positive failure weight"),
    ):
        if expression.is_positive is not True:
            raise VerificationError(description)

    # A normalized two-vector Gram matrix is positive iff |gamma| <= 1.
    abs_gamma_squared = sp.simplify(sp.expand_complex(sp.conjugate(gamma_r) * gamma_r))
    overlap_margin = (
        1 + 4 * r**2 + 4 * r**4 * sp.sin(theta / 2)**2
    ) / (1 + 2 * r**2)**2
    check_equal(
        1 - abs_gamma_squared, overlap_margin,
        "Nonnegative normalized Gram determinant",
    )
    if overlap_margin.is_positive is not True:
        raise VerificationError("Strict positive Gram determinant for xi<1")

    # Include the orthogonal path register: detector overlap does not mix
    # the two path columns in T^dagger T.
    e1, e2 = sp.Matrix([1, 0]), sp.Matrix([0, 1])
    success = sp.Matrix.hstack(
        sp.kronecker_product(e1, a1_r),
        sp.kronecker_product(e2, a2_r),
    )
    check_equal(success.H * success, eta_r * sp.eye(2), "Success instrument norm")
    completed = success.col_join(sp.sqrt(failure_weight) * sp.eye(2))
    check_equal(completed.H * completed, sp.eye(2), "Success/failure isometry")

    # xi=0 is already covered by r=0, but both endpoint values are exposed.
    for value, norm in ((sp.Integer(0), sp.Rational(1, 4)),
                        (sp.Integer(1), sp.Rational(1, 2))):
        endpoint_a1, endpoint_a2 = a1.subs(xi, value), a2.subs(xi, value)
        check_equal((endpoint_a1.H * endpoint_a1)[0], norm, f"First norm at xi={value}")
        check_equal((endpoint_a2.H * endpoint_a2)[0], norm, f"Second norm at xi={value}")
        endpoint_gamma = sp.simplify((endpoint_a2.H * endpoint_a1)[0] / norm)
        check_equal(endpoint_gamma, gamma_xi.subs(xi, value), f"Overlap at xi={value}")
        if value == 1:
            check_equal(
                1 - sp.conjugate(endpoint_gamma) * endpoint_gamma,
                sp.sin(theta / 2)**2,
                "Endpoint nonnegative normalized Gram determinant",
            )
            if (sp.sin(theta / 2)**2).is_nonnegative is not True:
                raise VerificationError("Gram positivity at xi=1")

    return {
        "name": "Conditional common-kernel accessibility and coherence",
        "success_norm": "eta_acc=(1+xi)/4",
        "raw_overlap": "xi*(1+exp(-I*theta))/4",
        "normalized_overlap": "xi*(1+exp(-I*theta))/(1+xi)",
        "eliminated_descriptor": "gamma=(1-1/(4*eta_acc))*(1+exp(-I*theta))",
        "joint_restriction": "4*eta_acc*gamma=(4*eta_acc-1)*(1+exp(-I*theta))",
        "domain": "0<=xi<=1; theta real",
        "accessibility_domain": "1/4<=eta_acc<=1/2",
        "complete_domain_certificate": "xi=r^2/(1+r^2), r>=0, plus xi=1 endpoint",
        "gram_positivity": "1-|gamma|^2=(1+4*r^2+4*r^4*sin(theta/2)^2)/(1+2*r^2)^2",
        "success_failure_isometry": True,
        "scope": "Conditional on the declared square-root lift, orthonormal detector basis and quantum instrument convention; no derivation of Hilbert structure or Born weights",
    }


def verify_source_conditioned_degree() -> dict[str, Any]:
    """Check the source-dependent sphere map and the full degree-sector formula.

    a is a selector-side functional of an existing scalar source field.
    The allowed representation domain excludes a = +/-1. The script checks
    the curvature identity and its primitive exactly and verifies the sector
    formula with parameterizations covering all three allowed intervals.
    Smoothness, homotopy invariance, and physical admissibility are addressed
    analytically in the paper, not inferred from finite tests.
    """
    a, u, alpha = sp.symbols("source_a source_u source_alpha", real=True)
    positive = sp.symbols("source_sector_parameter", positive=True)
    n = sp.Matrix([
        sp.sin(THETA) * sp.cos(PHI),
        sp.sin(THETA) * sp.sin(PHI),
        sp.cos(THETA),
    ])
    shifted = n + sp.Matrix([0, 0, a])
    radius_squared = 1 + a ** 2 + 2 * a * sp.cos(THETA)
    check_equal(shifted.dot(shifted), radius_squared, "Shifted-sphere norm")
    mapped = shifted / sp.sqrt(radius_squared)
    check_equal(mapped.dot(mapped), 1, "Normalized map takes values in S^2")
    expected_density = (
        sp.sin(THETA) * (1 + a * sp.cos(THETA))
        / radius_squared ** sp.Rational(3, 2)
    )
    actual_density = mapped.dot(mapped.diff(THETA).cross(mapped.diff(PHI)))
    check_equal(actual_density, expected_density, "Oriented pullback sphere-area density")
    pauli = (
        sp.Matrix([[0, 1], [1, 0]]),
        sp.Matrix([[0, -sp.I], [sp.I, 0]]),
        sp.diag(1, -1),
    )
    projector = (
        sp.eye(2) + sum((mapped[j] * pauli[j] for j in range(3)), sp.zeros(2))
    ) / 2
    check_equal(projector * projector, projector, "Source-built density projector idempotence")
    check_equal(sp.trace(projector), 1, "Source-built density projector trace")
    primitive = (u + a) / sp.sqrt(1 + a ** 2 + 2 * a * u)
    integrand = (1 + a * u) / (1 + a ** 2 + 2 * a * u) ** sp.Rational(3, 2)
    check_equal(sp.diff(primitive, u), integrand, "Exact Chern-integral primitive")
    # Factoring the endpoint quadratics exposes sqrt((a +/- 1)^2)=|a +/- 1|.
    chern_number = -(sp.factor(primitive.subs(u, 1))
                     - sp.factor(primitive.subs(u, -1))) / 2
    sectors = (
        ("-1<a<1", (positive - 1) / (positive + 1), -1),
        ("a>1", 1 + positive, 0),
        ("a<-1", -1 - positive, 0),
    )
    sector_results = {}
    for name, parameterization, expected in sectors:
        value = sp.simplify(chern_number.subs(a, parameterization))
        check_equal(value, expected, f"Chern number throughout {name}")
        sector_results[name] = str(value)
    check_equal(radius_squared.subs({a: 1, THETA: sp.pi}), 0,
                "Positive threshold singular at south selector pole")
    check_equal(radius_squared.subs({a: -1, THETA: 0}), 0,
                "Negative threshold singular at north selector pole")
    check_equal(mapped.subs(a, 0), n, "Zero source functional gives identity map")

    # Periodicity is a property of the original rotation-only profile.
    north, _ = hopf_data()
    original_projector = north * north.H
    unitary = sp.diag(sp.exp(-sp.I * alpha / 2), sp.exp(sp.I * alpha / 2))
    shifted_unitary = unitary.subs(alpha, alpha + 2 * sp.pi)
    check_equal(shifted_unitary, -unitary, "Spinor sign under a 2pi functional shift")
    check_equal(shifted_unitary * original_projector * shifted_unitary.H,
                unitary * original_projector * unitary.H,
                "Rotation-only density profile is 2pi-periodic")
    return {
        "name": "Source-conditioned sphere degree and rotation-profile periodicity",
        "normalized_sphere_map": "(n+a*e3)/sqrt(1+a^2+2*a*n3)",
        "excluded_functional_values": [-1, 1],
        "oriented_area_density": str(expected_density),
        "chern_integral_primitive": str(primitive),
        "chern_number_by_full_interval": sector_results,
        "source_dependent_projector_identity": True,
        "rotation_only_profile_period": "2*pi",
        "scope": "Source conditioning through a specified selector readout; degree constant on continuous admissible map families; no field admissibility or realization rule derived from A1--A4",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true",
        help="write a machine-readable result to standard output",
    )
    args = parser.parse_args()
    verifications = (
        verify_hopf_connection,
        verify_projectors,
        verify_pauli_cocycle,
        verify_powered_weights,
        verify_finite_jets,
        verify_positive_kernel,
        verify_finite_index,
        verify_source_selector_profile,
        verify_qubit_effect_transfer,
        verify_harmonic_reconstruction,
        verify_source_redundancy_and_nonlinear_readout,
        verify_bell_recovery_obstruction,
        verify_invisible_resolvent_sector,
        verify_common_kernel,
        verify_source_conditioned_degree,
    )
    results: list[dict[str, Any]] = []
    try:
        for verification in verifications:
            results.append(verification())
    except VerificationError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1

    report = {
        "status": "passed",
        "sympy_version": sp.__version__,
        "arithmetic": "exact symbolic",
        "results": results,
    }
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for result in results:
            print(f"PASS: {result['name']}")
        print(f"All explicit identities verified exactly with SymPy {sp.__version__}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

