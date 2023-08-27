""" Script that approximates the convection-diffusion equation 
using the FEniCS Framework v2019.1.0 for the unit velocity problem.

Author: Gabriel F. Barros - gabriel.barros@nacad.ufrj.br
"""

import numpy as np
from fenics import (
    DOLFIN_EPS,
    CellDiameter,
    Constant,
    DirichletBC,
    File,
    Function,
    FunctionSpace,
    TestFunction,
    TrialFunction,
    UnitSquareMesh,
    VectorElement,
    as_vector,
    div,
    dot,
    dx,
    grad,
    lhs,
    rhs,
    solve,
    sqrt,
)


class Boundaries:
    """
    Definition of the boundary conditions for the Unit Square.
    """
    def left(x, on_boundary):
        return x[0] < DOLFIN_EPS and on_boundary

    def top(x, on_boundary):
        return x[1] > 1.0 - DOLFIN_EPS and on_boundary

    def right(x, on_boundary):
        return x[0] > 1.0 - DOLFIN_EPS and on_boundary

    def bottom(x, on_boundary):
        return x[1] < DOLFIN_EPS and on_boundary


# ================= Parameters ===================
N = 10
Pe = 10
SUPG = True


# ================= Problem Definition ===================
mesh = UnitSquareMesh(N - 1, N - 1)
h = CellDiameter(mesh)
Q = FunctionSpace(mesh, "CG", 1)
dphi, v = TrialFunction(Q), TestFunction(Q)
phi = Function(Q)
V1 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, V1)
velocity = Function(V)
f = Constant(1.0)

# ================= Parameters ===========================
if Pe == 0:
    k = Constant(1.0)
    velocity = Constant((0.0, 0.0))
else:
    eps = np.sqrt(2.0) / (2 * Pe)
    k = Constant(eps)
    velocity = as_vector([1.0, 1.0])

# ================= Boundary Conditions===================
bc_value = Constant(0.0)
bc_left = DirichletBC(Q, bc_value, Boundaries.left)
bc_top = DirichletBC(Q, bc_value, Boundaries.top)
bc_right = DirichletBC(Q, bc_value, Boundaries.right)
bc_bottom = DirichletBC(Q, bc_value, Boundaries.bottom)
bcs = [bc_left, bc_top, bc_right, bc_bottom]

# ================= Variational Form ====================
F = v * dot(velocity, grad(dphi)) * dx + k * dot(grad(v), grad(dphi)) * dx - f * v * dx

if SUPG:
    r = dot(velocity, grad(dphi)) - k * div(grad(dphi)) - f
    vnorm = sqrt(dot(velocity, velocity))
    delta = h / (2.0 * vnorm)
    F += delta * dot(velocity, grad(v)) * r * dx


# ====================== Solver =========================
a = lhs(F)
L = rhs(F)
solve(a == L, phi, bcs)
d = np.sqrt(2 * (1.0 / N))
print("Local Pe: ", Pe * d)

# ========= File output =========
out_phi = File("results/phi.pvd", "compressed")  # Output file for velocity
out_phi << phi
