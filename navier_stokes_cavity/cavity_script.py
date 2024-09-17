""" Script that approximates the incompressible Navier-Stokes equations 
using the FEniCS Framework v2019.1.0 for the lid-driven square cavity flow 

Author: Gabriel F. Barros - gabriel.barros@nacad.ufrj.br
"""

import random

from fenics import (
    MPI,
    CellDiameter,
    Constant,
    DirichletBC,
    File,
    FiniteElement,
    Function,
    FunctionSpace,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    SubDomain,
    TestFunctions,
    TrialFunction,
    UnitSquareMesh,
    UserExpression,
    VectorElement,
    action,
    derivative,
    div,
    dot,
    dx,
    grad,
    inner,
    nabla_grad,
    near,
    parameters,
    split,
    sqrt,
)


class InitialConditions(UserExpression):
    """
    Class responsible for setting a suitable initial velocity field.
    """

    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        """
        Defines the values for initial conditions.
        """
        values[0] = 0.0 + 0.02 * (0.5 - random.random())  # u, v
        values[1] = 0.0  # p

    def value_shape(self):
        """
        Returns the shape of the values list (u, v and p)
        """
        return (3,)


def cavity_boundary_condition(W):
    """Function that defines the boundary conditions for the problem.

    Args:
        W (FunctionSpace object): Object containing the defined function space in FEniCS.

    Returns:
        list: list containing the DirichletBC objects in FEniCS.
    """

    class Lid(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], 1.0)

    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (
                near(x[0], 0.0) or near(x[0], 1.0) or near(x[1], 0.0)
            )

    def zero(x):
        return near(x[0], 0.0) and near(x[1], 0.0)

    # Dirichlet value
    g1 = Constant((1.0, 0.0))  # u, v
    g2 = Constant((0.0, 0.0))  # u, v
    g3 = Constant(0.0)  # p

    # Boundary Conditions
    bc1 = DirichletBC(W.sub(0), g1, Lid())  # u, v
    bc2 = DirichletBC(W.sub(0), g2, Walls())  # u, v
    bc3 = DirichletBC(W.sub(1), g3, zero, "pointwise")  # p
    bcs = [bc2, bc1, bc3]
    return bcs


# FEniCS parameters for performance (no need to change)
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

# ========= Simulation parameters =========
SUPG = True  # Enables SUPG stabilization
Re = 100  # Reynolds number
N = 25  # Number of nodes per dimension
f = Constant((0.0, 0.0))  # No body forces
nu = Constant(1.0 / Re)  # 1/Re for simplification

# ========= Simulation setup =========
mesh = UnitSquareMesh(N, N)  # Mesh creation on a Unit Square
P1v = VectorElement(
    "Lagrange", mesh.ufl_cell(), 2
)  # Quadratic velocity for Taylor-Hood elements
parameters["form_compiler"]["quadrature_degree"] = 3  # Gauss quadrature points
P1 = FiniteElement(
    "Lagrange", mesh.ufl_cell(), 1
)  # Linear pressure for Taylor-Hood elements
TH = P1v * P1  # Definition of Taylor-Hood elements
W = FunctionSpace(mesh, TH)  # Creation of the FunctionSpace on mesh

vp = TrialFunction(W)  # Trial functions of the weak form
(v, p) = split(vp)  # Split of the coupled vector into velocity and pressure
(w, q) = TestFunctions(W)  # Test functions of the weak form
vp_ = Function(
    W
)  # Creation of the Function that will get the nodal values after system solving
v_, p_ = split(vp_)  # Split of the coupled vector into velocity and pressure
u_init = InitialConditions()  # Instantiation of the InitialConditions for velocity
vp_.interpolate(
    u_init
)  # Interpolation of the initial conditions into the Function vector.
bcs = cavity_boundary_condition(W)  # Definition of boundary conditions


# ========= Weak form of the equations =========
F = (
    inner(dot(v_, nabla_grad(v)), w)
    + nu * inner(grad(w), grad(v))
    - inner(p, div(w))
    + inner(q, div(v))
    - inner(f, w)
) * dx

# ========= SUPG stabilization =========
if SUPG:
    h = CellDiameter(mesh)  # Computation of cell diameter for elements in the mesh
    vnorm = sqrt(dot(v_, v_))  # Computation of velocity norms
    R = (
        dot(v_, nabla_grad(v)) - nu * div(grad(v)) + grad(p) - f
    )  # Definition of the residual
    tau = ((2.0 * vnorm / h) ** 2 + 9.0 * (4.0 * nu / (h * 2)) ** 2) ** (
        -0.5
    )  # Evaluation of tau
    F_supg = tau * inner(R, dot(v_, nabla_grad(w))) * dx  # SUPG term
    F += F_supg  # Updating the weak form

# ========= Problem definition and solver =========
F1 = action(F, vp_)  # Assembling the system
J = derivative(F1, vp_, vp)  # Jacobian of the matrix
problem = NonlinearVariationalProblem(F1, vp_, bcs, J)  # Setting the nonlinear system
solver = NonlinearVariationalSolver(problem)  # Setting the nonlinear solver
prm = solver.parameters  # Assigning the parameters
prm["nonlinear_solver"] = "newton"  # Newton solver
prm["newton_solver"]["absolute_tolerance"] = 1e-6
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["convergence_criterion"] = "incremental"
prm["newton_solver"]["maximum_iterations"] = 20
prm["newton_solver"]["relaxation_parameter"] = 0.9
prm["newton_solver"]["linear_solver"] = "direct"
# Prevents the code from breaking during nonconvergence
prm["newton_solver"]["error_on_nonconvergence"] = False
converged_flag, nIter = solver.solve()  # Solves the system

# ========= File output =========
v_, p_ = vp_.split(True)  # Splitting function
out_u = File("results/velocity.pvd", "compressed")  # Output file for velocity
out_p = File("results/pressure.pvd", "compressed")  # Output file for pressure
out_u << v_
out_p << p_
