from fenics import *
import numpy as np
import random

class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)

    def eval(self, values, x):
        values[0] = 0.0 + 0.02 * (0.5 - random.random())
        values[1] = 0.0

    def value_shape(self):
        return (3,)


def cavity_boundary_condition(W):
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
    g1 = Expression(("exp(-(pow(x[0] - 0.5, 2)/(2*pow(0.18,2))))", 0.0), degree = 2)
    # g1 = Constant((1.0, 0.0))
    g2 = Constant((0.0, 0.0))
    g3 = Constant(0.0)
    # conditions
    bc1 = DirichletBC(W.sub(0), g1, Lid())
    bc2 = DirichletBC(W.sub(0), g2, Walls())
    bc3 = DirichletBC(W.sub(1), g3, zero, "pointwise")
    bcs = [bc1, bc2, bc3]
    return bcs

out_u = File("results/velocity.pvd", "compressed")
out_p = File("results/pressure.pvd", "compressed")


parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True

SUPG = True
Fr = 1.0
N = 20
Froude_inv = Constant(1.0 / pow(Fr, 2))
f = Froude_inv * Constant((0.0, 0.0))
nu = Constant(100.0)




mesh = UnitSquareMesh(N, N)
P1v = VectorElement("Lagrange", mesh.ufl_cell(), 2)
parameters["form_compiler"]["quadrature_degree"] = 3
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P1v * P1
W = FunctionSpace(mesh, TH)
vp = TrialFunction(W)
(v, p) = split(vp)
(w, q) = TestFunctions(W)
vp_ = Function(W)
v_, p_ = split(vp_)
u_init = InitialConditions()
vp_.interpolate(u_init)
vn = Function(W.sub(0).collapse())
vnn = Function(W.sub(0).collapse())
bcs = cavity_boundary_condition(W)

F = (
    inner(dot(v_, nabla_grad(v)), w)
    + nu * inner(grad(w), grad(v))
    - inner(p, div(w))
    + inner(q, div(v))
    - inner(f, w)
) * dx

if SUPG:
    h = CellDiameter(mesh)
    vnorm = sqrt(dot(v_, v_))
    R = dot(v_, nabla_grad(v)) - nu * div(grad(v)) + grad(p) - f
    tau = ((2.0 * vnorm / h) ** 2 + 9.0 * (4.0 * nu / (h * 2)) ** 2) ** (-0.5)
    vnorm = sqrt(dot(v_, v_))
    F_supg = tau * inner(R, dot(v_, nabla_grad(w))) * dx

if SUPG == True:
    F += F_supg

F1 = action(F, vp_)
J = derivative(F1, vp_, vp)

problem = NonlinearVariationalProblem(F1, vp_, bcs, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["nonlinear_solver"] = "newton"
prm["newton_solver"]["absolute_tolerance"] = 1e-6
prm["newton_solver"]["relative_tolerance"] = 1e-6
prm["newton_solver"]["convergence_criterion"] = "incremental"
prm["newton_solver"]["maximum_iterations"] = 20
prm["newton_solver"]["relaxation_parameter"] = 0.9
prm["newton_solver"]["linear_solver"] = "direct"
prm["newton_solver"]["error_on_nonconvergence"] = False

# if Iterative_Solver_NS == True:
#     prm["newton_solver"]["linear_solver"] = "gmres"
#     prm["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1e-5
#     prm["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1e-5
#     prm["newton_solver"]["krylov_solver"]["maximum_iterations"] = 75000
#     # prm["newton_solver"]["krylov_solver"]["monitor_convergence"] = True
#     # prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
#     # prm["newton_solver"]["krylov_solver"]["gmres"]["restart"] = 40
#     prm["newton_solver"]["krylov_solver"]["error_on_nonconvergence"] = False
#     prm["newton_solver"]["preconditioner"] = "none"
#     # info(prm, True)
#     # list_krylov_solver_preconditioners()


converged_flag, nIter = solver.solve()

v_, p_ = vp_.split(True)
out_u << v_
out_p << p_
