# from future import print_function
from fenics import *
import numpy as np
import random
import time as tm

class InitialConditions(UserExpression):
    def __init__(self, **kwargs):
        random.seed(2 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.0 + 0.02*(0.5 - random.random())
        values[1] = 0.0
    def value_shape(self):
        return (3,)

def erro(v00, v0, v, dt, dtprev, tauAbs, tauRel):
    u_local     = v.vector().get_local()
    v0_         = v0.vector().get_local()
    v00_        = v00.vector().get_local()
    eta         = (dt + dtprev)/dt
    LTE         = -(1.0/eta)*u_local + (1.0/(eta-1.0))*v0_ - (1.0/(eta*(eta-1.0)))*v00_
    # comm      = MPI.comm_world
    # rank      = comm.Get_rank()
    # node_size = comm.Get_size()
    e_local     = LTE
    sum_local   = 0.0
    for i in range(u_local.size):
        tmp = max(abs(u_local[i]),abs(u_local[i] + e_local[i]))
        tmp = e_local[i]/(tauAbs + tauRel*tmp)
        sum_local += tmp*tmp
    return sum_local

def PC11(cont, phi, phi0, dt, enorm, eu, eu0, eu00, accept):
    dtMin = 1.e-30
    dtMax = 1.e+3
    # A = ElementArea(mesh)
    if cont == 0:
        eu = enorm
    if cont == 1:
        eu0 = eu
        eu = enorm
    if cont >= 2:
        eu00 = eu0
        eu0 = eu
        eu = enorm
    auxC = (dt)/(dtprev)
    auxB = 0.9*(eu0/(eu*eu))**(1/3)
    coef =  auxB * auxC
    if accept == 0 and coef >= 1.0:
        coef = 0.8
    rcoef =  min(10,max(coef, 0.1))
    dtcalc = rcoef*dt
    dt = dtcalc
    dt = min(dtMax, max(dtcalc, dtMin))
    # print("step= {0:d} dtcalc= {1:10.3e} coef= {2:5.3g} next_dt= {3:10.3e} accept= {4:d}"\
    #     .format(cont,dtcalc,coef,dt, accept))
    return dt, eu, eu0, eu00

def Cavity_boundary_condition(W):
    class Lid(SubDomain):
        def inside(self, x, on_boundary):
            return(on_boundary and near (x[1], 1.0))

    class Walls(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (near(x[0], 0.0) or near(x[0], 1.0) or near(x[1], 0.0))

    def zero(x):
        return (near(x[0], 0.0) and near(x[1], 0.0))
    # Dirichlet value
    #g1 = Expression(("exp(-(pow(x[0] - 0.5, 2)/(2*pow(0.18,2))))", 0.0), degree = 2)
    g1 = Constant((1.0, 0.0))
    g2 = Constant((0.0, 0.0))
    g3 = Constant(0.0)
    # conditions
    bc1 = DirichletBC(W.sub(0), g1, Lid())
    bc2 = DirichletBC(W.sub(0), g2, Walls())
    bc3 = DirichletBC(W.sub(1), g3, zero, 'pointwise')
    bcs = [bc1, bc2, bc3]
    return bcs

Chronometer                 = False
VTK_Output                  = True                        # True = VTK / False = HDF5
SUPG                        = True    
Directory                   = "/mnt/c/temp/Results_NS/"             # "/mnt/c/temp/Results_NS/"
Directory_PP                = "/mnt/c/temp/"
Directory_CD                = "/mnt/c/temp/Results_CD/"             # "/mnt/c/temp/Results_NS/"
Reynolds_Ramp               = True
Iterative_Solver_NS         = True                        # True = GMRES / False = LU
Problem_NS                  = "Cavity"
SUPS                        = True                        # True = SUPS / False = Taylor-Hood    
LSIC                        = True 
set_log_level(40)
comm      = MPI.comm_world
rank      = comm.Get_rank()

parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True

Re                          = 100
N                           = 30
t                           = 0.0
Fr                          = 1.0
Froude_inv                  = Constant(1.0/pow(Fr,2))
f                           = Froude_inv*Constant((0.0,0.0)) 
nu                          = Constant(1/Re) 
Number_of_Snapshots         = 10

mesh        = UnitSquareMesh(N, N)

if Chronometer == True:
    start = tm.time()


if SUPS == False:
    P1v         = VectorElement("Lagrange", mesh.ufl_cell(), 2)
    parameters["form_compiler"]["quadrature_degree"] = 3
else:
    P1v         = VectorElement("Lagrange", mesh.ufl_cell(), 1)
    parameters["form_compiler"]["quadrature_degree"] = 1


P1          = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH          = P1v*P1
W           = FunctionSpace(mesh, TH)
vp          = TrialFunction(W)
(v, p)      = split(vp)
(w, q)      = TestFunctions(W)
vp_         = Function(W)
v_, p_      = split(vp_)
u_init = InitialConditions()
vp_.interpolate(u_init)
vn          = Function(W.sub(0).collapse())
vnn         = Function(W.sub(0).collapse())
bcs         = Cavity_boundary_condition(W)
h           = CellDiameter(mesh)
vnorm           = sqrt(dot(v_, v_))
dt          = 0.0
F           = (inner(dot(v_,nabla_grad(v)), w) + nu*inner(grad(w), grad(v)) - inner(p,div(w)) + inner(q,div(v)) - inner(f, w))*dx
R           = dot(v_,nabla_grad(v)) - nu*div(grad(v)) + grad(p) - f
tau         = ((2.0*vnorm/h)**2 + 9.0*(4.0*nu/(h*2))**2)**(-0.5)


#Stabilization

vnorm       = sqrt(dot(v_, v_))
tau_lsic    = (vnorm*h)/2.0
F_lsic      = tau_lsic*inner(div(v), div(w))*dx
F_supg      = tau*inner(R, dot(v_, nabla_grad(w)))*dx
F_pspg      = tau*inner(R, grad(q))*dx

if SUPG == True:
    F      += F_supg

if SUPS == True:
    F      += F_pspg

if LSIC == True:
    F      += F_lsic


F1          = action(F, vp_)
J           = derivative(F1, vp_, vp)

problem = NonlinearVariationalProblem(F1, vp_, bcs, J)
solver  = NonlinearVariationalSolver(problem)
prm = solver.parameters
prm["nonlinear_solver"]                         ="newton"
prm["newton_solver"]["absolute_tolerance"]      = 1E-6
prm["newton_solver"]["relative_tolerance"]      = 1E-6
prm["newton_solver"]["convergence_criterion"]   = "incremental"
prm["newton_solver"]["maximum_iterations"]      = 20
prm["newton_solver"]["relaxation_parameter"]    = 0.9
prm["newton_solver"]["linear_solver"]           = "direct"
prm["newton_solver"]["error_on_nonconvergence"] = False

if Iterative_Solver_NS == True:
    prm["newton_solver"]["linear_solver"] = "gmres"
    prm["newton_solver"]["krylov_solver"]["absolute_tolerance"] = 1E-5
    prm["newton_solver"]["krylov_solver"]["relative_tolerance"] = 1E-5
    prm["newton_solver"]["krylov_solver"]["maximum_iterations"] = 75000
    #prm["newton_solver"]["krylov_solver"]["monitor_convergence"] = True
    #prm["newton_solver"]["krylov_solver"]["nonzero_initial_guess"] = True
    #prm["newton_solver"]["krylov_solver"]["gmres"]["restart"] = 40
    prm["newton_solver"]["krylov_solver"]["error_on_nonconvergence"] = False
    prm["newton_solver"]["preconditioner"] = "none"
    #info(prm, True)
    #list_krylov_solver_preconditioners()



if VTK_Output == True:
    filenameu = Directory + "velocity.pvd"
    filenamep = Directory + "pressure.pvd"
    out_u = File(filenameu, "compressed")
    out_p = File(filenamep, "compressed")

    def export_output(v_, p_, t):
        out_u << (v_,t)
        out_p << (p_,t)
        return 
else:
    filenameu = Directory + "velocity.xdmf"
    filenamep = Directory + "pressure.xdmf"
    out_u = XDMFFile(filenameu)
    out_p = XDMFFile(filenamep)

    def export_output(v_, p_, t):
        out_u.write(v_, t)
        out_p.write(p_, t)
        return 



np.random.seed(0)
values = np.random.randint(100, 500, Number_of_Snapshots)
print(values)
for i in range(Number_of_Snapshots):
    Re = values[i]
    nu.assign(1/values[i])
    if Reynolds_Ramp == False:
        converged_flag, nIter = solver.solve()
    else:
        step    = Re*0.1
        Pass_Flag = True
        Re_ramp = Re*0.1
        while Re_ramp + 1.e-2 < Re:
            if Pass_Flag == True:
                Re_ramp = Re_ramp + 0.1*Re
            else:
                Re_ramp = Re_ramp - 0.05*Re
            if rank == 0:
                print("Re = {}".format(Re_ramp))
            nu.assign(1/Re_ramp)   
            nIter, converged_flag = solver.solve()
            #print("Converged Flag = {0:g}      nIter = {1:d}".format(converged_flag,nIter))

            if converged_flag ==  False:
                Pass_Flag = False
                v_   = oldv
            else:
                Pass_Flag = True  
                oldv = v_  
    v_ , p_ = vp_.split(True)
    export_output(v_, p_, t)


if Chronometer == True:
    end = tm.time()
    print("======== EXECUTION TIME: {0:10.3e} seconds. ========".format(end - start))


#TO DO: EXPORT IN HDF5 FOR PARALLEL RUNNING
v_vec = Vector(MPI.comm_self, v_.vector().local_size())
v_.vector().gather(v_vec, W.sub(0).collapse().dofmap().dofs())
#print(len(v_.vector().get_local()))

g_vec = Vector(MPI.comm_world, W.sub(0).collapse().dim())
g_vec = v_.vector().gather_on_zero()  
mine = MPI.comm_world.bcast(g_vec)


# Reconstruct
if MPI.comm_world.rank == 0:
    my_g = g_vec
else:
    my_g = Vector(MPI.comm_self, W.sub(0).collapse().dim())
    my_g.set_local(mine)

print(len(mine))
#return