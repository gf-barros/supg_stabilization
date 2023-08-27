from dolfin import *



def left(x, on_boundary):
    return x[0] < DOLFIN_EPS and on_boundary
def top(x, on_boundary):
    return x[1] > 1.0 - DOLFIN_EPS and on_boundary
def right(x, on_boundary):
    return x[0] > 1.0 - DOLFIN_EPS and on_boundary
def bottom(x, on_boundary):
    return x[1] < DOLFIN_EPS and on_boundary


#================= Problem Definition ===================
mesh = UnitSquareMesh(100, 100)
h = CellDiameter(mesh)
Q = FunctionSpace(mesh, "CG", 1)
dphi, v = TrialFunction(Q), TestFunction(Q)
phi = Function(Q)


#================= Parameters ===========================
velocity = as_vector([1.0, 1.0])
k = 0.01
f = Constant(1.0)


#================= Boundary Conditions===================
BC1= Constant(0.0)
bc_left = DirichletBC(Q, BC1, left)
bc_top = DirichletBC(Q, BC1, top)
bc_right = DirichletBC(Q, BC1, right)
bc_bottom = DirichletBC(Q, BC1, bottom)
bcs = [bc_left, bc_top, bc_right, bc_bottom]



#================= Variational Form ====================
F = v*dot(velocity, grad(dphi))*dx + k*dot(grad(v), grad(dphi))*dx - f*v*dx



#================= Stabilization =======================
r = dot(velocity, grad(dphi)) - k*div(grad(dphi)) - f
vnorm = sqrt(dot(velocity, velocity))
delta = h/(2.0*vnorm)
F += delta*dot(velocity, grad(v))*r*dx






#====================== Solver =========================
a = lhs(F)
L = rhs(F)
solve(a == L, phi, bcs)




#====================== Output =========================
outfile = File("phi.pvd")
outfile << phi