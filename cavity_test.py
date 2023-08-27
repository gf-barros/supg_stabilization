from ns_function import *
from numpy.random import seed
from numpy.random import randint
from dolfin import HDF5File

comm      = MPI.comm_world
rank      = comm.Get_rank()

N       = 10

seed(0)

values = randint(100, 500, N)
if rank == 0:
    print(values)
for i in range(N):
    Re = values[i]
    if rank == 0:
        print("Reynolds number is: {0:d}  Cont: {1:d}".format(Re,i))
    print("Statement")
    Solution = CavityProblem(Re, cont=i)
    #print(max(Solution))

# filename = 'Snapshots/snapshot_0.h5'
# snapshot = HDF5File(mesh.mpi_comm(), filename, 'r')
# snapshot.write(v_, "velocity")