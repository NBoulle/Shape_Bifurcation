from firedrake import *
from fireshape import PdeConstraint
from slepc4py import SLEPc
from firedrake.petsc import PETSc
import numpy as np

# Load file to load initial guesses
import sys
sys.path.append('../utils')
from vtktools import vtu

class Moore_Spence(PdeConstraint):
    """
    Define the PDE constraint as a Moore-Spence system.
    """

    # Initialize the system
    def __init__(self, mesh_m):
        super().__init__()

        # Create function space
        self.mesh = mesh_m
        self.Vu = VectorFunctionSpace(self.mesh, "CG", 2)
        self.Vp = FunctionSpace(self.mesh, "CG", 1)
        self.V = MixedFunctionSpace([self.Vu, self.Vp])
        self.R = FunctionSpace(self.mesh, "R", 0)
        self.Z = MixedFunctionSpace([self.V, self.R, self.V])

        # Create solution
        self.sol_opt = Function(self.Z)
        self.solution_n = Function(self.Z)
        self.solution = Function(self.Z, name="State")

        # Initialize some constans
        self.diff = Constant(-1.0)
        self.init_area = assemble(1.0*dx(self.mesh))
        self.failed_to_solve = False

        # Define Moore-Spence system of equations
        u, p, lmbda, wu, wp = split(self.solution)
        tu, tp, tlmbda, twu, twp = TestFunctions(self.Z)
        F1 = self.residual(u, p, lmbda, tu, tp)
        zero = Function(self.V)
        F2 = derivative(self.residual(u, p, lmbda, twu, twp), self.solution, as_vector([w for w in wu] + [wp] + [0] + [w for w in zero]))
        area = assemble(1.0*dx(self.mesh))
        F3 = (inner(wu, wu) - 1/area)*tlmbda*dx
        self.F = F1 + F2 + F3

        # Define boundary conditions
        x = SpatialCoordinate(self.mesh)
        poiseuille = interpolate(as_vector([-(x[1] + 1) * (x[1] - 1), 0.0]), self.Z.sub(0))
        bc_inflow = DirichletBC(self.Z.sub(0), poiseuille, 10)
        bc_wall = DirichletBC(self.Z.sub(0), Constant((0, 0)), [13, 14])
        bc_wall2 = DirichletBC(self.Z.sub(3), Constant((0, 0)), [10, 13, 14])
        self.bcs = [bc_inflow, bc_wall, bc_wall2]

        # Solver parameters for solving the Moore-Spence system
        self.solver_params = {
                            "mat_type": "matfree",
                            "snes_type": "newtonls",
                            "snes_monitor": None,
                            "snes_converged_reason": None,
                            "snes_linesearch_type": "l2",
                            "snes_linesearch_maxstep": 0.5,
                            "snes_linesearch_damping": 0.5,
                            "snes_max_it": 100,
                            "snes_atol": 1.0e-8,
                            "snes_rtol": 0.0,
                            "ksp_type": "fgmres",
                            "ksp_max_it": 10,
                            "pc_type": "fieldsplit",
                            "pc_fieldsplit_type": "schur",
                            "pc_fieldsplit_schur_fact_type": "full",
                            "pc_fieldsplit_0_fields": "0,1,3,4",
                            "pc_fieldsplit_1_fields": "2",
                            "fieldsplit_0_ksp_type": "preonly",
                            "fieldsplit_0_pc_type": "python",
                            "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
                            "fieldsplit_0_assembled_pc_type": "lu",
                            "fieldsplit_0_assembled_pc_factor_mat_solver_type": "mumps",
                            "fieldsplit_0_assembled_mat_mumps_icntl_14": 200,
                            "mat_mumps_icntl_14": 200,
                            "fieldsplit_1_ksp_type": "gmres",
                            "fieldsplit_1_ksp_max_it": 1,
                            "fieldsplit_1_ksp_convergence_test": "skip",
                            "fieldsplit_1_pc_type": "none",
                            }

    # Residual of the Navier-Stokes equations
    def residual(self, u, p, lmbda, v, q):
        Re = lmbda
        F = (
              1.0/Re * inner(grad(u), grad(v))*dx
            + inner(grad(u)*u, v)*dx
            - div(v)*p*dx
            + q*div(u)*dx
            )
        return F

    # Print bifurcation parameter
    def print_lambda(self,sol):
        with sol.sub(1).dat.vec_ro as x:
            param = x.norm()
        print("### lambda = %f ###\n"%param)

    # Compute initial guess for the Moore-Spence system
    def compute_guess(self):

        # Load the initial guess
        vtu_class = vtu("Initial_guesses/0.vtu")
        W = VectorFunctionSpace(self.mesh, "CG", 2, dim=2)
        X = interpolate(SpatialCoordinate(self.mesh), W)
        reader = lambda X: vtu_class.ProbeData(np.c_[X, np.zeros(X.shape[0])], "Velocity")[:,0:2]
        u_vtk = Function(self.Vu)
        u_vtk.dat.data[:] = reader(X.dat.data_ro)
        W = VectorFunctionSpace(self.mesh, "CG", 1, dim=2)
        X = interpolate(SpatialCoordinate(self.mesh), W)
        reader = lambda X: vtu_class.ProbeData(np.c_[X, np.zeros(X.shape[0])], "Pressure")[:,0]
        p_vtk = Function(self.Vp)
        p_vtk.dat.data[:] = reader(X.dat.data_ro)
        u_init = Function(self.V)
        u_init.split()[0].assign(u_vtk)
        u_init.split()[1].assign(p_vtk)

        # Initial bifurcation parameter
        init_lambda = Constant(19.0)

        # Assign guess and lambda to the solution
        self.solution.split()[0].assign(u_init.split()[0])
        self.solution.split()[1].assign(u_init.split()[1])
        self.solution.split()[2].assign(init_lambda)

        # Compute the corresponding eigenvector
        eigenfunctions = self.get_initial_guess(init_lambda, u_init)

        # Record the min difference with the initial guess
        min_diff = float("inf")

        # Try all the eigenmodes and use the solution closer to the initial guess
        for guess_i in range(len(eigenfunctions)):
            print("### Solving Moore-Spence system with eigenmode %d / %d ###" % (guess_i+1, len(eigenfunctions)))
            try:
                # Assign guess, lambda, and eigenmode to the solution
                self.solution.split()[0].assign(u_init.split()[0])
                self.solution.split()[1].assign(u_init.split()[1])
                self.solution.split()[2].assign(init_lambda)
                self.solution.split()[3].assign(eigenfunctions[guess_i].split()[0])
                self.solution.split()[4].assign(eigenfunctions[guess_i].split()[1])

                solve(self.F == 0, self.solution, bcs=self.bcs,
                      solver_parameters=self.solver_params)

                # Test if the result is sufficiently close to the initial guess
                diff_guess = norm(self.solution.split()[0]-u_init.split()[0])
                if  diff_guess < min_diff:
                    min_diff = diff_guess
                    # Record the best initial guess
                    self.solution_n.assign(self.solution)
            except:
                if (guess_i == len(eigenfunctions)-1) and (min_diff == float("inf")):
                    raise StopIteration("Moore-Spence system didn't converge for the initial guess")

        # Assign the best initial guess
        self.solution.assign(self.solution_n)

    # Compute initial guess for the eigenfunctions
    def get_initial_guess(self, lm, u_init = None):

        # Set up functions and Dirichlet bcs
        th = Function(self.V)
        tth = TestFunction(self.V)
        (u, p) = split(th)
        (v, q) = split(tth)

        # Boundary conditions
        bcs = [DirichletBC(self.V.sub(0), Constant((0, 0)), [10, 13, 14])]

        print("### Using prescribed initial guess ###")
        th.assign(u_init)

        # Set up the eigenvalue solver
        J = derivative(self.residual(u, p, lm, v, q), th, TrialFunction(self.V))
        A = assemble(J, bcs=bcs, mat_type="aij")
        M = assemble(inner(TestFunction(self.V), TrialFunction(self.V))*dx, bcs=bcs, mat_type="aij")

        # There must be a better way of doing this
        from firedrake.preconditioners.patch import bcdofs
        for bc in bcs:
            # Ensure symmetry of M
            M.M.handle.zeroRowsColumns(bcdofs(bc), diag=0)

        # Number of eigenvalues to try
        num_eigenvalues = 1

        # Solver options
        opts = PETSc.Options()

        parameters = {
             "mat_type": "aij",
             "eps_monitor_conv" : None,
             "eps_converged_reason": None,
             "eps_type": "krylovschur",
             "eps_nev" : num_eigenvalues,
             "eps_max_it": 50,
             "eps_tol" : 1e-10,
             "eps_which": "smallest_magnitude",
             "st_type": "sinvert",
             "st_ksp_type": "preonly",
             "st_pc_type": "lu",
             "st_pc_factor_mat_solver_type": "mumps",
             "st_ksp_max_it": 10,
             }

        for k in parameters:
            opts[k] = parameters[k]

        # Solve the eigenvalue problem
        eps = SLEPc.EPS().create(comm=COMM_WORLD)
        eps.setOperators(A.M.handle, M.M.handle)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setFromOptions()
        print("### Solving eigenvalue problem ###")
        eps.solve()

        # Extract the eigenfunctions
        eigenfunctions = []
        eigenfunction = Function(self.V, name="Eigenfunction")
        for i in range(min(eps.getConverged(), num_eigenvalues)):
            with eigenfunction.dat.vec_wo as x:
                eps.getEigenvector(i,x)
            eigenfunctions.append(eigenfunction.copy(deepcopy=True))
        return eigenfunctions

    # Solve Moore-Spence system
    def solve(self):
        super().solve()

        # Print relative domain area wrt to initial area
        area = assemble(1.0*dx(self.mesh))
        print("\n### changed area = %f ###" % (area/self.init_area))

        try:
            print("### Solving Moore-Spence system ###")
            solve(self.F == 0, self.solution, bcs=self.bcs,
                 solver_parameters=self.solver_params)
            self.failed_to_solve = False

            # Record the last successful PDE step
            self.solution_n.assign(self.solution)

        except:
            # Assign last successful optimization step
            self.solution.assign(self.sol_opt) # return Nan in that case, check if the solution has changed
            self.failed_to_solve = True
