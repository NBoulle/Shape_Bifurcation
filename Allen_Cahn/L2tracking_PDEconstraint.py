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

        # Create function spaces
        self.mesh = mesh_m
        self.V = FunctionSpace(self.mesh, "CG", 1)
        self.R = FunctionSpace(self.mesh, "R", 0)
        self.Z = MixedFunctionSpace([self.V, self.R, self.V])

        # Create solution
        self.sol_opt = Function(self.Z)
        self.solution_n = Function(self.Z)
        self.solution = Function(self.Z, name="State")

        # Initialize some constants
        self.diff = Constant(-1.0)
        self.init_area = assemble(1.0*dx(self.mesh))

        # Define Moore-Spence system of equations
        theta, lmbda, phi = split(self.solution)
        theta_n, lmbda_n, phi_n = split(self.solution_n)
        ttheta, tlmbda, tphi = TestFunctions(self.Z)
        F1 = self.residual(theta, lmbda, ttheta)
        F2 = derivative(self.residual(theta, lmbda, tphi), self.solution, as_vector([phi, 0, 0]))
        F3 = (inner(phi, phi) - 1)*tlmbda*dx
        self.F = F1 + F2 + F3

        # Define boundary conditions
        self.bcs = [DirichletBC(self.Z.sub(0), 0.0, "on_boundary"), DirichletBC(self.Z.sub(2), 0.0, "on_boundary")]

        # Solver parameters for solving the Moore-Spence system
        self.solver_params = {
                            "mat_type": "matfree",
                            "snes_type": "newtonls",
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
                            "pc_fieldsplit_0_fields": "0,2",
                            "pc_fieldsplit_1_fields": "1",
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

    # Residual of the Allen-Cahn equation
    def residual(self, u, lmbda, v):
        delta = 0.25
        F = (
            + delta * inner(grad(u), grad(v))*dx
            - inner(lmbda*u + u**3 - u**5, v)*dx
            )
        return F

    # Print bifurcation parameter
    def print_lambda(self, sol):
        with sol.sub(1).dat.vec_ro as x:
            param = x.norm()
        print("### lambda = %f ###\n"%param)

    # Compute initial guess for the Moore-Spence system
    def compute_guess(self, domain, branch):

        # Check that the initial guess exists
        if domain == "round_square":
            if branch == 1:
                init_lambda = Constant(1.1)
            elif branch == 3:
                init_lambda = Constant(4.8)
            else:
                raise ValueError("Unknown branch : %d for domain : %s" % (branch, domain))
        else:
            if branch == 1:
                init_lambda = Constant(1.4)
            elif branch == 5:
                init_lambda = Constant(10)
            else:
                raise ValueError("Unknown branch : %d for domain : %s" % (branch, domain))

        # Load the initial guess
        vtu_class = vtu("Initial_guesses/%s/%s.vtu" %(domain, branch))
        W = VectorFunctionSpace(self.mesh, "CG", 1, dim=2)
        X = interpolate(SpatialCoordinate(self.mesh), W)
        reader = lambda X: vtu_class.ProbeData(np.c_[X, np.zeros(X.shape[0])], "solution")[:,0]
        u_init = Function(self.V)
        u_init.dat.data[:] = reader(X.dat.data_ro)

        # Assign guess and lambda to the solution
        self.solution.split()[0].assign(u_init)
        self.solution.split()[1].assign(init_lambda)

        # Compute the corresponding eigenvector
        eigenfunctions = self.get_initial_guess(init_lambda, u_init)

        # Record the min difference with the initial guess
        min_diff = float("inf")

        # Try all the eigenmodes and use the solution closer to the initial guess
        for guess_i in range(len(eigenfunctions)):
            print("### Solving Moore-Spence system with eigenmode %d / %d ###" % (guess_i+1, len(eigenfunctions)))
            try:
                # Assign guess, lambda, and eigenmode to the solution
                self.solution.split()[0].assign(u_init)
                self.solution.split()[1].assign(init_lambda)
                self.solution.split()[2].assign(eigenfunctions[guess_i])

                solve(self.F == 0, self.solution, bcs=self.bcs,
                     solver_parameters=self.solver_params)

                # Test if the result is sufficiently close to the initial guess
                diff_guess = norm(self.solution[0]-u_init)
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
        bcs = [DirichletBC(self.V, 0.0, "on_boundary")]

        # Check if an initial function is given
        if u_init is None:
            print("### Solving PDE to find initial guess ###")
            # Using guess for parameter lm, solve for state theta (th)
            A = self.residual(th, lm, tth)

            params = {
                    "snes_max_it": 100,
                    "snes_atol": 1.0e-9,
                    "snes_rtol": 0.0,
                    "snes_converged_reason": None,
                    "snes_linesearch_type": "basic",
                    "ksp_type": "preonly",
                    "pc_type": "lu"
                    }
            solve(A == 0, th, bcs=bcs, solver_parameters = params)

        else:
            print("### Using prescribed initial guess ###")
            th.assign(u_init)

        # Set up the eigenvalue solver
        B = derivative(self.residual(th, lm, TestFunction(self.V)), th, TrialFunction(self.V))
        petsc_M = assemble(inner(TestFunction(self.V), TrialFunction(self.V))*dx, bcs=bcs).petscmat
        petsc_B = assemble(B, bcs=bcs).petscmat

        # Number of eigenvalues to try
        num_eigenvalues = 3

        # Solver options
        opts = PETSc.Options()
        opts.setValue("eps_converged_reason", None)
        opts.setValue("eps_monitor_conv", None)
        opts.setValue("eps_target_magnitude", None)
        opts.setValue("eps_target", 0)
        opts.setValue("st_type", "sinvert")

        # Solve the eigenvalue problem
        eps = SLEPc.EPS().create(comm=COMM_WORLD)
        eps.setDimensions(num_eigenvalues)
        eps.setOperators(petsc_B, petsc_M)
        eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        eps.setFromOptions()

        # Solve the eigenvalue problem
        print("### Solving eigenvalue problem ###")
        eps.solve()

        # Extract the eigenfunctions
        eigenfunctions = []
        eigenfunction = Function(self.V, name="Eigenfunction")
        for i in range(min(eps.getConverged(), num_eigenvalues)):
            with eigenfunction.dat.vec_wo as x:
                eps.getEigenvector(i,x)
            print(norm(eigenfunction))
            eigenfunctions.append(eigenfunction.copy(deepcopy=True))

        return eigenfunctions

    # Solve Moore-Spence system
    def solve(self):
        super().solve()

        # Print relative domain area wrt to initial area
        area = assemble(1.0*dx(self.mesh))
        print("\n### changed area = %f ###" % (area/self.init_area))

        # Try to solve the Moore-Spence system
        try:
            print("### Solving Moore-Spence system ###")
            solve(self.F == 0, self.solution, bcs=self.bcs,
                 solver_parameters=self.solver_params)

            # Record the last successful PDE step
            self.solution_n.assign(self.solution)

        except:
            # Assign last successful optimization step
            self.solution.assign(self.sol_opt) # return Nan in that case, check if the solution has changed
