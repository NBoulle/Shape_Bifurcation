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
        self.V = VectorFunctionSpace(self.mesh, "CG", 1)
        self.R = FunctionSpace(self.mesh, "R", 0)
        self.Z = MixedFunctionSpace([self.V, self.R, self.V])

        # Record best mesh
        self.mesh_opt = mesh_m

        # Create solution
        self.sol_opt = Function(self.Z)
        self.solution_n = Function(self.Z)
        self.solution = Function(self.Z, name="State")

        # Initialize some constants
        self.diff = Constant(-1.0)
        self.init_area = assemble(1.0*dx(self.mesh))

        self.failed_to_solve = False

        # Define Moore-Spence system of equations
        theta, lmbda, phi = split(self.solution)
        theta_n, lmbda_n, phi_n = split(self.solution_n)
        ttheta, tlmbda, tphi = TestFunctions(self.Z)
        F1 = self.residual(theta, lmbda, ttheta)
        zero = Function(self.V)
        F2_ = self.residual(theta, lmbda, tphi)
        F2 = derivative(F2_, self.solution, as_vector([w for w in phi] + [0] + [w for w in zero]))
        F3 = (inner(phi, phi) - 1)*tlmbda*dx
        self.F = F1 + F2 + F3

        # Define boundary conditions
        self.bcs = [DirichletBC(self.Z.sub(0), Constant((0.0,  0.0)), 1),
                    DirichletBC(self.Z.sub(2), Constant((0.0,  0.0)), 1)]

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

    # Residual of the hyperelasticity equations
    def residual(self, u, lmbda, v):

        # Bifurcation parameter
        eps = lmbda

        # Body force per unit volume
        B   = Constant((0.0, -1000))

        # Kinematics
        I = Identity(2)             # Identity tensor
        F = I + grad(u)             # Deformation gradient
        C = F.T*F                   # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ic = tr(C)
        J  = det(F)

        # Elasticity parameters
        E, nu = 1000000.0, 0.3
        mu, lmbda_cte = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu/2)*(Ic - 2) - mu*ln(J) + (lmbda_cte/2)*(ln(J))**2

        # Total potential energy
        T = Constant((-50000, 0.0))
        Energy = psi*dx - dot(B, u)*dx
        Func = derivative(Energy, u, v)

        u0 = eps*Constant((-1, 0.0))
        gamma = 10**15
        P1 = mu*F*(Ic*I)
        P2 = -mu*F.T + (lmbda/2)*ln(J)*F.T
        n = FacetNormal(self.mesh)
        t = (P1+P2)*n

        Func = Func + gamma*inner(u-u0,v)*ds(2) - inner(t,v)*ds(2)

        return Func

    # Print bifurcation parameter
    def print_lambda(self,sol):
        with sol.sub(1).dat.vec_ro as x:
            param = x.norm()
        print("### lambda = %f ###\n"%param)

    # Compute initial guess for the Moore-Spence system
    def compute_guess(self):

        # Load the initial guess
        vtu_class = vtu("Initial_guesses/0.vtu")
        X = interpolate(SpatialCoordinate(self.mesh), self.V)
        reader = lambda X: vtu_class.ProbeData(np.c_[X, np.zeros(X.shape[0])], "u")[:,0:2]
        u_init = Function(self.V)
        u_init.dat.data[:] = reader(X.dat.data_ro)

        # Initial bifurcation parameter
        init_lambda = Constant(0.04)

        # Assign guess and lambda to the solution
        self.solution.split()[0].assign(u_init)
        self.solution.split()[1].assign(init_lambda)

        # Compute the corresponding eigenvector
        eigenfunctions = self.get_initial_guess(init_lambda, u_init)

        # Record the min difference with the initial guess
        min_diff = float("inf")

        # Try all the eigenmodes and use the solution closer to the initial guess
        for guess_i in range(len(eigenfunctions)):
            print("### Solving Moore-Spence system with eigenmode %d / %d ###" % (guess_i, len(eigenfunctions)-1))
            try:
                # Assign guess, lambda, and eigenmode to the solution
                self.solution.split()[0].assign(u_init)
                self.solution.split()[1].assign(init_lambda)
                self.solution.split()[2].assign(eigenfunctions[guess_i])

                solve(self.F == 0, self.solution, bcs=self.bcs,
                      solver_parameters=self.solver_params)

                # Test if the result is sufficiently close to the initial guess
                diff_guess = norm(self.solution.split()[0]-u_init)
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
        bcs = [DirichletBC(self.V, Constant((0.0,  0.0)), 1)]

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
        J = derivative(self.residual(th, lm, TestFunction(self.V)), th, TrialFunction(self.V))
        A = assemble(J, bcs=bcs)
        M = assemble(inner(TestFunction(self.V), TrialFunction(self.V))*dx, bcs=bcs)

        # There must be a better way of doing this
        from firedrake.preconditioners.patch import bcdofs
        for bc in bcs:
            # Ensure symmetry of M
            M.M.handle.zeroRowsColumns(bcdofs(bc), diag=0)

        # Number of eigenvalues to try
        num_eigenvalues = 3

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
        eps.setDimensions(num_eigenvalues)
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
            print(norm(eigenfunction))
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
            # assign last successful optimization step
            self.solution.assign(self.sol_opt) # return Nan in that case, check if the solution has changed
            self.failed_to_solve = True
