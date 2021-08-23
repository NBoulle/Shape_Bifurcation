from firedrake import *
from fireshape import ShapeObjective
from L2tracking_PDEconstraint import Moore_Spence
import numpy as np

class L2trackingObjective(ShapeObjective):
    """
    Define the objective function.
    """

    # Initialization
    def __init__(self, pde_solver: Moore_Spence, Q, pvd_path, results_path, target, *args, **kwargs):
        super().__init__(Q, *args, **kwargs)

        self.pde_solver = pde_solver
        self.u = pde_solver.solution
        self.Q = Q
        self.Vdet = FunctionSpace(Q.mesh_r, "DG", 0)
        self.detDT = Function(self.Vdet)
        self.minVal = np.inf

        # Define callback
        self.cb = self.callback
        self.f_pvd = Function(self.pde_solver.V)
        self.f_pvd.rename("u","u")
        self.pvd_file = File(pvd_path)
        self.results_path = results_path
        open(self.results_path, 'w+').close()
        self.its = 0

        # Target parameter
        self.target = target
        self.f = Function(pde_solver.V).interpolate(Constant(1.0))

    def callback(self):
        # Compute value of the functional
        area = assemble(self.f*dx)
        value = assemble((1/area)*(self.u[1] - self.target)**2 * dx)

        # Write the solution at each step
        self.pvd_file.write(self.f_pvd.interpolate(self.pde_solver.solution[0]))

        # Save the value of the function to a file
        f1 = open(self.results_path, "a+")
        f1.writelines("%d,%e\n"%(self.its, value))
        f1.close()
        self.its = self.its + 1

    def value_form(self):
        """
        Evaluate misfit functional.
        """
        u = self.u

        # Normalize the functional by the area of the domain
        area = assemble(self.f*dx)
        value = (1/area)*(u[1] - self.target)**2 * dx

        # Print useful informations
        self.detDT.interpolate(det(grad(self.Q.T)))
        with u.sub(1).dat.vec_ro as x:
            param = x.norm()
        RelatDiff = norm(self.pde_solver.solution[0]-self.pde_solver.sol_opt[0]) / norm(self.pde_solver.solution[0])
        print("lambda = %e, functional = %e, diff = %e, det = %e" % (param, (param-self.target)**2, RelatDiff, min(self.detDT.vector())))
        value_assemble = assemble(value)

        # Ensure that we stay on the same branch
        if norm(self.pde_solver.solution[0]-self.pde_solver.sol_opt[0]) > 0.1*norm(self.pde_solver.solution[0]):
            value = np.nan * dx(self.pde_solver.mesh)
            value_assemble = float("inf")

        # Ensure not self intersection
        if min(self.detDT.vector()) <= 0.0:
            value = np.nan * dx(self.pde_solver.mesh)
            value_assemble = float("inf")

        # Use the last optimization improvement as initial guess
        if value_assemble < self.minVal:
            self.pde_solver.sol_opt.assign(self.pde_solver.solution)
            self.minVal = value_assemble
        else:
            self.pde_solver.solution.assign(self.pde_solver.sol_opt)
        print("value = %e, norm sol = %e"%(value_assemble, norm(self.pde_solver.solution[0])))

        return value
