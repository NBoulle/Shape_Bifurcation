from firedrake import *
from fireshape import *
import ROL
import numpy as np

# Load Python files
from L2tracking_PDEconstraint import Moore_Spence
from L2tracking_objective import L2trackingObjective

def shape_optimization(target):

    # Setup the initial mesh and problem
    mesh = RectangleMesh(40, 40, 1, 0.1)
    Q = FeControlSpace(mesh)
    inner = ElasticityInnerProduct(Q, fixed_bids=[1,2])
    q = ControlVector(Q, inner)

    # Setup PDE constraint
    mesh_m = Q.mesh_m
    e = Moore_Spence(mesh_m)

    # Compute the initial guess for the Moore-Spence system using deflation solution
    e.compute_guess()
    e.sol_opt.assign(e.solution)
    print("### initial state found ###")

    # Create directory for saving results
    directory = "Result"
    if not os.path.exists(directory):
        os.makedirs(directory)
    pvd_path = directory + "/solution/u.pvd"
    results_path = directory + "/iterations.csv"

    # Save the initial mesh
    np.savetxt(directory + "/mesh_init.txt", Q.mesh_m.coordinates.dat.data[:,:], delimiter=',')

    # Create PDEconstrained objective functional
    J_ = L2trackingObjective(e, Q, pvd_path, results_path, target)
    J = ReducedObjective(J_, e)

    # ROL parameters
    params_dict = {
                    'Status Test':{'Gradient Tolerance':1e-6,
                                   'Step Tolerance':1e-12,
                                   'Iteration Limit':100},
                    'Step':{'Type':'Trust Region',
                            'Trust Region':{'Initial Radius': 1e-4,
                                            'Maximum Radius':1e8,
                                            'Subproblem Solver':'Dogleg',
                                            'Radius Growing Rate':2.5,
                                            'Step Acceptance Threshold':0.05,
                                            'Radius Shrinking Threshold':0.05,
                                            'Radius Growing Threshold':0.9,
                                            'Radius Shrinking Rate (Negative rho)':0.0625,
                                            'Radius Shrinking Rate (Positive rho)':0.25,
                                            'Radius Growing Rate':2.5,
                                            'Sufficient Decrease Parameter':1e-4,
                                            'Safeguard Size':100.0,
                                           }
                           },
                    'General':{'Print Verbosity':0, #set to any number >0 for increased verbosity
                               'Secant':{'Type':'Limited-Memory BFGS', #BFGS-based Hessian-update in trust-region model
                                         'Maximum Storage':10
                                        }
                              }
                    }

    # Solve the optimization problem
    params = ROL.ParameterList(params_dict, "Parameters")
    problem = ROL.OptimizationProblem(J, q)
    solver = ROL.OptimizationSolver(problem, params)
    solver.solve()

    # Write final lambda and save the final mesh
    with e.sol_opt.sub(1).dat.vec_ro as x:
        param = x.norm()
    print("\nlambda = %e"%param)
    np.savetxt(directory + "/mesh_final.txt", e.mesh_opt.coordinates.dat.data[:,:], delimiter=',')

if __name__ == "__main__":
    """
    Set the target bifurcation parameter.

    The default settings correspond to reproducing Fig 10d of the paper.
    """

    target = 0.1

    # Run the shape optimization
    shape_optimization(target)
