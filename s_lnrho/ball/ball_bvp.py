import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)




# Parameters
Ro = 1
Nphi, Ntheta, Nr = 1, 1, 32
Rayleigh = 1e3
Prandtl = 1
epsilon = 1e-2
t_buoy = np.sqrt(1/epsilon)
max_timestep = t_buoy
stop_sim_time = 2000*t_buoy
kappa = epsilon*(Rayleigh * Prandtl)**(-1/2)
nu = epsilon*(Rayleigh / Prandtl)**(-1/2)
gamma = 5/3
R = 1
Cv = R * 1 / (gamma-1)
Cp = R + Cv

timestepper = d3.SBDF2
dtype = np.float64
mesh = None
#mesh = [16,16]

def ball_HSE_BVP(N2_func, g_func, Lconv_func,  Nr=Nr, Ro=Ro, gamma=5/3, R=1):

    # Bases
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=np.float64, mesh=None, comm=MPI.COMM_SELF)
    basis = d3.BallBasis(coords, shape=(1, 1, Nr), radius=Ro, dealias=1, dtype=np.float64)
    s2_basis = basis.S2_basis()
    phi, theta, r = dist.local_grids(basis)
    phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)

    # Fields
    ln_rho = dist.Field(name='ln_rho', bases=basis)
    s = dist.Field(name='s', bases=basis)
    Q = dist.Field(name='Q', bases=basis)
    g_phi = dist.Field(name='g_phi', bases=basis)
    tau_s = dist.Field(name='tau_s', bases=s2_basis)
    tau_Q = dist.Field(name='tau_Q', bases=s2_basis)
    tau_rho = dist.Field(name='tau_rho', bases=s2_basis)
    tau_g_phi = dist.Field(name='tau_g_phi', bases=s2_basis)
    N2_input = dist.Field(name='N2_input', bases=basis)



    grad_pom0 = dist.VectorField(coords, name='grad_pom0', bases=basis.radial_basis)
    grad_s0 = dist.VectorField(coords, name='grad_s0', bases=basis.radial_basis)
    pom0 = dist.Field(name='pom0', bases=basis.radial_basis)
    g = dist.VectorField(coords, name='g', bases=basis.radial_basis)
    Q = dist.Field(name='Q', bases=basis)
    ones = dist.Field(name='ones', bases=basis)
    ones['g'] = 1

    # Substitutions
    Cv = R / (gamma-1)
    Cp = R + Cv
    er = dist.VectorField(coords, bases=basis.radial_basis)
    er['g'][2] = 1
    rvec = dist.VectorField(coords, bases=basis.radial_basis)
    rvec['g'][2] = r

    lift_basis = basis.derivative_basis(0)
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    ln_pomega = gamma*(s/Cp + ((gamma-1)/gamma)*ln_rho)
    pomega = np.exp(ln_pomega) # = R * T
    log = np.log
    HSE = gamma*pomega*(d3.grad(ln_rho) + d3.grad(s)/Cp) - g*ones
    N2 = -g@d3.grad(s)/Cp

    Fconv = dist.VectorField(coords, name='Fconv', bases=basis)
    Fconv['g'][2] = Lconv_func(r)/ (4*np.pi*r**2)
    N2_input['g'] = N2_func(r)
    g['g'][2] = g_func(r)

    # Problem
    problem = d3.NLBVP([ln_rho, s, Q, g_phi, tau_s, tau_rho, tau_g_phi], namespace=locals())
    problem.add_equation("grad(s)/Cp + grad(ln_rho) + rvec*lift(tau_rho) = g/(gamma*pomega)") #hydrostatic equilibrium
    problem.add_equation("N2 + lift(tau_s) = N2_input")
    problem.add_equation("Q = div(Fconv)")
    problem.add_equation("g*ones + grad(g_phi) + rvec*lift(tau_g_phi) = 0")

    problem.add_equation("ln_rho(r=Ro) = 0")
    problem.add_equation("ln_pomega(r=Ro) = log(R)")
    problem.add_equation("g_phi(r=Ro) = 0")

    ncc_cutoff=1e-10
    tolerance=1e-10
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')

    #Update stratification
    fig = plt.figure()
    ax1 = fig.add_subplot(3,2,1)
    ax2 = fig.add_subplot(3,2,2)
    ax3 = fig.add_subplot(3,2,3)
    ax4 = fig.add_subplot(3,2,4)
    ax5 = fig.add_subplot(3,2,5)
    ax6 = fig.add_subplot(3,2,6)
    ax1.plot(r.ravel(), ln_pomega.evaluate()['g'].ravel(), label='ln_pomega')
    ax1.plot(r.ravel(), ln_rho['g'].ravel(), label='ln_rho')
    ax1.legend()
    ax2.plot(r.ravel(), HSE.evaluate()['g'][2].ravel(), label='HSE')
    ax2.legend()
    ax3.plot(r.ravel(), g['g'][2].ravel(), label=r'$g$')
    ax3.legend()
    ax4.plot(r.ravel(), N2.evaluate()['g'].ravel(), label=r'$N^2$')
    ax4.plot(r.ravel(), -N2.evaluate()['g'].ravel(), label=r'$-N^2$')
    ax4.set_yscale('log')
    ax4.legend()
    ax5.plot(r.ravel(), Q['g'].ravel(), label='Q')
    ax5.legend()
    ax6.plot(r.ravel(), 4*np.pi*(r**2*Fconv['g'][2]).ravel(), label='Fconv')
    ax6.legend()
    fig.savefig('stratification.png', bbox_inches='tight', dpi=300)


    atmosphere = dict()
    atmosphere['r'] = r
    atmosphere['g'] = np.copy(g['g'])
    atmosphere['s0'] = np.copy(s['g'])
    atmosphere['grad_s0'] = np.copy(d3.grad(s).evaluate()['g'])
    atmosphere['grad_pom0'] = np.copy(d3.grad(pomega).evaluate()['g'])
    atmosphere['pom0'] = np.copy(pomega.evaluate()['g'])
    atmosphere['grad_ln_pom0'] = np.copy(d3.grad(ln_pomega).evaluate()['g'])
    atmosphere['grad_ln_rho0'] = np.copy(d3.grad(ln_rho).evaluate()['g'])
    atmosphere['rho0'] = np.copy(np.exp(ln_rho).evaluate()['g'])
    atmosphere['ln_rho0'] = np.copy(ln_rho['g'])
    atmosphere['Q'] = np.copy(Q['g'])
    atmosphere['g_phi'] = np.copy(g_phi['g'])
    return atmosphere


if __name__ == '__main__':
    epsilon = 1e-2
    N2_func = lambda r: 0*r
    g_func = lambda r: -r
    Lconv_func = lambda r: epsilon * r**3 * one_to_zero(r, 0.7*Ro, width=0.1*Ro)

    ball_HSE_BVP(N2_func, g_func, Lconv_func,  Nr=Nr, Ro=Ro, gamma=5/3, R=1)
