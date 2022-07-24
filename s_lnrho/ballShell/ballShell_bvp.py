import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)




# Parameters
Ri, Ro = 1.05, 2
Nphi, Ntheta, NrB, NrS = 1, 1, 32, 32
Rayleigh = 1e3
Prandtl = 1
epsilon = 1e-2
kappa = epsilon*(Rayleigh * Prandtl)**(-1/2)
nu = epsilon*(Rayleigh / Prandtl)**(-1/2)
gamma = 5/3
R = 1
Cv = R * 1 / (gamma-1)
Cp = R + Cv

def ballShell_HSE_BVP(N2_func, g_func, Lconv_func,  Nrs=(NrB, NrS), radii=(Ri, Ro),gamma=5/3, R=1):
    B1_Nr, B2_Nr = Nrs
    Ri, Ro = radii
    log = np.log

    # Bases
    coords = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist = d3.Distributor(coords, dtype=np.float64, mesh=None, comm=MPI.COMM_SELF)
    B1_basis = d3.BallBasis(coords, shape=(1, 1, B1_Nr), radius=Ri, dealias=1, dtype=np.float64)
    B1_s2_basis = B1_basis.S2_basis()
    B1_phi, B1_theta, B1_r = dist.local_grids(B1_basis)
    B1_phi_de, B1_theta_de, B1_r_de = dist.local_grids(B1_basis, scales=B1_basis.dealias)
    B2_basis = d3.ShellBasis(coords, shape=(1, 1, B2_Nr), radii=radii, dealias=1, dtype=np.float64)
    B2_s2_basis = B2_basis.S2_basis()
    B2_phi, B2_theta, B2_r = dist.local_grids(B2_basis)
    B2_phi_de, B2_theta_de, B2_r_de = dist.local_grids(B2_basis, scales=B2_basis.dealias)

    # Fields
    B1_ln_rho = dist.Field(name='ln_rho', bases=B1_basis)
    B1_s = dist.Field(name='s', bases=B1_basis)
    B1_Q = dist.Field(name='Q', bases=B1_basis)
    B1_tau_s = dist.Field(name='tau_s', bases=B1_s2_basis)
    B1_tau_Q = dist.Field(name='tau_Q', bases=B1_s2_basis)
    B1_tau_rho = dist.Field(name='tau_rho', bases=B1_s2_basis)
    B1_N2_input = dist.Field(name='N2_input', bases=B1_basis)
    B2_ln_rho = dist.Field(name='ln_rho', bases=B2_basis)
    B2_s = dist.Field(name='s', bases=B2_basis)
    B2_Q = dist.Field(name='Q', bases=B2_basis)
    B2_tau_s = dist.Field(name='tau_s', bases=B2_s2_basis)
    B2_tau_Q = dist.Field(name='tau_Q', bases=B2_s2_basis)
    B2_tau_rho = dist.Field(name='tau_rho', bases=B2_s2_basis)
    B2_N2_input = dist.Field(name='N2_input', bases=B2_basis)


    B1_grad_pom0 = dist.VectorField(coords, name='grad_pom0', bases=B1_basis.radial_basis)
    B1_grad_s0 = dist.VectorField(coords, name='grad_s0', bases=B1_basis.radial_basis)
    B1_pom0 = dist.Field(name='pom0', bases=B1_basis.radial_basis)
    B1_g = dist.VectorField(coords, name='g', bases=B1_basis.radial_basis)
    B1_Q = dist.Field(name='Q', bases=B1_basis)
    B1_ones = dist.Field(name='ones', bases=B1_basis)
    B1_ones['g'] = 1
    B2_grad_pom0 = dist.VectorField(coords, name='grad_pom0', bases=B2_basis.radial_basis)
    B2_grad_s0 = dist.VectorField(coords, name='grad_s0', bases=B2_basis.radial_basis)
    B2_pom0 = dist.Field(name='pom0', bases=B2_basis.radial_basis)
    B2_g = dist.VectorField(coords, name='g', bases=B2_basis.radial_basis)
    B2_Q = dist.Field(name='Q', bases=B2_basis)
    B2_ones = dist.Field(name='ones', bases=B2_basis)
    B2_ones['g'] = 1


    # Substitutions
    Cv = R / (gamma-1)
    Cp = R + Cv
    B1_er = dist.VectorField(coords, bases=B1_basis.radial_basis)
    B1_er['g'][2] = 1
    B1_rvec = dist.VectorField(coords, bases=B1_basis.radial_basis)
    B1_rvec['g'][2] = B1_r
    B2_er = dist.VectorField(coords, bases=B2_basis.radial_basis)
    B2_er['g'][2] = 1
    B2_rvec = dist.VectorField(coords, bases=B2_basis.radial_basis)
    B2_rvec['g'][2] = B2_r

    B1_lift_basis = B1_basis.derivative_basis(0)
    B1_lift = lambda A: d3.Lift(A, B1_lift_basis, -1)
    B2_lift_basis = B2_basis.derivative_basis(2)
    B2_lift = lambda A: d3.Lift(A, B2_lift_basis, -1)

    B1_ln_pomega = gamma*(B1_s/Cp + ((gamma-1)/gamma)*B1_ln_rho)
    B1_pomega = np.exp(B1_ln_pomega) # = R * T
    B1_HSE = gamma*B1_pomega*(d3.grad(B1_ln_rho) + d3.grad(B1_s)/Cp) - B1_g*B1_ones
    B1_N2 = -B1_g@d3.grad(B1_s)/Cp
    B1_Fconv = dist.VectorField(coords, name='Fconv', bases=B1_basis)
    B1_Fconv['g'][2] = Lconv_func(B1_r)/ (4*np.pi*B1_r**2)
    B1_N2_input['g'] = N2_func(B1_r)
    B1_g['g'][2] = g_func(B1_r)
    B2_ln_pomega = gamma*(B2_s/Cp + ((gamma-1)/gamma)*B2_ln_rho)
    B2_pomega = np.exp(B2_ln_pomega) # = R * T
    B2_HSE = gamma*B2_pomega*(d3.grad(B2_ln_rho) + d3.grad(B2_s)/Cp) - B2_g*B2_ones
    B2_N2 = -B2_g@d3.grad(B2_s)/Cp
    B2_Fconv = dist.VectorField(coords, name='Fconv', bases=B2_basis)
    B2_Fconv['g'][2] = Lconv_func(B2_r)/ (4*np.pi*B2_r**2)
    B2_N2_input['g'] = N2_func(B2_r)
    B2_g['g'][2] = g_func(B2_r)


    # Problem
    problem = d3.NLBVP([B1_ln_rho, B1_s, B1_Q, B2_ln_rho, B2_s, B2_Q, B1_tau_s, B1_tau_rho, B2_tau_s, B2_tau_rho], namespace=locals())
    problem.add_equation("grad(B1_s)/Cp + grad(B1_ln_rho) + B1_rvec*B1_lift(B1_tau_rho) = B1_g/(gamma*B1_pomega)") #hydrostatic equilibrium
    problem.add_equation("B1_N2 + B1_lift(B1_tau_s) = B1_N2_input")
    problem.add_equation("B1_Q = div(B1_Fconv)") 
    problem.add_equation("grad(B2_s)/Cp + grad(B2_ln_rho) + B2_rvec*B2_lift(B2_tau_rho) = B2_g/(gamma*B2_pomega)") #hydrostatic equilibrium
    problem.add_equation("B2_N2 + B2_lift(B2_tau_s) = B2_N2_input")
    problem.add_equation("B2_Q = div(B2_Fconv)") 

    problem.add_equation("B1_s(r=Ri) - B2_s(r=Ri) = 0")
    problem.add_equation("B1_ln_rho(r=Ri) - B2_ln_rho(r=Ri) = 0")
    problem.add_equation("B1_ln_rho(r=1) = 0")
    problem.add_equation("B1_ln_pomega(r=1) = log(R)")

    ncc_cutoff=1e-10
    tolerance=1e-10
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')

    r = np.concatenate((B1_r.ravel(), B2_r.ravel()))
    ln_pomega = np.concatenate((B1_ln_pomega.evaluate()['g'].ravel(), B2_ln_pomega.evaluate()['g'].ravel()))
    ln_rho = np.concatenate((B1_ln_rho['g'].ravel(), B2_ln_rho['g'].ravel()))
    g = np.concatenate((B1_g['g'][2].ravel(), B2_g['g'][2].ravel()))
    N2 = np.concatenate((B1_N2.evaluate()['g'].ravel(), B2_N2.evaluate()['g'].ravel()))
    HSE = np.concatenate((B1_HSE.evaluate()['g'][2].ravel(), B2_HSE.evaluate()['g'][2].ravel()))
    Q = np.concatenate((B1_Q['g'].ravel(), B2_Q['g'].ravel()))
    Lconv = 4*np.pi*np.concatenate(((B1_r**2*B1_Fconv['g'][2]).ravel(), (B2_r**2*B2_Fconv['g'][2]).ravel()))
    grad_s = np.concatenate((d3.grad(B1_s).evaluate()['g'][2].ravel(), d3.grad(B2_s).evaluate()['g'][2].ravel()))
    grad_pomega = np.concatenate((d3.grad(B1_pomega).evaluate()['g'][2].ravel(), d3.grad(B2_pomega).evaluate()['g'][2].ravel()))
    pomega = np.concatenate((B1_pomega.evaluate()['g'].ravel(), B2_pomega.evaluate()['g'].ravel()))
    grad_ln_pomega = np.concatenate((d3.grad(B1_ln_pomega).evaluate()['g'][2].ravel(), d3.grad(B2_ln_pomega).evaluate()['g'][2].ravel()))
    grad_ln_rho = np.concatenate((d3.grad(B1_ln_rho).evaluate()['g'][2].ravel(), d3.grad(B2_ln_rho).evaluate()['g'][2].ravel()))
    rho = np.concatenate((np.exp(B1_ln_rho).evaluate()['g'].ravel(), np.exp(B2_ln_rho).evaluate()['g'].ravel()))

    #Update stratification
    fig = plt.figure()
    ax1 = fig.add_subplot(3,2,1)
    ax2 = fig.add_subplot(3,2,2)
    ax3 = fig.add_subplot(3,2,3)
    ax4 = fig.add_subplot(3,2,4)
    ax5 = fig.add_subplot(3,2,5)
    ax6 = fig.add_subplot(3,2,6)
    ax1.plot(r, ln_pomega, label='ln_pomega')
    ax1.plot(r, ln_rho, label='ln_rho')
    ax1.legend()
    ax2.plot(r, HSE, label='HSE')
    ax2.legend()
    ax3.plot(r, g, label=r'$g$')
    ax3.legend()
    ax4.plot(r, N2, label=r'$N^2$')
    ax4.plot(r, -N2, label=r'$-N^2$')
    ax4.set_yscale('log')
    ax4.legend()
    ax5.plot(r, Q, label='Q')
    ax5.legend()
    ax6.plot(r, Lconv, label='Lconv')
    ax6.legend()
    fig.savefig('stratification.png', bbox_inches='tight', dpi=300)


    interp_kwargs = { 'fill_value' : 'extrapolate', 'bounds_error' : False }
    atmosphere = dict()
    atmosphere['r'] = r
    atmosphere['g'] = interp1d(r, g, **interp_kwargs)
    atmosphere['grad_s0'] = interp1d(r, grad_s, **interp_kwargs)
    atmosphere['grad_pom0'] = interp1d(r, grad_pomega, **interp_kwargs)
    atmosphere['pom0'] = interp1d(r, pomega, **interp_kwargs) 
    atmosphere['grad_ln_pom0'] = interp1d(r, grad_ln_pomega, **interp_kwargs)
    atmosphere['grad_ln_rho0'] = interp1d(r, grad_ln_rho, **interp_kwargs)
    atmosphere['rho0'] = interp1d(r, rho, **interp_kwargs)
    atmosphere['ln_rho0'] = interp1d(r, ln_rho, **interp_kwargs)
    atmosphere['Q'] = interp1d(r, Q, **interp_kwargs)
    return atmosphere


if __name__ == '__main__':
    epsilon = 1e-2
    N2_func = lambda r: zero_to_one(r, 0.8*Ri, width=0.1*Ri)
    g_func = lambda r: -r
    Lconv_func = lambda r: epsilon * r**3 * one_to_zero(r, 0.7*Ri, width=0.1*Ri)

    ballShell_HSE_BVP(N2_func, g_func, Lconv_func, Nrs=(NrB, NrS), radii=(Ri,Ro), gamma=5/3, R=1)
