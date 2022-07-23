"""
Dedalus script simulating Boussinesq convection in a spherical shell. This script
demonstrates solving an initial value problem in the shell. It can be ran serially
or in parallel, and uses the built-in analysis framework to save data snapshots
to HDF5 files. The `plot_shell.py` script can be used to produce plots from the
saved data. The simulation should take about 10 cpu-minutes to run.

The problem is non-dimensionalized using the shell thickness and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

For incompressible hydro with two boundaries, we need two tau terms for each the
velocity and buoyancy. Here we choose to use a first-order formulation, putting
one tau term each on auxiliary first-order gradient variables and the others in
the PDE, and lifting them all to the first derivative basis. This formulation puts
a tau term in the divergence constraint, as required for this geometry.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shell_convection.py
    $ mpiexec -n 4 python3 plot_shell.py snapshots/*.h5
"""

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
Ri, Ro, Ro2 = 0, 1, 2
Nphi, Ntheta, Nr = 1, 32, 32
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

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh, comm=MPI.COMM_SELF)
basis = d3.BallBasis(coords, shape=(1, 1, Nr), radius=Ro, dealias=1, dtype=dtype)
basis2 = d3.ShellBasis(coords, shape=(1, 1, Nr), radii=(Ro, Ro2), dealias=1, dtype=dtype)
s2_basis = basis.S2_basis()
B2_s2_basis = basis2.S2_basis()
phi, theta, r = dist.local_grids(basis)
B2_phi, B2_theta, B2_r = dist.local_grids(basis2)


# Parameters
lift = lambda A: d3.Lift(A, basis.derivative_basis(0), -1)
B2_lift = lambda A, n: d3.Lift(A, basis2.derivative_basis(2), n)

#gravity
dsdr_floor = -1e-8
dSdr_goal      = dist.VectorField(coords, name='dSdr_goal', bases=basis)
B2_dSdr_goal   = dist.VectorField(coords, name='dSdr_goal', bases=basis2)
dSdr_goal['g'][2] = dsdr_floor + zero_to_one(r, 0.8, width=0.1)
B2_dSdr_goal['g'][2] = dsdr_floor + zero_to_one(B2_r, 0.8, width=0.1)


g   = dist.VectorField(coords, name='g', bases=basis)
B2_g   = dist.VectorField(coords, name='g', bases=basis2)
g['g'][2] = -r
B2_g['g'][2] = -B2_r

r_vec = dist.VectorField(coords, bases=basis.radial_basis)
B2_r_vec = dist.VectorField(coords, bases=basis2.radial_basis)
r_vec['g'][2] = r
B2_r_vec['g'][2] = B2_r


T       = dist.Field(name='T', bases=basis)
ln_rho  = dist.Field(name='ln_rho', bases=basis)
tau_T   = dist.Field(name='tau_T', bases=s2_basis)
tau_rho = dist.Field(name='tau_rho', bases=s2_basis)
B2_T       = dist.Field(name='T', bases=basis2)
B2_ln_rho  = dist.Field(name='ln_rho', bases=basis2)
B2_tau_T   = dist.Field(name='tau_T', bases=B2_s2_basis)
B2_tau_rho = dist.Field(name='tau_rho', bases=B2_s2_basis)


ln_T  = np.log(T)
rho   = np.exp(ln_rho)
dS_dr = Cp * ((1/gamma) * d3.grad(ln_T) - ((gamma-1)/gamma)*d3.grad(ln_rho))
N2_op = -g@dS_dr/Cp
grad_ln_T = d3.grad(ln_T)
grad_ln_rho = d3.grad(ln_rho)
HSE = (-R*(d3.grad(ln_T) + d3.grad(ln_rho)) + g/T)

B2_ln_T  = np.log(B2_T)
B2_rho   = np.exp(B2_ln_rho)
B2_dS_dr = Cp * ((1/gamma) * d3.grad(B2_ln_T) - ((gamma-1)/gamma)*d3.grad(B2_ln_rho))
B2_N2_op = -B2_g@B2_dS_dr/Cp
B2_grad_ln_T = d3.grad(B2_ln_T)
B2_grad_ln_rho = d3.grad(B2_ln_rho)
B2_HSE = (-R*(B2_grad_ln_T + B2_grad_ln_rho) + B2_g/B2_T)

#TODO: go from here
T['g']   = -(r**2 - Ro2**2) + 1
B2_T['g']   = -(B2_r**2 - Ro2**2) + 1

ln_rho['g'] = (1/(gamma-1))*np.log(T['g'])
B2_ln_rho['g'] = (1/(gamma-1))*np.log(B2_T['g'])

pi = np.pi

variables = [T, ln_rho, B2_T, B2_ln_rho, tau_T, tau_rho, B2_tau_T, B2_tau_rho]

problem = d3.NLBVP(variables, namespace=locals())

er = dist.VectorField(coords)
er['g'][2] = 1

problem.add_equation("- R*(grad(T)) + r_vec*lift(tau_T) = + R*T*grad(ln_rho) - g")
problem.add_equation("R*r_vec@grad(ln_rho) + lift(tau_rho) = (Cp/gamma)*r_vec@grad(ln_T) - r_vec@dSdr_goal")
problem.add_equation("- R*(grad(B2_T)) + B2_r_vec*B2_lift(B2_tau_T, -1) = + R*B2_T*grad(B2_ln_rho) - B2_g")
problem.add_equation("R*B2_r_vec@grad(B2_ln_rho) + B2_lift(B2_tau_rho, -1) = (Cp/gamma)*B2_r_vec@grad(B2_ln_T) - B2_r_vec@B2_dSdr_goal")

problem.add_equation("T(r=Ro) - B2_T(r=Ro) = 0")
problem.add_equation("ln_rho(r=Ro) - B2_ln_rho(r=Ro) = 0")
problem.add_equation("B2_T(r=Ro2) = 1")
problem.add_equation("B2_ln_rho(r=Ro2) = 0")


ncc_cutoff=1e-10
tolerance=1e-10
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
pert_norm = np.inf
while pert_norm > tolerance:
    solver.newton_iteration(damping=1)
    pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    logger.info(f'Perturbation norm: {pert_norm:.3e}')
#    plt.plot(r.ravel(), dSdr_goal['g'][2].ravel(), c='k')
#    plt.plot(B2_r.ravel(), B2_dSdr_goal['g'][2].ravel(), c='k')
#    plt.plot(r.ravel(), dS_dr.evaluate()['g'][2].ravel())
#    plt.plot(r.ravel(), -dS_dr.evaluate()['g'][2].ravel(), ls='--')
#    plt.yscale('log')
#    plt.show()

#Update stratification
ln_T.evaluate()
B2_ln_T.evaluate()
ln_rho.evaluate()
B2_ln_rho.evaluate()

#stitch together stratification
rs = [r.ravel(), B2_r.ravel()]
gs = [g['g'][2].ravel(), B2_g['g'][2].ravel()]
N2s = [N2_op.evaluate()['g'].ravel(), B2_N2_op.evaluate()['g'].ravel()]
HSEs = [HSE.evaluate()['g'][2].ravel(), B2_HSE.evaluate()['g'][2].ravel()]
ln_Ts = [np.copy(ln_T.evaluate()['g'].ravel()), np.copy(B2_ln_T.evaluate()['g'].ravel())]
Ts = [np.copy(T['g'].ravel()), np.copy(B2_T['g'].ravel())]
ln_rhos = [np.copy(ln_rho['g'].ravel()), np.copy(B2_ln_rho['g'].ravel())]
dS_drs = [dS_dr.evaluate()['g'][2].ravel(), B2_dS_dr.evaluate()['g'][2].ravel()]

r = np.concatenate(rs, axis=-1)
g = np.concatenate(gs, axis=-1)
N2 = np.concatenate(N2s, axis=-1)
HSE = np.concatenate(HSEs, axis=-1)
ln_T = np.concatenate(ln_Ts, axis=-1)
T = np.concatenate(Ts, axis=-1)
ln_rho = np.concatenate(ln_rhos, axis=-1)
dS_dr = np.concatenate(dS_drs, axis=-1)

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)
ax1.plot(r.ravel(), ln_T.ravel(), label='ln_T')
ax1.plot(r.ravel(), ln_rho.ravel(), label='ln_rho')
ax1.legend()
ax2.plot(r.ravel(), HSE.ravel(), label='HSE')
ax2.legend()
ax3.plot(r.ravel(), g.ravel(), label=r'$g$')
ax3.legend()
ax4.plot(r.ravel(), N2.ravel(), label=r'$N^2$')
ax4.plot(r.ravel(), -N2.ravel(), label=r'$-N^2$')
#ax4.plot(r.ravel(), (N2_func(r)).ravel(), label=r'$N^2$ goal', ls='--')
ax4.set_yscale('log')
#yticks = (np.max(np.abs(N2.ravel()[r.ravel() < 0.5])), np.max(N2_func(r).ravel()))
#ax4.set_yticks(yticks)
#ax4.set_yticklabels(['{:.1e}'.format(n) for n in yticks])
ax4.legend()
fig.savefig('stratification.png', bbox_inches='tight', dpi=300)
#    plt.show()


atmosphere = dict()
atmosphere['r'] = r
atmosphere['ln_T'] = ln_T
atmosphere['T'] = T
atmosphere['N2'] = N2
atmosphere['ln_rho'] = ln_rho
atmosphere['dS_dr'] = dS_dr



