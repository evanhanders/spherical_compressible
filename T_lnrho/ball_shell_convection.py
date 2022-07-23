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
dealias = 3/2
timestepper = d3.SBDF2
dtype = np.float64
mesh = None
#mesh = [16,16]

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis = d3.BallBasis(coords, shape=(Nphi, Ntheta, Nr), radius=Ro, dealias=dealias, dtype=dtype)
basis2 = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ro, Ro2), dealias=dealias, dtype=dtype)
s2_basis = basis.S2_basis()
B2_s2_basis = basis2.S2_basis()

# Fields
ln_rho1 = dist.Field(name='ln_rho1', bases=basis)
T1 = dist.Field(name='T1', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)
tau_T1 = dist.Field(name='tau_T1', bases=s2_basis)
tau_T2 = dist.Field(name='tau_T2', bases=s2_basis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=s2_basis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=s2_basis)

grad_ln_rho0 = dist.VectorField(coords, name='grad_ln_rho0', bases=basis.radial_basis)
grad_ln_T0 = dist.VectorField(coords, name='grad_ln_T0', bases=basis.radial_basis)
grad_T0 = dist.VectorField(coords, name='grad_T0', bases=basis.radial_basis)
dS0_dr = dist.VectorField(coords, name='dS0_dr', bases=basis.radial_basis)
T0 = dist.Field(name='T0', bases=basis.radial_basis)
rho0 = dist.Field(name='rho0', bases=basis.radial_basis)
Q = dist.Field(name='Q', bases=basis)
eye = dist.TensorField(coords, name='I')
eye['g'] = 0
for i in range(3):
    eye['g'][i,i] = 1

B2_ln_rho1 = dist.Field(name='ln_rho1', bases=basis2)
B2_T1 = dist.Field(name='T1', bases=basis2)
B2_u = dist.VectorField(coords, name='u', bases=basis2)
B2_tau_T1 = dist.Field(name='tau_T1', bases=B2_s2_basis)
B2_tau_T2 = dist.Field(name='tau_T2', bases=B2_s2_basis)
B2_tau_u1 = dist.VectorField(coords, name='tau_u1', bases=B2_s2_basis)
B2_tau_u2 = dist.VectorField(coords, name='tau_u2', bases=B2_s2_basis)

B2_grad_ln_rho0 = dist.VectorField(coords, name='grad_ln_rho0', bases=basis2.radial_basis)
B2_grad_ln_T0 = dist.VectorField(coords, name='grad_ln_T0', bases=basis2.radial_basis)
B2_grad_T0 = dist.VectorField(coords, name='grad_T0', bases=basis2.radial_basis)
B2_dS0_dr = dist.VectorField(coords, name='dS0_dr', bases=basis2.radial_basis)
B2_T0 = dist.Field(name='T0', bases=basis2.radial_basis)
B2_rho0 = dist.Field(name='rho0', bases=basis2.radial_basis)
B2_Q = dist.Field(name='Q', bases=basis2)


# Substitutions
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
phi, theta, r = dist.local_grids(basis)
er = dist.VectorField(coords, bases=basis.radial_basis)
er['g'][2] = 1
rvec = dist.VectorField(coords, bases=basis.radial_basis)
rvec['g'][2] = r
B2_phi, B2_theta, B2_r = dist.local_grids(basis2)
B2_er = dist.VectorField(coords, bases=basis2.radial_basis)
B2_er['g'][2] = 1
B2_rvec = dist.VectorField(coords, bases=basis2.radial_basis)
B2_rvec['g'][2] = B2_r

lift_basis = basis.derivative_basis(0)
lift = lambda A: d3.Lift(A, lift_basis, -1)
B2_lift_basis = basis2.derivative_basis(2)
B2_lift = lambda A, n: d3.Lift(A, B2_lift_basis, n)
#grad_u = d3.grad(u) + rvec*lift(tau_u1) # First-order reduction
#grad_b = d3.grad(b) + rvec*lift(tau_b1) # First-order reduction

grad_u = d3.grad(u)
div_u = d3.div(u)

E = 0.5*(grad_u + d3.trans(grad_u))
sigma = 2*(E - (1/3)*div_u*eye)
viscous_diffusion_L = nu*(d3.div(sigma) + d3.dot(sigma, grad_ln_rho0))
viscous_diffusion_R = nu*d3.dot(sigma, d3.grad(ln_rho1))
VH = 2 * nu * (d3.trace(d3.dot(E,E)) - (1/3)*div_u*div_u)

thermal_diffusion_L = gamma*kappa*(d3.lap(T1) + d3.dot(d3.grad(T1), grad_ln_rho0))
thermal_diffusion_R = gamma*kappa*(d3.dot(d3.grad(T1), d3.grad(ln_rho1)))



B2_grad_u = d3.grad(B2_u)
B2_div_u = d3.div(B2_u)

B2_E = 0.5*(B2_grad_u + d3.trans(B2_grad_u))
B2_sigma = 2*(B2_E - (1/3)*B2_div_u*eye)
B2_viscous_diffusion_L = nu*(d3.div(B2_sigma) + d3.dot(B2_sigma, B2_grad_ln_rho0))
B2_viscous_diffusion_R = nu*d3.dot(B2_sigma, d3.grad(B2_ln_rho1))
B2_VH = 2 * nu * (d3.trace(d3.dot(B2_E,B2_E)) - (1/3)*B2_div_u*B2_div_u)


B2_thermal_diffusion_L = gamma*kappa*(d3.lap(B2_T1) + d3.dot(d3.grad(B2_T1), B2_grad_ln_rho0))
B2_thermal_diffusion_R = gamma*kappa*(d3.dot(d3.grad(B2_T1), d3.grad(B2_ln_rho1)))


Q['g'] = epsilon*one_to_zero(r, 0.5, width=0.1)
B2_Q['g'] = epsilon*one_to_zero(B2_r, 0.5, width=0.1)


#Analytical background state
#assume g_vec = -r & T = 1 at outer boundary.
A = -1/2
B = 3
T_func = lambda r: A * r**2 + B
grad_T_func = lambda r: 2 * A * r
grad_ln_T_func = lambda r: grad_T_func(r)/T_func(r)
grad_ln_rho_func = lambda r: (1/(gamma-1)) * grad_ln_T_func(r)
rho_func = lambda r: T_func(r)**(1/(gamma-1))
T0['g'] = T_func(r)
grad_T0['g'][2] = grad_T_func(r)
grad_ln_T0['g'][2] = grad_ln_T_func(r)
dS0_dr['g'][2] = 0
grad_ln_rho0['g'][2] = grad_ln_rho_func(r)
rho0['g'] = rho_func(r)


B2_T0['g'] = T_func(B2_r)
B2_grad_T0['g'][2] = grad_T_func(B2_r)
B2_grad_ln_T0['g'][2] = grad_ln_T_func(B2_r)
B2_dS0_dr['g'][2] = 0
B2_grad_ln_rho0['g'][2] = grad_ln_rho_func(B2_r)
B2_rho0['g'] = rho_func(B2_r)





################################################################################################
#BVP for background state
BVP_dist = d3.Distributor(coords, dtype=dtype, mesh=mesh, comm=MPI.COMM_SELF)
BVP_basis = d3.BallBasis(coords, shape=(1, 1, Nr), radius=Ro, dealias=1, dtype=dtype)
BVP_basis2 = d3.ShellBasis(coords, shape=(1, 1, Nr), radii=(Ro, Ro2), dealias=1, dtype=dtype)
BVP_s2_basis = basis.S2_basis()
BVP_B2_s2_basis = basis2.S2_basis()
BVP_phi, BVP_theta, BVP_r = BVP_dist.local_grids(BVP_basis)
BVP_B2_phi, BVP_B2_theta, BVP_B2_r = BVP_dist.local_grids(BVP_basis2)


# Parameters
BVP_lift = lambda A: d3.Lift(A, BVP_basis.derivative_basis(0), -1)
BVP_B2_lift = lambda A, n: d3.Lift(A, BVP_basis2.derivative_basis(2), n)

#gravity
BVP_dSdr_goal      = BVP_dist.VectorField(coords, name='dSdr_goal', bases=BVP_basis)
BVP_B2_dSdr_goal   = BVP_dist.VectorField(coords, name='dSdr_goal', bases=BVP_basis2)
BVP_dSdr_goal['g'][2] = zero_to_one(BVP_r, 0.8, width=0.1)
BVP_B2_dSdr_goal['g'][2] = zero_to_one(BVP_B2_r, 0.8, width=0.1)


BVP_g   = BVP_dist.VectorField(coords, name='g', bases=BVP_basis)
BVP_B2_g   = BVP_dist.VectorField(coords, name='g', bases=BVP_basis2)
BVP_g['g'][2] = -BVP_r
BVP_B2_g['g'][2] = -BVP_B2_r

BVP_r_vec = dist.VectorField(coords, bases=BVP_basis.radial_basis)
BVP_B2_r_vec = dist.VectorField(coords, bases=BVP_basis2.radial_basis)
BVP_r_vec['g'][2] = BVP_r
BVP_B2_r_vec['g'][2] = BVP_B2_r


BVP_T       = BVP_dist.Field(name='T', bases=BVP_basis)
BVP_ln_rho  = BVP_dist.Field(name='ln_rho', bases=BVP_basis)
BVP_tau_T   = BVP_dist.Field(name='tau_T', bases=BVP_s2_basis)
BVP_tau_rho = BVP_dist.Field(name='tau_rho', bases=BVP_s2_basis)
BVP_B2_T       = BVP_dist.Field(name='T', bases=BVP_basis2)
BVP_B2_ln_rho  = BVP_dist.Field(name='ln_rho', bases=BVP_basis2)
BVP_B2_tau_T   = BVP_dist.Field(name='tau_T', bases=BVP_B2_s2_basis)
BVP_B2_tau_rho = BVP_dist.Field(name='tau_rho', bases=BVP_B2_s2_basis)


BVP_ln_T  = np.log(BVP_T)
BVP_rho   = np.exp(BVP_ln_rho)
BVP_dS_dr = Cp * ((1/gamma) * d3.grad(BVP_ln_T) - ((gamma-1)/gamma)*d3.grad(BVP_ln_rho))
BVP_N2_op = -BVP_g@BVP_dS_dr/Cp
BVP_B2_ln_T  = np.log(BVP_B2_T)
BVP_B2_rho   = np.exp(BVP_B2_ln_rho)
BVP_B2_dS_dr = Cp * ((1/gamma) * d3.grad(BVP_B2_ln_T) - ((gamma-1)/gamma)*d3.grad(BVP_B2_ln_rho))
BVP_B2_N2_op = -BVP_B2_g@BVP_B2_dS_dr/Cp

#TODO: go from here
BVP_T['g']   = -(BVP_r**2 - Ro2**2) + 1
BVP_B2_T['g']   = -(BVP_B2_r**2 - Ro2**2) + 1

BVP_ln_rho['g'] = (1/(gamma-1))*np.log(BVP_T['g'])
BVP_B2_ln_rho['g'] = (1/(gamma-1))*np.log(BVP_B2_T['g'])

pi = np.pi

variables = [BVP_T, BVP_ln_rho, BVP_B2_T, BVP_B2_ln_rho, BVP_tau_T, BVP_tau_rho, BVP_B2_tau_T, BVP_B2_tau_rho]

problem = d3.NLBVP(variables, namespace=locals())

print(locals().keys())


er = dist.VectorField(coords)
er['g'][2] = 1

problem.add_equation("- R*(grad(BVP_T)) + er*BVP_lift(BVP_tau_T) = + R*BVP_T*grad(BVP_ln_rho) - BVP_g")
problem.add_equation("-R*BVP_r_vec@grad(BVP_ln_rho) - BVP_lift(BVP_tau_rho) = -(Cp/gamma)*BVP_r_vec@grad(BVP_ln_rho) + BVP_r_vec@BVP_dSdr_goal")
problem.add_equation("- R*(grad(BVP_B2_T)) + BVP_B2_r_vec*BVP_B2_lift(BVP_B2_tau_T, -1) = + R*BVP_B2_T*grad(BVP_B2_ln_rho) - BVP_B2_g")
problem.add_equation("-R*BVP_B2_r_vec@grad(BVP_B2_ln_rho) - BVP_B2_lift(BVP_B2_tau_rho, -1) = -(Cp/gamma)*BVP_B2_r_vec@grad(BVP_B2_ln_rho) + BVP_B2_r_vec@BVP_B2_dSdr_goal")

problem.add_equation("BVP_T(r=Ro) - BVP_B2_T(r=Ro) = 0")
problem.add_equation("BVP_ln_rho(r=Ro) - BVP_B2_ln_rho(r=Ro) = 0")
problem.add_equation("BVP_B2_T(r=Ro2) = 1")
problem.add_equation("BVP_B2_ln_rho(r=Ro2) = 0")


ncc_cutoff=1e-10
tolerance=1e-10
solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
pert_norm = np.inf
while pert_norm > tolerance:
    solver.newton_iteration(damping=1)
    pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
    logger.info(f'Perturbation norm: {pert_norm:.3e}')
    plt.plot(BVP_r.ravel(), BVP_N2_op_B.evaluate()['g'].ravel())
    plt.yscale('log')
    plt.show()

#rs = []
#gs = []
#N2s = []
#HSEs = []
#ln_Ts = []
#Ts = []
#ln_rhos = []
#dS_drs = []
#
#for k, basis in bases.items():
#    ln_T = namespace['ln_T_{}'.format(k)].evaluate()
#    T = namespace['T_{}'.format(k)].evaluate()
#    ln_rho = namespace['ln_rho_{}'.format(k)]
#    dS_dr = namespace['dS_dr_{}'.format(k)].evaluate()
#    g = namespace['g_{}'.format(k)]
#    N2_op = namespace['N2_op_{}'.format(k)]
#    N2 = N2_op.evaluate()
#
#    grad_ln_T = d3.grad(ln_T).evaluate()
#    grad_ln_rho = d3.grad(ln_rho).evaluate()
#    HSE = (-R*(d3.grad(ln_T) + d3.grad(ln_rho)) + g/T).evaluate()
#
#    phi, theta, r = dist.local_grids(basis, scales=(1,1,scales))
#    dS_dr.change_scales((1,1,scales))
#    ln_T.change_scales((1,1,scales))
#    T.change_scales((1,1,scales))
#    ln_rho.change_scales((1,1,scales))
#    N2.change_scales((1,1,scales))
#    HSE.change_scales((1,1,scales))
#    g.change_scales((1,1,scales))
#
#    rs.append(r)
#    gs.append(g['g'])
#    N2s.append(N2['g'])
#    ln_Ts.append(ln_T['g'])
#    Ts.append(T['g'])
#    ln_rhos.append(ln_rho['g'])
#    dS_drs.append(dS_dr['g'])
#    HSEs.append(HSE['g'])
#
#r = np.concatenate(rs, axis=-1)
#g = np.concatenate(gs, axis=-1)
#N2 = np.concatenate(N2s, axis=-1)
#HSE = np.concatenate(HSEs, axis=-1)
#ln_T = np.concatenate(ln_Ts, axis=-1)
#T = np.concatenate(Ts, axis=-1)
#ln_rho = np.concatenate(ln_rhos, axis=-1)
#dS_dr = np.concatenate(dS_drs, axis=-1)
#
#fig = plt.figure()
#ax1 = fig.add_subplot(4,1,1)
#ax2 = fig.add_subplot(4,1,2)
#ax3 = fig.add_subplot(4,1,3)
#ax4 = fig.add_subplot(4,1,4)
#ax1.plot(r.ravel(), ln_T.ravel(), label='ln_T')
#ax1.plot(r.ravel(), ln_rho.ravel(), label='ln_rho')
#ax1.legend()
#ax2.plot(r.ravel(), HSE[2,:].ravel(), label='HSE')
#ax2.legend()
#ax3.plot(r.ravel(), g[2,:].ravel(), label=r'$g$')
#ax3.legend()
#ax4.plot(r.ravel(), N2.ravel(), label=r'$N^2$')
#ax4.plot(r.ravel(), -N2.ravel(), label=r'$-N^2$')
#ax4.plot(r.ravel(), (N2_func(r)).ravel(), label=r'$N^2$ goal', ls='--')
#ax4.set_yscale('log')
#yticks = (np.max(np.abs(N2.ravel()[r.ravel() < 0.5])), np.max(N2_func(r).ravel()))
#ax4.set_yticks(yticks)
#ax4.set_yticklabels(['{:.1e}'.format(n) for n in yticks])
#ax4.legend()
#fig.savefig('stratification.png', bbox_inches='tight', dpi=300)
##    plt.show()
#
#
#atmosphere = dict()
#atmosphere['r'] = r
#atmosphere['ln_T'] = ln_T
#atmosphere['T'] = T
#atmosphere['N2'] = N2
#atmosphere['ln_rho'] = ln_rho
#atmosphere['dS_dr'] = dS_dr


# Problem
problem = d3.IVP([ln_rho1, T1, u, B2_ln_rho1, B2_T1, B2_u, tau_T1, tau_u1, B2_tau_T1, B2_tau_T2, B2_tau_u1, B2_tau_u2], namespace=locals())
problem.add_equation("dt(ln_rho1) + div_u + u@grad_ln_rho0 = -u@grad(ln_rho1)")
problem.add_equation("dt(T1) + T0*(gamma-1)*div_u + dot(u, grad_T0) - thermal_diffusion_L + lift(tau_T1) = - u@grad(T1) - T1*(gamma-1)*div_u + (1/Cv)*VH + thermal_diffusion_R + (1/rho0)*kappa*Q/Cv")
problem.add_equation("dt(u) - viscous_diffusion_L + R*(grad(T1) + T1*grad_ln_rho0 + T0*grad(ln_rho1)) + lift(tau_u1) = - u@grad(u) + -R*T1*grad(ln_rho1) + viscous_diffusion_R")
problem.add_equation("dt(B2_ln_rho1) + B2_div_u + B2_u@B2_grad_ln_rho0 + (1/nu)*B2_rvec@B2_lift(B2_tau_u2, -1) = -B2_u@grad(B2_ln_rho1)")
problem.add_equation("dt(B2_T1) + B2_T0*(gamma-1)*B2_div_u + dot(B2_u, B2_grad_T0) - B2_thermal_diffusion_L + B2_lift(B2_tau_T1, -1) + B2_lift(B2_tau_T2, -2) = - B2_u@grad(B2_T1) - B2_T1*(gamma-1)*B2_div_u + (1/Cv)*B2_VH + B2_thermal_diffusion_R + (1/B2_rho0)*kappa*B2_Q/Cv")
problem.add_equation("dt(B2_u) - B2_viscous_diffusion_L + R*(grad(B2_T1) + B2_T1*B2_grad_ln_rho0 + B2_T0*grad(B2_ln_rho1)) + B2_lift(B2_tau_u1, -1) + B2_lift(B2_tau_u2, -2) = - B2_u@grad(B2_u) + -R*B2_T1*grad(B2_ln_rho1) + B2_viscous_diffusion_R")
#problem.add_equation("dt(ln_rho1) + div_u + u@grad_ln_rho0 + rvec@lift(tau_u2, -1) = -u@grad(ln_rho1)")
#problem.add_equation("dt(T1) + T0*(gamma-1)*div_u + dot(u, grad_T0) - kappa*lap(T1) + lift(tau_T1, -1) + kappa*lift(tau_T2, -2) = - u@grad(T1) - T1*(gamma-1)*div_u + (1/Cv)*VH")
#problem.add_equation("dt(u) - viscous_diffusion_L + R*(grad(T1) + T1*grad_ln_rho0 + T0*grad(ln_rho1)) + lift(tau_u1, -1) + nu*lift(tau_u2, -2) = - u@grad(u) + -R*T1*grad(ln_rho1) + viscous_diffusion_R")
#problem.add_equation("dt(B2_ln_rho1) + B2_div_u + B2_u@B2_grad_ln_rho0 + B2_rvec@B2_lift(B2_tau_u2, -1) = -B2_u@grad(B2_ln_rho1)")
#problem.add_equation("dt(B2_T1) + B2_T0*(gamma-1)*B2_div_u + dot(B2_u, B2_grad_T0) - kappa*lap(B2_T1) + B2_lift(B2_tau_T1, -1) + kappa*B2_lift(B2_tau_T2, -2) = - B2_u@grad(B2_T1) - B2_T1*(gamma-1)*B2_div_u + (1/Cv)*B2_VH")
#problem.add_equation("dt(B2_u) - B2_viscous_diffusion_L + R*(grad(B2_T1) + B2_T1*B2_grad_ln_rho0 + B2_T0*grad(B2_ln_rho1)) + B2_lift(B2_tau_u1, -1) + nu*B2_lift(B2_tau_u2, -2) = - B2_u@grad(B2_u) + -R*B2_T1*grad(B2_ln_rho1) + B2_viscous_diffusion_R")

#problem.add_equation("T1(r=Ri) = 0")
#problem.add_equation("radial(u(r=Ri)) = 0")
#problem.add_equation("angular(radial(E(r=Ri))) = 0")

problem.add_equation("B2_T1(r=Ro2) = 0")
#problem.add_equation("radial(grad(B2_T1)(r=Ro2)) = -(Ro2**2/3)*epsilon")
problem.add_equation("radial(B2_u(r=Ro2)) = 0")
problem.add_equation("angular(radial(B2_E(r=Ro2))) = 0")

problem.add_equation("T1(r=Ro) - B2_T1(r=Ro) = 0")
problem.add_equation("u(r=Ro) - B2_u(r=Ro) = 0")
problem.add_equation("angular(radial(sigma(r=Ro) - B2_sigma(r=Ro)))= 0")
problem.add_equation("ln_rho1(r=Ro) - B2_ln_rho1(r=Ro) =  0")
problem.add_equation("radial(grad(T1)(r=Ro) - grad(B2_T1)(r=Ro)) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
T1.fill_random('g', seed=42, distribution='normal', scale=1e-6*epsilon) # Random noise
T1['g'] *= r**2
B2_T1.fill_random('g', seed=42, distribution='normal', scale=1e-6*epsilon) # Random noise
B2_T1['g'] *= B2_r**2

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=t_buoy, max_writes=10)
snapshots.add_task(T1(theta=np.pi/2), name='T1_eq')
snapshots.add_task(T1(phi=0), name='T1(phi=0)')
snapshots.add_task(T1(phi=np.pi), name='T1(phi=pi)')
snapshots.add_task(u(phi=0), name='u(phi=0)')
snapshots.add_task(u(phi=np.pi), name='u(phi=pi)')
snapshots.add_task(B2_T1(theta=np.pi/2), name='B2_T1_eq')
snapshots.add_task(B2_T1(phi=0), name='B2_T1(phi=0)')
snapshots.add_task(B2_T1(phi=np.pi), name='B2_T1(phi=pi)')
snapshots.add_task(B2_u(phi=0), name='B2_u(phi=0)')
snapshots.add_task(B2_u(phi=np.pi), name='B2_u(phi=pi)')


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
CFL.add_velocity(B2_u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(u@u)/nu, name='Re')
flow.add_property(np.sqrt(B2_u@B2_u)/nu, name='B2_Re')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_Re = flow.max('Re')
            B2_max_Re = flow.max('B2_Re')
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, np.max((max_Re, B2_max_Re))))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
