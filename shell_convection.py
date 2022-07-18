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
import logging
logger = logging.getLogger(__name__)


# Parameters
Ri, Ro, Ro2 = 1, 2, 3
Nphi, Ntheta, Nr = 1, 16, 16 
Rayleigh = 3500
Prandtl = 1
dealias = 3/2
timestepper = d3.SBDF2
dtype = np.float64
mesh = None

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis = d3.ShellBasis(coords, shape=(Nphi, Ntheta, Nr), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
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

lift_basis = basis.derivative_basis(2)
lift = lambda A, n: d3.Lift(A, lift_basis, n)
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


B2_grad_u = d3.grad(B2_u)
B2_div_u = d3.div(B2_u)

B2_E = 0.5*(B2_grad_u + d3.trans(B2_grad_u))
B2_sigma = 2*(B2_E - (1/3)*B2_div_u*eye)
B2_viscous_diffusion_L = nu*(d3.div(B2_sigma) + d3.dot(B2_sigma, B2_grad_ln_rho0))
B2_viscous_diffusion_R = nu*d3.dot(B2_sigma, d3.grad(B2_ln_rho1))
B2_VH = 2 * nu * (d3.trace(d3.dot(B2_E,B2_E)) - (1/3)*B2_div_u*B2_div_u)

#solve for hydrostatic equilibrium background state
#assume g_vec = -r
T0['g'] = -1*r + (Cp/epsilon)
grad_T0['g'][2] = -1
grad_ln_T0['g'][2] = -1 / T0['g']
dS0_dr['g'][2] = -epsilon
grad_ln_rho0['g'][2] = ((-1/R)*dS0_dr['g'][2] + (1/(gamma-1)) * grad_ln_T0['g'][2])


#solve for hydrostatic equilibrium background state
#assume g_vec = -r
B2_T0['g'] = -1*B2_r + (Cp/epsilon)
B2_grad_T0['g'][2] = -1
B2_grad_ln_T0['g'][2] = -1 / B2_T0['g']
B2_dS0_dr['g'][2] = -epsilon
B2_grad_ln_rho0['g'][2] = ((-1/R)*B2_dS0_dr['g'][2] + (1/(gamma-1)) * B2_grad_ln_T0['g'][2])




# Problem
problem = d3.IVP([ln_rho1, T1, u, B2_ln_rho1, B2_T1, B2_u, tau_T1, tau_T2, tau_u1, tau_u2, B2_tau_T1, B2_tau_T2, B2_tau_u1, B2_tau_u2], namespace=locals())
problem.add_equation("dt(ln_rho1) + div_u + u@grad_ln_rho0 + (1/nu)*rvec@lift(tau_u2, -1) = -u@grad(ln_rho1)")
problem.add_equation("dt(T1) + T0*(gamma-1)*div_u + dot(u, grad_T0) - kappa*lap(T1) + lift(tau_T1, -1) + lift(tau_T2, -2) = - u@grad(T1) - T1*(gamma-1)*div_u + (1/Cv)*VH")
problem.add_equation("dt(u) - viscous_diffusion_L + R*(grad(T1) + T1*grad_ln_rho0 + T0*grad(ln_rho1)) + lift(tau_u1, -1) + lift(tau_u2, -2) = - u@grad(u) + -R*T1*grad(ln_rho1) + viscous_diffusion_R")
problem.add_equation("dt(B2_ln_rho1) + B2_div_u + B2_u@B2_grad_ln_rho0 + (1/nu)*B2_rvec@B2_lift(B2_tau_u2, -1) = -B2_u@grad(B2_ln_rho1)")
problem.add_equation("dt(B2_T1) + B2_T0*(gamma-1)*B2_div_u + dot(B2_u, B2_grad_T0) - kappa*lap(B2_T1) + B2_lift(B2_tau_T1, -1) + B2_lift(B2_tau_T2, -2) = - B2_u@grad(B2_T1) - B2_T1*(gamma-1)*B2_div_u + (1/Cv)*B2_VH")
problem.add_equation("dt(B2_u) - B2_viscous_diffusion_L + R*(grad(B2_T1) + B2_T1*B2_grad_ln_rho0 + B2_T0*grad(B2_ln_rho1)) + B2_lift(B2_tau_u1, -1) + B2_lift(B2_tau_u2, -2) = - B2_u@grad(B2_u) + -R*B2_T1*grad(B2_ln_rho1) + B2_viscous_diffusion_R")
problem.add_equation("T1(r=Ri) = 0")
problem.add_equation("u(r=Ri) = 0")
problem.add_equation("B2_T1(r=Ro2) = 0")
problem.add_equation("B2_u(r=Ro2) = 0")

problem.add_equation("T1(r=Ro) - B2_T1(r=Ro) = 0")
problem.add_equation("u(r=Ro) - B2_u(r=Ro) = 0")
problem.add_equation("angular(radial(sigma(r=Ro) - B2_sigma(r=Ro)))= 0")
problem.add_equation("ln_rho1(r=Ro) - B2_ln_rho1(r=Ro) =  0")
problem.add_equation("radial(grad(T1)(r=Ro) - grad(B2_T1)(r=Ro)) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
T1.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
T1['g'] *= (r - Ri) * (Ro - r) # Damp noise at walls
B2_T1.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
B2_T1['g'] *= (B2_r - Ro) * (Ro2 - B2_r) # Damp noise at walls

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=10*t_buoy, max_writes=10)
snapshots.add_task(T1(theta=np.pi/2), name='T1_eq')
snapshots.add_task(T1(phi=0), name='T1(phi=0)')
snapshots.add_task(B2_T1(theta=np.pi/2), name='B2_T1_eq')
snapshots.add_task(B2_T1(phi=0), name='B2_T1(phi=0)')


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)
CFL.add_velocity(B2_u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
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
