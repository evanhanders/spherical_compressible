"""
Dedalus script simulating Boussinesq convection in a spherical shell. This script
demonstrates solving an initial value problem in the shell. It can be ran serially
or in parallel, and uses the built-in analysis framework to save data snapshots
to HDF5 files. The `plot_shell.py` script can be used to produce plots from the
saved data. The simulation should take about 10 cpu-minutes to run.

The problem is non-dimensionalized using the shell thickness and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    chi = (Rayleigh * Prandtl)**(-1/2)
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

from ball_bvp import ball_HSE_BVP

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)




# Parameters
Ro = 1
Nphi, Ntheta, Nr = 1, 16, 32
#Nphi, Ntheta, Nr = 1, 32, 32
Rayleigh = 5e2
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
s2_basis = basis.S2_basis()

# Fields
ln_rho1 = dist.Field(name='ln_rho1', bases=basis)
s1 = dist.Field(name='s1', bases=basis)
u = dist.VectorField(coords, name='u', bases=basis)
tau_s1 = dist.Field(name='tau_s1', bases=s2_basis)
tau_T2 = dist.Field(name='tau_T2', bases=s2_basis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=s2_basis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=s2_basis)
PE_int = dist.Field(name='PE_int')
KE_int = dist.Field(name='KE_int')
IE_int = dist.Field(name='IE_int')

KE_last = dist.Field(name='KE_last')
KE_now = dist.Field(name='KE_now')

grad_pom0 = dist.VectorField(coords, name='grad_pom0', bases=basis.radial_basis)
grad_s0 = dist.VectorField(coords, name='grad_s0', bases=basis.radial_basis)
grad_ln_pom0 = dist.VectorField(coords, name='grad_ln_pom0', bases=basis.radial_basis)
grad_ln_rho0 = dist.VectorField(coords, name='grad_ln_rho0', bases=basis.radial_basis)
pom0 = dist.Field(name='pom0', bases=basis.radial_basis)
ln_rho0 = dist.Field(name='ln_rho0', bases=basis.radial_basis)
s0 = dist.Field(name='s0', bases=basis.radial_basis)
rho0 = dist.Field(name='rho0', bases=basis.radial_basis)
g = dist.VectorField(coords, name='g', bases=basis.radial_basis)
g_phi = dist.Field(name='g_phi', bases=basis.radial_basis)
Q = dist.Field(name='Q', bases=basis)
ones = dist.Field(name='ones', bases=basis)
ones['g'] = 1
eye = dist.TensorField(coords, name='I')
eye['g'] = 0
for i in range(3):
    eye['g'][i,i] = 1

# Substitutions
epsilon = 1e-2
t_buoy = np.sqrt(1/epsilon)
max_timestep = t_buoy/10 #TODO: add N^2 timestep constraint
stop_sim_time = 2000*t_buoy
chi = epsilon*(Rayleigh * Prandtl)**(-1/2)
nu = epsilon*(Rayleigh / Prandtl)**(-1/2)
gamma = 5/3
R = 1
Cv = R * 1 / (gamma-1)
Cp = R + Cv
phi, theta, r = dist.local_grids(basis)
phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)
er = dist.VectorField(coords, bases=basis.radial_basis)
er['g'][2] = 1
rvec = dist.VectorField(coords, bases=basis.radial_basis)
rvec['g'][2] = r

lift_basis = basis.derivative_basis(0)
lift = lambda A: d3.Lift(A, lift_basis, -1)
integ = lambda A: d3.Integrate(A, coords)


grad_u = d3.grad(u)
grad_s1 = d3.grad(s1)
grad_ln_rho1 = d3.grad(ln_rho1)
div_u = d3.div(u)


#Analytical background state
#assume g_vec = -r & T = rho = 1 at outer boundary.
N2_func = lambda r: 0*r#0.5*zero_to_one(r, 0.8*Ro, width=0.1*Ro)
g_func = lambda r: -r
Lconv_func = lambda r: epsilon * r**3 * one_to_zero(r, 0.5*Ro, width=0.1*Ro)
atmosphere = ball_HSE_BVP(N2_func, g_func, Lconv_func,  Nr=Nr, Ro=Ro, gamma=5/3, R=1)

g['g'] = atmosphere['g']
g_phi['g'] = atmosphere['g_phi']
grad_s0['g'] = atmosphere['grad_s0']
grad_pom0['g'] = atmosphere['grad_pom0']
pom0['g'] = atmosphere['pom0']
grad_ln_pom0['g'] = atmosphere['grad_ln_pom0']
grad_ln_rho0['g'] = atmosphere['grad_ln_rho0']
rho0['g'] = atmosphere['rho0']
ln_rho0['g'] = atmosphere['ln_rho0']
s0['g'] = atmosphere['s0']
inv_pom0 = (1/pom0).evaluate()
Q['g'] = atmosphere['Q']
P0 = rho0*pom0


pom1_over_pom0 = gamma*(s1/Cp + ((gamma-1)/gamma)*ln_rho1)
grad_pom1_over_pom0 = gamma*(grad_s1/Cp + ((gamma-1)/gamma)*grad_ln_rho1)
pom1 = pom0*pom1_over_pom0
grad_pom1 = grad_pom0*pom1_over_pom0 + pom0*grad_pom1_over_pom0
pom_fluc_over_pom0 = np.exp(pom1_over_pom0) - (1 + pom1_over_pom0)
pom_fluc = pom0*pom_fluc_over_pom0

pom_full = ones*pom0 + pom1 + pom_fluc
rho_full = rho0*np.exp(ln_rho1)
P_full = pom_full*rho_full
momentum = rho_full * u

#Momentum terms
background_HSE = gamma*pom0*(grad_ln_rho0 + grad_s0/Cp) - g
linear_HSE = gamma*pom0*(grad_ln_rho1 + grad_s1/Cp) + g * pom1_over_pom0
nonlinear_HSE = d3.grad(P_full)/rho_full - ones*gamma*pom0*(grad_ln_rho0 + grad_s0/Cp) - linear_HSE
#gamma*(pom1 + pom_fluc)*(grad_ln_rho1 + grad_s1/Cp) + g*pom_fluc_over_pom0

E = 0.5*(grad_u + d3.trans(grad_u))
sigma = 2*(E - (1/3)*div_u*eye)
viscous_diffusion_L = 2*nu*d3.lap(u)
viscous_diffusion_R = (1/rho_full)*d3.div(rho_full*nu*sigma) - viscous_diffusion_L
#viscous_diffusion_L = nu*(d3.div(sigma) + d3.dot(sigma, grad_ln_rho0))
#viscous_diffusion_R = nu*d3.dot(sigma, d3.grad(ln_rho1))
VH = 2 * nu * (d3.trace(d3.dot(E,E)) - (1/3)*div_u*div_u)

u_squared = u@u
KE = 0.5 * rho_full * u_squared
PE = rho_full * g_phi
IE = P_full*(Cv/R)
PE0 = (rho0 * g_phi).evaluate()
IE0 = (P0*(Cv/R)).evaluate()
PE1 = PE - (PE0*ones).evaluate()
IE1 = IE - (IE0*ones).evaluate()
TE = KE + PE + IE
FE = KE + PE1 + IE1


#Thermal terms
thermal_diffusion_L = (gamma/(gamma-1))*chi*(d3.div(grad_pom1*inv_pom0) + (grad_pom1*inv_pom0)@(grad_ln_rho0 + grad_ln_pom0))
thermal_diffusion_R = (gamma/(gamma-1))*chi*(R/(rho_full*pom_full))*d3.div(rho_full*(grad_pom1 + d3.grad(pom_fluc))) - thermal_diffusion_L

EOS = (s1 + s0*ones)/Cp - ( (1/(gamma)) * (np.log(pom0*ones + pom1 + pom_fluc) - np.log(R)) - ((gamma-1)/(gamma)) * (ln_rho1 + ln_rho0*ones)) 
EOS_goodness = np.sqrt(EOS*EOS)
HSE = linear_HSE + nonlinear_HSE


#Production of energy
viscous_production = momentum @ (viscous_diffusion_L + viscous_diffusion_R) + rho_full*VH
thermal_diff_production = (P_full/R) * (thermal_diffusion_R + thermal_diffusion_L)
Q_production = Q
PdV_production = momentum @ (-nonlinear_HSE - linear_HSE - ones*gamma*pom0*(grad_ln_rho0 + grad_s0/Cp)) - P_full*d3.div(u)
tau_production = -momentum @ lift(tau_u1) - (P_full/R) * lift(tau_s1)

PE_production = -(d3.div(g_phi*momentum) + momentum@g)
KE_production = momentum@(viscous_diffusion_L + viscous_diffusion_R - nonlinear_HSE - linear_HSE - lift(tau_u1)) - d3.div(u*KE)
IE_production = thermal_diff_production + Q_production - P_full*d3.div(u) - d3.div(u*IE) + rho_full*VH  + momentum@lift(tau_u1)
#IE_production = thermal_diff_production + Q_production - P_full*d3.div(u) - (P_full/R)*lift(tau_s1) - d3.div(u*IE) + rho_full*VH 
E_production = PE_production + KE_production + IE_production


# Problem
problem = d3.IVP([ln_rho1, s1, u, tau_s1, tau_u1, PE_int, KE_int, IE_int], namespace=locals())
problem.add_equation("dt(ln_rho1) + div_u + u@grad_ln_rho0 = -u@grad(ln_rho1)")
problem.add_equation("dt(s1) + dot(u, grad_s0) - thermal_diffusion_L + lift(tau_s1) = - u@grad(s1) + thermal_diffusion_R + (R/P_full)*(Q + rho_full*VH) + lift(tau_s1) + momentum@lift(tau_u1)*(R/P_full)")
problem.add_equation("dt(u) - viscous_diffusion_L + linear_HSE + lift(tau_u1) = - u@grad(u) - nonlinear_HSE + viscous_diffusion_R")

#problem.add_equation("s1(r=Ro) = 0")
problem.add_equation("radial(grad_pom1(r=Ro)) = -radial(grad(pom_fluc)(r=Ro))")
problem.add_equation("radial(u(r=Ro)) = 0")
problem.add_equation("angular(radial(E(r=Ro))) = 0")

problem.add_equation("dt(PE_int) = integ(PE_production)")
problem.add_equation("dt(KE_int) = integ(KE_production)")
problem.add_equation("dt(IE_int) = integ(IE_production)")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
s1.fill_random('g', seed=42, distribution='normal', scale=1e-6*epsilon) # Random noise
s1['g'] *= r**2 - Ro**2

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=2*max_timestep, max_writes=10)
snapshots.add_task(s1(theta=np.pi/2), name='s1_eq')
snapshots.add_task(s1(phi=0), name='s1(phi=0)')
snapshots.add_task(s1(phi=np.pi), name='s1(phi=pi)')
snapshots.add_task(u(phi=0), name='u(phi=0)')
snapshots.add_task(u(phi=np.pi), name='u(phi=pi)')
snapshots.add_task(u(theta=np.pi/2), name='u_eq')
snapshots.add_task((linear_HSE)(phi=0), name='HSE(phi=0)')
snapshots.add_task((linear_HSE)(phi=np.pi), name='HSE(phi=pi)')


# Analysis
TE_int = KE_int + PE_int + IE_int
FE_int = TE_int - integ((IE0+PE0)*ones)
scalars = solver.evaluator.add_file_handler('scalars', sim_dt=max_timestep, max_writes=100)
scalars.add_task(integ(KE), name='KE')
scalars.add_task(integ(IE), name='IE')
scalars.add_task(integ(PE), name='PE')
scalars.add_task(integ(TE), name='TE')
scalars.add_task(integ(IE1), name='IE1')
scalars.add_task(integ(PE1), name='PE1')
scalars.add_task(integ(FE), name='FE')
scalars.add_task(integ(KE) + integ(IE1) + integ(PE1), name='FE_pieces')
scalars.add_task(integ(EOS_goodness), name='EOS')
scalars.add_task(integ(viscous_production), name='prod_visc')
scalars.add_task(integ(thermal_diff_production), name='prod_thermdiff')
scalars.add_task(integ(Q_production), name='prod_Q')
scalars.add_task(integ(PdV_production), name='prod_PdV')
scalars.add_task(integ(tau_production), name='prod_tau')
scalars.add_task(PE_int, name='PE_int')
scalars.add_task(PE_int - integ(PE0*ones), name='PE1_int')
scalars.add_task(KE_int, name='KE_int')
scalars.add_task(IE_int, name='IE_int')
scalars.add_task(IE_int - integ(IE0*ones), name='IE1_int')
scalars.add_task(TE_int, name='TE_int')
scalars.add_task(FE_int, name='FE_int')
scalars.add_task(integ(E_production), name='E_production')
# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(u@u)/nu, name='Re')
flow.add_property(integ(KE), name='KE')
flow.add_property(integ(PE), name='PE')
flow.add_property(integ(IE), name='IE')
flow.add_property(integ(PE1), name='PE1')
flow.add_property(integ(IE1), name='IE1')
flow.add_property(integ(FE), name='FE')
flow.add_property(integ(viscous_production), name='prod_visc')
flow.add_property(integ(thermal_diff_production), name='prod_thermdiff')
flow.add_property(integ(Q_production), name='prod_Q')
flow.add_property(integ(PdV_production), name='prod_PdV')
flow.add_property(integ(tau_production), name='prod_tau')
flow.add_property(integ(EOS_goodness), name='EOS')
flow.add_property(FE_int, name='FE_int')

#EOS = (grad_s0*ones + grad_s1)/Cp - ((1/gamma)*d3.grad(np.log(pom0*ones + pom1 + pom_fluc)) - ((gamma-1)/(gamma))*(grad_ln_rho0*ones + grad_ln_rho1))

PE_int['g'] = integ(PE).evaluate()['g']
PE_int0 = np.copy(PE_int['g']).min()


IE_int['g'] = integ(IE).evaluate()['g']
IE_int0 = np.copy(IE_int['g']).min()

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)

        #Possible that we can add a source term to fix this....
        KE_last['g'] = KE_now['g']
        KE_now['g'] = integ(KE).evaluate()['g']
        delta_KE = KE_now['g'] - KE_last['g']
        expected_delta_KE = integ(KE_production).evaluate()['g']*timestep


        # I think this points towards a problem with energy conservation from viscous diffusion tensor stuff...
        if (solver.iteration-1) % 1 == 0:
            max_Re = flow.max('Re')
            FE = flow.max("FE") #flow.max("PE1") + flow.max("IE1") + flow.max("KE")
            logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%e, EOS=%e, FE_int=%e / FE=%e' %(solver.iteration, solver.sim_time, timestep, max_Re, flow.max("EOS"), flow.max("FE_int"), FE))
#            logger.info("FE: {:.3e}, KE: {:.3e}, PE1: {:.3e}, IE1: {:.3e}".format(FE, flow.max("KE"), flow.max("PE1"), flow.max("IE1")))
#            total_production = integ(E_production).evaluate()['g'].min()
#            logger.info("Production ({:.3e}) - visc: {:.3e}, thermdiff: {:.3e}, Q: {:.3e}, PdV: {:.3e}, tau: {:.3e}".format(total_production, flow.max("prod_visc"), flow.max("prod_thermdiff"), flow.max("prod_Q"), flow.max("prod_PdV"), flow.max("prod_tau")))
#            logger.info("PE {:.3e} (diff {:.3e}), KE {:.3e} (diff {:.3e}), IE {:.3e} (diff {:.3e})".\
#                        format(PE_int['g'].min() - PE_int0, PE_int['g'].min() - flow.max("PE"), \
#                               KE_int['g'].min(), KE_int['g'].min() - flow.max("KE"), \
#                               IE_int['g'].min() - IE_int0, IE_int['g'].min() - flow.max("IE")))
        if np.isnan(max_Re):
            raise ValueError("Re is NaN")
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
