import numpy as np
import dedalus.public as d3
from mpi4py import MPI
import logging
logger = logging.getLogger(__name__)

from ballShell_bvp import ballShell_HSE_BVP

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Parameters
Ri, Ro = 1.05, 2
Nphi, Ntheta, NrB, NrS = 1, 16, 32, 16
Reynolds = 3e2
Prandtl = 1
Peclet = Prandtl * Reynolds
epsilon = 1e-4
chi = (epsilon**(1/2))/Peclet
nu  = (epsilon**(1/2))/Reynolds
gamma = 5/3
R = 1
Cv = R * 1 / (gamma-1)
Cp = R + Cv
dealias = 3/2
dtype = np.float64
mesh = None
#mesh = [16,16]


#Atmosphere setup
N2_func = lambda r: 0.5*zero_to_one(r, 0.8*Ri, width=0.1*Ri)
g_func = lambda r: -r
Lconv_func = lambda r: epsilon * r**3 * one_to_zero(r, 0.7*Ri, width=0.1*Ri)
#Lconv_func = lambda r: epsilon * chi * r**3 * one_to_zero(r, 0.7*Ri, width=0.1*Ri)
atmosphere = ballShell_HSE_BVP(N2_func, g_func, Lconv_func, Nrs=(NrB, NrS), radii=(Ri,Ro), gamma=gamma, R=R)

#timestepping
timestepper = d3.SBDF2
safety = 0.15
t_buoy = np.sqrt(1/epsilon)
#max_timestep = t_buoy/10
max_timestep = np.min((t_buoy/10, 0.5/np.sqrt(N2_func(Ro))))
stop_sim_time = 100*t_buoy

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=np.float64, mesh=None, comm=MPI.COMM_SELF)
B1_basis = d3.BallBasis(coords, shape=(Nphi, Ntheta, NrB), radius=Ri, dealias=dealias, dtype=dtype)
B1_s2_basis = B1_basis.S2_basis()
B1_phi, B1_theta, B1_r = dist.local_grids(B1_basis)
B1_phi_de, B1_theta_de, B1_r_de = dist.local_grids(B1_basis, scales=B1_basis.dealias)
B2_basis = d3.ShellBasis(coords, shape=(Nphi, Ntheta, NrS), radii=(Ri, Ro), dealias=dealias, dtype=dtype)
B2_s2_basis = B2_basis.S2_basis()
B2_phi, B2_theta, B2_r = dist.local_grids(B2_basis)
B2_phi_de, B2_theta_de, B2_r_de = dist.local_grids(B2_basis, scales=B2_basis.dealias)

# Fields
B1_ln_rho1 = dist.Field(name='ln_rho1', bases=B1_basis)
B1_s1 = dist.Field(name='s1', bases=B1_basis)
B1_u = dist.VectorField(coords, name='u', bases=B1_basis)
B2_ln_rho1 = dist.Field(name='ln_rho1', bases=B2_basis)
B2_s1 = dist.Field(name='s1', bases=B2_basis)
B2_u = dist.VectorField(coords, name='u', bases=B2_basis)
B1_tau_s = dist.Field(name='tau_s', bases=B1_s2_basis)
B1_tau_u = dist.VectorField(coords, name='tau_u', bases=B1_s2_basis)
B2_tau_s1 = dist.Field(name='tau_s1', bases=B2_s2_basis)
B2_tau_s2 = dist.Field(name='tau_s2', bases=B2_s2_basis)
B2_tau_u1 = dist.VectorField(coords, name='tau_u1', bases=B2_s2_basis)
B2_tau_u2 = dist.VectorField(coords, name='tau_u2', bases=B2_s2_basis)


B1_grad_pom0 = dist.VectorField(coords, name='grad_pom0', bases=B1_basis.radial_basis)
B1_grad_ln_pom0 = dist.VectorField(coords, name='grad_ln_pom0', bases=B1_basis.radial_basis)
B1_grad_ln_rho0 = dist.VectorField(coords, name='grad_ln_rho0', bases=B1_basis.radial_basis)
B1_grad_s0 = dist.VectorField(coords, name='grad_s0', bases=B1_basis.radial_basis)
B1_pom0 = dist.Field(name='pom0', bases=B1_basis.radial_basis)
B1_rho0 = dist.Field(name='rho0', bases=B1_basis.radial_basis)
B1_ln_rho0 = dist.Field(name='ln_rho0', bases=B1_basis.radial_basis)
B1_g = dist.VectorField(coords, name='g', bases=B1_basis.radial_basis)
B1_Q = dist.Field(name='Q', bases=B1_basis)
B1_ones = dist.Field(name='ones', bases=B1_basis)
B1_ones['g'] = 1
B2_grad_pom0 = dist.VectorField(coords, name='grad_pom0', bases=B2_basis.radial_basis)
B2_grad_ln_pom0 = dist.VectorField(coords, name='grad_ln_pom0', bases=B2_basis.radial_basis)
B2_grad_ln_rho0 = dist.VectorField(coords, name='grad_ln_rho0', bases=B2_basis.radial_basis)
B2_grad_s0 = dist.VectorField(coords, name='grad_s0', bases=B2_basis.radial_basis)
B2_pom0 = dist.Field(name='pom0', bases=B2_basis.radial_basis)
B2_rho0 = dist.Field(name='rho0', bases=B2_basis.radial_basis)
B2_ln_rho0 = dist.Field(name='ln_rho0', bases=B2_basis.radial_basis)
B2_g = dist.VectorField(coords, name='g', bases=B2_basis.radial_basis)
B2_Q = dist.Field(name='Q', bases=B2_basis)
B2_ones = dist.Field(name='ones', bases=B2_basis)
B2_ones['g'] = 1

eye = dist.TensorField(coords, name='I')
eye['g'] = 0
for i in range(3):
    eye['g'][i,i] = 1
er = dist.VectorField(coords, name='er')
er['g'][2] = 1

B1_rvec = dist.VectorField(coords, bases=B1_basis.radial_basis)
B1_rvec['g'][2] = B1_r
B2_rvec = dist.VectorField(coords, bases=B2_basis.radial_basis)
B2_rvec['g'][2] = B2_r

#Substitutions / operators
B1_lift_basis = B1_basis.derivative_basis(0)
B1_lift = lambda A: d3.Lift(A, B1_lift_basis, -1)
B2_lift_basis = B2_basis.derivative_basis(2)
B2_lift = lambda A, n: d3.Lift(A, B2_lift_basis, n)

B1_grad_u = d3.grad(B1_u)
B1_grad_s1 = d3.grad(B1_s1)
B1_grad_ln_rho1 = d3.grad(B1_ln_rho1)
B1_div_u = d3.div(B1_u)
B2_grad_u = d3.grad(B2_u)
B2_grad_s1 = d3.grad(B2_s1)
B2_grad_ln_rho1 = d3.grad(B2_ln_rho1)
B2_div_u = d3.div(B2_u)

#Setup NCCs / background
B1_g['g'] = atmosphere['g'](B1_r)
B1_grad_s0['g'] = atmosphere['grad_s0'](B1_r)
B1_grad_pom0['g'] = atmosphere['grad_pom0'](B1_r)
B1_pom0['g'] = atmosphere['pom0'](B1_r)
B1_grad_ln_pom0['g'] = atmosphere['grad_ln_pom0'](B1_r)
B1_grad_ln_rho0['g'] = atmosphere['grad_ln_rho0'](B1_r)
B1_rho0['g'] = atmosphere['rho0'](B1_r)
B1_ln_rho0['g'] = atmosphere['ln_rho0'](B1_r)
B1_Q['g'] = atmosphere['Q'](B1_r)
B1_inv_pom0 = (1/B1_pom0).evaluate()
B2_g['g'] = atmosphere['g'](B2_r)
B2_grad_s0['g'] = atmosphere['grad_s0'](B2_r)
B2_grad_pom0['g'] = atmosphere['grad_pom0'](B2_r)
B2_pom0['g'] = atmosphere['pom0'](B2_r)
B2_grad_ln_pom0['g'] = atmosphere['grad_ln_pom0'](B2_r)
B2_grad_ln_rho0['g'] = atmosphere['grad_ln_rho0'](B2_r)
B2_rho0['g'] = atmosphere['rho0'](B2_r)
B2_ln_rho0['g'] = atmosphere['ln_rho0'](B2_r)
B2_Q['g'] = atmosphere['Q'](B2_r)
B2_inv_pom0 = (1/B2_pom0).evaluate()

#Fluctuating thermodynamics
B1_pom1_over_pom0 = gamma*(B1_s1/Cp + ((gamma-1)/gamma)*B1_ln_rho1)
B1_grad_pom1_over_pom0 = gamma*(B1_grad_s1/Cp + ((gamma-1)/gamma)*B1_grad_ln_rho1)
B1_pom1 = B1_pom0*B1_pom1_over_pom0
B1_grad_pom1 = B1_grad_pom0*B1_pom1_over_pom0 + B1_pom0*B1_grad_pom1_over_pom0
B1_pom_fluc_over_pom0 = np.exp(B1_pom1_over_pom0) - (1 + B1_pom1_over_pom0)
B1_pom_fluc = B1_pom0*B1_pom_fluc_over_pom0
B1_pom_full = B1_ones*B1_pom0 + B1_pom1 + B1_pom_fluc
B1_rho_full = B1_rho0*np.exp(B1_ln_rho1)
B2_pom1_over_pom0 = gamma*(B2_s1/Cp + ((gamma-1)/gamma)*B2_ln_rho1)
B2_grad_pom1_over_pom0 = gamma*(B2_grad_s1/Cp + ((gamma-1)/gamma)*B2_grad_ln_rho1)
B2_pom1 = B2_pom0*B2_pom1_over_pom0
B2_grad_pom1 = B2_grad_pom0*B2_pom1_over_pom0 + B2_pom0*B2_grad_pom1_over_pom0
B2_pom_fluc_over_pom0 = np.exp(B2_pom1_over_pom0) - (1 + B2_pom1_over_pom0)
B2_pom_fluc = B2_pom0*B2_pom_fluc_over_pom0
B2_pom_full = B2_ones*B2_pom0 + B2_pom1 + B2_pom_fluc
B2_rho_full = B2_rho0*np.exp(B2_ln_rho1)

#Momentum terms
B1_background_HSE = gamma*B1_pom0*(B1_grad_ln_rho0 + B1_grad_s0/Cp) - B1_g
B1_linear_HSE = gamma*B1_pom0*(B1_grad_ln_rho1 + B1_grad_s1/Cp) + B1_g * B1_pom1_over_pom0
B1_nonlinear_HSE = gamma*(B1_pom1 + B1_pom_fluc)*(B1_grad_ln_rho1 + B1_grad_s1/Cp) + B1_g*B1_pom_fluc_over_pom0
B2_background_HSE = gamma*B2_pom0*(B2_grad_ln_rho0 + B2_grad_s0/Cp) - B2_g
B2_linear_HSE = gamma*B2_pom0*(B2_grad_ln_rho1 + B2_grad_s1/Cp) + B2_g * B2_pom1_over_pom0
B2_nonlinear_HSE = gamma*(B2_pom1 + B2_pom_fluc)*(B2_grad_ln_rho1 + B2_grad_s1/Cp) + B2_g*B2_pom_fluc_over_pom0

B1_E = 0.5*(B1_grad_u + d3.trans(B1_grad_u))
B1_sigma = 2*(B1_E - (1/3)*B1_div_u*eye)
B1_viscous_diffusion_L = nu*(d3.div(B1_sigma) + d3.dot(B1_sigma, B1_grad_ln_rho0))
B1_viscous_diffusion_R = nu*d3.dot(B1_sigma, d3.grad(B1_ln_rho1))
B1_VH = 2 * nu * (d3.trace(d3.dot(B1_E,B1_E)) - (1/3)*B1_div_u*B1_div_u)
B2_E = 0.5*(B2_grad_u + d3.trans(B2_grad_u))
B2_sigma = 2*(B2_E - (1/3)*B2_div_u*eye)
B2_viscous_diffusion_L = nu*(d3.div(B2_sigma) + d3.dot(B2_sigma, B2_grad_ln_rho0))
B2_viscous_diffusion_R = nu*d3.dot(B2_sigma, d3.grad(B2_ln_rho1))
B2_VH = 2 * nu * (d3.trace(d3.dot(B2_E,B2_E)) - (1/3)*B2_div_u*B2_div_u)

#Thermal terms
##Entropy diffusion
#thermal_diffusion_L = chi*(d3.lap(s1) + grad_s1@(grad_ln_pom0 + grad_ln_rho0))
#thermal_diffusion_R = chi*(R/(rho_full*pom_full))*d3.div(rho_full*pom_full*grad_s1) - thermal_diffusion_L

#Temperature diffusion - assumes constant and uniform chi
B1_thermal_diffusion_L = (gamma/(gamma-1))*chi*(d3.div(B1_grad_pom1*B1_inv_pom0) + (B1_grad_pom1*B1_inv_pom0)@(B1_grad_ln_rho0 + B1_grad_ln_pom0))
B1_thermal_diffusion_R = (gamma/(gamma-1))*chi*(R/(B1_rho_full*B1_pom_full))*d3.div(B1_rho_full*(B1_grad_pom1 + d3.grad(B1_pom_fluc))) - B1_thermal_diffusion_L
B2_thermal_diffusion_L = (gamma/(gamma-1))*chi*(d3.div(B2_grad_pom1*B2_inv_pom0) + (B2_grad_pom1*B2_inv_pom0)@(B2_grad_ln_rho0 + B2_grad_ln_pom0))
B2_thermal_diffusion_R = (gamma/(gamma-1))*chi*(R/(B2_rho_full*B2_pom_full))*d3.div(B2_rho_full*(B2_grad_pom1 + d3.grad(B2_pom_fluc))) - B2_thermal_diffusion_L

##Simple entropy diffusion
#B1_thermal_diffusion_L = chi*d3.lap(B1_s1)
#B1_thermal_diffusion_R = 0
#B2_thermal_diffusion_L = chi*d3.lap(B2_s1)
#B2_thermal_diffusion_R = 0


# Problem
problem = d3.IVP([B1_ln_rho1, B1_s1, B1_u, B2_ln_rho1, B2_s1, B2_u, B1_tau_s, B1_tau_u, B2_tau_s1, B2_tau_s2, B2_tau_u1, B2_tau_u2 ], namespace=locals())
problem.add_equation("dt(B1_ln_rho1) + B1_div_u + B1_u@B1_grad_ln_rho0 = -B1_u@grad(B1_ln_rho1)")
problem.add_equation("dt(B1_s1) + dot(B1_u, B1_grad_s0) - B1_thermal_diffusion_L + B1_lift(B1_tau_s) = - B1_u@grad(B1_s1) + B1_thermal_diffusion_R + (R/(B1_rho_full*B1_pom_full))*(B1_Q + B1_VH)")
problem.add_equation("dt(B1_u) - B1_viscous_diffusion_L + B1_linear_HSE + B1_lift(B1_tau_u) = - B1_u@grad(B1_u) - B1_nonlinear_HSE + B1_viscous_diffusion_R")
problem.add_equation("dt(B2_ln_rho1) + B2_div_u + B2_u@B2_grad_ln_rho0 + (1/nu)*B2_rvec@B2_lift(B2_tau_u2, -1) = -B2_u@grad(B2_ln_rho1)")
problem.add_equation("dt(B2_s1) + dot(B2_u, B2_grad_s0) - B2_thermal_diffusion_L + B2_lift(B2_tau_s1, -1) + B2_lift(B2_tau_s2, -2) = - B2_u@grad(B2_s1) + B2_thermal_diffusion_R + (R/(B2_rho_full*B2_pom_full))*(B2_Q + B2_VH)")
problem.add_equation("dt(B2_u) - B2_viscous_diffusion_L + B2_linear_HSE + B2_lift(B2_tau_u1, -1) + B2_lift(B2_tau_u2, -2) = - B2_u@grad(B2_u) - B2_nonlinear_HSE + B2_viscous_diffusion_R")

problem.add_equation("B2_s1(r=Ro) = 0")
problem.add_equation("radial(B2_u(r=Ro)) = 0")
problem.add_equation("angular(radial(B2_E(r=Ro))) = 0")

problem.add_equation("B1_s1(r=Ri) - B2_s1(r=Ri) = 0")
problem.add_equation("B1_u(r=Ri) - B2_u(r=Ri) = 0")
problem.add_equation("angular(radial(B1_sigma(r=Ri) - B2_sigma(r=Ri))) = 0")
problem.add_equation("B1_ln_rho1(r=Ri) - B2_ln_rho1(r=Ri) = 0")
problem.add_equation("radial(B1_grad_pom1(r=Ri) - B2_grad_pom1(r=Ri)) = -radial(grad(B1_pom_fluc)(r=Ri) - grad(B2_pom_fluc)(r=Ri))")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
B1_s1.fill_random('g', seed=42, distribution='normal', scale=1e-3*epsilon) # Random noise
B1_s1['g'] *= B1_r**2 * one_to_zero(B1_r, 0.8*Ri, width=0.2*Ri)

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=t_buoy/5, max_writes=10)
snapshots.add_task(B1_s1(theta=np.pi/2), name='B1_s1_eq')
snapshots.add_task(B1_s1(phi=0), name='B1_s1(phi=0)')
snapshots.add_task(B1_s1(phi=np.pi), name='B1_s1(phi=pi)')
snapshots.add_task(B1_u(phi=0), name='B1_u(phi=0)')
snapshots.add_task(B1_u(phi=np.pi), name='B1_u(phi=pi)')
snapshots.add_task(B1_u(theta=np.pi/2), name='B1_u_eq')
snapshots.add_task(B2_s1(theta=np.pi/2), name='B2_s1_eq')
snapshots.add_task(B2_s1(phi=0), name='B2_s1(phi=0)')
snapshots.add_task(B2_s1(phi=np.pi), name='B2_s1(phi=pi)')
snapshots.add_task(B2_u(phi=0), name='B2_u(phi=0)')
snapshots.add_task(B2_u(phi=np.pi), name='B2_u(phi=pi)')
snapshots.add_task(B2_u(theta=np.pi/2), name='B2_u_eq')

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=1, safety=safety, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(B1_u)
CFL.add_velocity(B2_u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=1)
flow.add_property(np.sqrt(B1_u@B1_u)/nu, name='Re')
flow.add_property(np.sqrt(B1_u@B1_u)/np.sqrt(B1_pom_full), name='Ma')

#EOS = (grad_s0*ones + grad_s1)/Cp - ((1/gamma)*d3.grad(np.log(pom0*ones + pom1 + pom_fluc)) - ((gamma-1)/(gamma))*(grad_ln_rho0*ones + grad_ln_rho1))

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 1 == 0:
            max_Re = flow.max('Re')
            max_Ma = flow.max('Ma')
            logger.info('Iteration=%i, Time=%e / %e, dt=%e / %e, max(Re)=%e, max(Ma)=%e' %(solver.iteration, solver.sim_time, solver.sim_time/t_buoy, timestep, timestep/t_buoy, max_Re, max_Ma))
        if np.isnan(max_Re):
            raise ValueError("Re is NaN")
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
