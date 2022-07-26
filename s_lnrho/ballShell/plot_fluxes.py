"""
This script plots the average temperature structure of the simulation, averaged over a specified number of writes.

Usage:
    plot_fluxes.py [options]

Options:
    --root_dir=<str>         Path to root run directory [default: ./]
    --data_dir=<str>         Name of data handler directory [default: profiles]
    --fig_name=<str>         Name of figure output directory & figures [default: flux_plots]
    --start_file=<int>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<int>          Total number of files to plot
    --roll_writes=<int>      Number of writes over which to take average
    --dpi=<int>              Image pixel density [default: 200]

    --col_inch=<float>        Figure width (inches) [default: 6]
    --row_inch=<float>       Figure height (inches) [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
import h5py
import numpy as np

from plotpal.profiles import RolledProfilePlotter
from plotpal.file_reader import match_basis

# Read in master output directory
root_dir    = args['--root_dir']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_file  = int(args['--start_file'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

roll_writes = args['--roll_writes']
if roll_writes is not None:
    roll_writes = int(roll_writes)

tasks = ['F_cond', 'F_KE', 'F_PE', 'F_enth', 'F_visc']
handler_tasks = []
for prefix in ['B1_', 'B2_']:
    handler_tasks += ['{}{}'.format(prefix, t) for t in tasks]

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)
epsilon = 1e-6
Ri = 1.05
Lconv_func = lambda r: epsilon * r**3 * one_to_zero(r, 0.7*Ri, width=0.1*Ri)

from palettable.colorbrewer.qualitative import Dark2_7
Dark2_7 = Dark2_7.mpl_colors
def luminosities(ax, dictionary, index):
    rs = np.concatenate((match_basis(dictionary['B1_F_KE'], 'r'), match_basis(dictionary['B2_F_KE'], 'r')))
    KEs = np.concatenate((dictionary['B1_F_KE'][index].ravel(), dictionary['B2_F_KE'][index].ravel()))
    PEs = np.concatenate((dictionary['B1_F_PE'][index].ravel(), dictionary['B2_F_PE'][index].ravel()))
    enths = np.concatenate((dictionary['B1_F_enth'][index].ravel(), dictionary['B2_F_enth'][index].ravel()))
    viscs = np.concatenate((dictionary['B1_F_visc'][index].ravel(), dictionary['B2_F_visc'][index].ravel()))
    conds = np.concatenate((dictionary['B1_F_cond'][index].ravel(), dictionary['B2_F_cond'][index].ravel()))

    sim_lum = Lconv_func(rs)
    sum = KEs +enths + viscs + conds + PEs
    ax.plot(rs, sim_lum, label='goal', c='k')

    ax.plot(rs, 4*np.pi*rs**2*KEs, label='KE', c=Dark2_7[0])
    ax.plot(rs, 4*np.pi*rs**2*enths, label='enth', c=Dark2_7[1])
    ax.plot(rs, 4*np.pi*rs**2*viscs, label='visc', c=Dark2_7[3])
    ax.plot(rs, 4*np.pi*rs**2*conds, label='cond', c=Dark2_7[4])
    ax.plot(rs, 4*np.pi*rs**2*PEs, label='PE', c=Dark2_7[6])
    ax.plot(rs, 4*np.pi*rs**2*sum, label='sum', c=Dark2_7[5], ls='--')
    ax.legend()
    

# Create Plotter object, tell it which fields to plot
plotter = RolledProfilePlotter(root_dir, file_dir=data_dir, out_name=fig_name, roll_writes=roll_writes, start_file=start_file, n_files=n_files)
plotter.setup_grid(num_rows=1, num_cols=1, col_inch=float(args['--col_inch']), row_inch=float(args['--row_inch']))
plotter.add_line('r', luminosities, grid_num=0, needed_tasks=handler_tasks)
plotter.plot_lines()
