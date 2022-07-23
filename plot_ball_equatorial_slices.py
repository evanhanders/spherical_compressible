"""
This script plots snapshots of the evolution of a 2D slice through the equator of a BallBasis simulation.

Usage:
    ball_plot_equatorial_slices.py <root_dir> [options]
    ball_plot_equatorial_slices.py <root_dir> --radius=<r> [options]
    ball_plot_equatorial_slices.py <root_dir> --mesa_file=<f> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: snapshots]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots_equatorial]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

if args['--radius'] is not None:
    r_inner = 0
    r_outer = float(args['--radius'])
elif args['--mesa_file'] is not None:
    import h5py
    with h5py.File(args['--mesa_file'], 'r') as f:
        r_inner = 0
        r_outer = float(f['r_inner'][()])
else:
    r_inner = 0
    r_outer = 1

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, data_dir, fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_inch' : int(args['--col_inch']), 'row_inch' : int(args['--row_inch']) }

# Just plot a single plot (1x1 grid) of the field "T eq"
# remove_x_mean option removes the (numpy horizontal mean) over phi
# divide_x_mean divides the radial mean(abs(T eq)) over the phi direction
plotter.setup_grid(num_rows=2, num_cols=2, polar=True, **plotter_kwargs)
kwargs = {'azimuth_basis' : 'phi', 'radial_basis' : 'r', 'r_inner' : r_inner, 'r_outer' : r_outer}
plotter.add_polar_colormesh('s1_eq', remove_x_mean=False, divide_x_mean=False, **kwargs)
plotter.add_polar_colormesh('u_eq', vector_ind=0, cmap='PuOr_r', **kwargs)
plotter.add_polar_colormesh('u_eq', vector_ind=1, cmap='PuOr_r', **kwargs)
plotter.add_polar_colormesh('u_eq', vector_ind=2, cmap='PuOr_r', **kwargs)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
