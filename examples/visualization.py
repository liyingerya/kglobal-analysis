"""Small reusable visualization examples, using synthetic data by default.

Run: python examples/visualization.py --output /path/to/new/output
Optional --movie-case, --energy-case, --distribution-case use existing cases.
No derived temperatures, magnetic flux, or published spectrum conversions.
"""
import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from kglobal_analysis import KGlobalCase
from kglobal_analysis.analysis import index_cut, spacetime
from kglobal_analysis.animation import iter_movie_frames, render_frame_sequence
from kglobal_analysis.plotting import (
    plot_scalar_map, plot_spacetime, plot_energy_overlay, plot_distribution,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--movie-case', type=Path)
    parser.add_argument('--energy-case', type=Path)
    parser.add_argument('--distribution-case', type=Path)
    args = parser.parse_args()
    # Explicit headless choice for this batch example only, not package globals.
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    args.output.mkdir(parents=True, exist_ok=False)
    if args.movie_case:
        data = KGlobalCase(args.movie_case).movie('bx', byteorder='little').to_xarray()
        chosen = data.time.values[[0, -1]]
    else:
        x, y = np.indices((12, 7))
        data = xr.DataArray(np.stack([np.sin(x/3+i)*np.cos(y/4) for i in range(3)]),
                            dims=('time', 'x', 'y'), coords={'time': [1.2, 1.4, 1.6]},
                            name='synthetic scalar')
        chosen = [1.2, 1.6]

    frame = next(iter_movie_frames(data, times=[chosen[0]])).data
    result = plot_scalar_map(frame, horizontal='x', vertical='y')
    result.figure.savefig(args.output/'map.png')
    plt.close(result.figure)

    # The caller explicitly chooses y index 0, not a presumed sheet center.
    line = index_cut(frame, line_axis='x', fixed_indices={'y': 0})
    fig, ax = plt.subplots()
    ax.plot(np.arange(line.size), line.values)
    ax.set(xlabel='x index', ylabel='stored scalar', title='Explicit y index 0')
    fig.savefig(args.output/'cut.png')
    plt.close(fig)

    # Supplied scalar contour, not reconstructed psi or magnetic field lines.
    contour = xr.ones_like(frame).cumsum('x') + xr.ones_like(frame).cumsum('y')
    result = plot_scalar_map(frame, horizontal='x', vertical='y', contours=contour)
    result.figure.savefig(args.output/'supplied-contours.png')
    plt.close(result.figure)
    del frame, line, contour, result

    # Fixed limits are user policy. Two large panels may retain >512 MiB of data.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for event, ax in zip(iter_movie_frames(data, times=chosen), axes):
        plot_scalar_map(event.data, horizontal='x', vertical='y', ax=ax,
                        normalization='signed', vmin=-2, vmax=2, title=f't={event.time}')
    del event
    fig.savefig(args.output/'panels.png')
    plt.close(fig)
    del fig, axes

    st = spacetime(data, line_axis='x', fixed_indices={'y': 0}, times=chosen)
    result = plot_spacetime(st, line_axis='x')
    result.figure.savefig(args.output/'spacetime.png')
    plt.close(result.figure)
    render_frame_sequence(data, args.output/'frames', horizontal='x', vertical='y',
                          times=chosen, color_policy='global')

    if args.energy_case:
        case = KGlobalCase(args.energy_case)
        spectra = {s: case.energy_spectrum(s, checkpoint='016') for s in ('electron', 'ion')}
        results = plot_energy_overlay(spectra, xscale='log', yscale='log')
        results[0].figure.savefig(args.output/'energy-estimators.png')
        plt.close(results[0].figure)
        # Explicit example-level transformed coordinate, without changing y.
        spectrum = spectra['electron']
        N = 3  # User-supplied illustration; not inferred from the checkpoint.
        x = np.array(spectrum.centers, copy=True) / 2**N
        transform_provenance = {'operation': 'energy_center / 2**N', 'N': N,
                                'ordinate': 'unchanged legacy estimator; no Jacobian'}
        fig, ax = plt.subplots()
        ax.plot(x, spectrum.values)
        ax.set(xlabel=f'energy center / 2^{N} (user transform)', ylabel='legacy energy estimator')
        fig.savefig(args.output/'transformed-coordinate.png', metadata={'Description': str(transform_provenance)})
        plt.close(fig)
    if args.distribution_case:
        case = KGlobalCase(args.distribution_case)
        for label, dist in (
            ('velocity', case.parallel_perpendicular_velocity_distribution('electron', checkpoint='016')),
            ('position', case.position_parallel_velocity_distribution('ion', position_axis='x', checkpoint='016')),
        ):
            result = plot_distribution(dist, coordinate_mode='centers', normalization='log')
            result.figure.savefig(args.output/f'{label}-bin-mass.png')
            plt.close(result.figure)


if __name__ == '__main__':
    main()
