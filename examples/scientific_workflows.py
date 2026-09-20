"""Run with --case PATH --output DIR; requires matched movies and the plot extra.

Default: particle-ion density plus flux. NR thermal/firehose examples require
an NR producer; this script never overrides those scientific gates. Optional
reduced products have separate input paths/checkpoint identifiers.
"""
import argparse
from pathlib import Path

import kglobal_analysis as kga
from kglobal_analysis.analysis import spacetime
from kglobal_analysis.animation import render_frame_sequence
from kglobal_analysis.plotting import plot_scalar_map, plot_spacetime, plot_energy_overlay, plot_distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quantity', default='number_density')
    parser.add_argument('--species', choices=('ion', 'electron'), default='ion')
    parser.add_argument('--byteorder', choices=('little', 'big'), default='little')
    parser.add_argument('--events', type=int, nargs='+', default=[0])
    parser.add_argument('--y-index', type=int, required=True, help='Explicit line selection; no inferred centerline')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--energy-case')
    parser.add_argument('--distribution-case')
    parser.add_argument('--checkpoint', help='Reduced-product identifier, never a movie time')
    args = parser.parse_args()
    case = kga.KGlobalCase(args.case)
    for item in kga.particle_quantity_catalog(case.particle_moment_profile()):
        print(item)
    # This graph needs no geometry; useful independently for local scalar algebra.
    scalar = kga.particle_quantity_series(case, args.quantity, species=args.species, byteorder=args.byteorder)
    print('Lazy scalar:', scalar.name, scalar.sizes)
    workflow = kga.particle_map_series(case, args.quantity, species=args.species, byteorder=args.byteorder)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for event in args.events:
        if not 0 <= event < workflow.sizes['time']:
            parser.error('event index outside the actual movie sequence')
        frame = workflow.isel(time=event).compute(scheduler='synchronous')
        view = plot_scalar_map(frame.field, horizontal='x', vertical='y', contours=frame.psi)
        view.figure.savefig(out/f'map_{event:06d}.png')
        view.figure.clear()
        del frame, view
    times = workflow.time.values[args.events].tolist()
    xt = spacetime(workflow.field, line_axis='x', fixed_indices={'y': args.y_index}, times=times)
    view = plot_spacetime(xt, line_axis='x')
    view.figure.savefig(out/'spacetime.png')
    view.figure.clear()
    if args.render:
        render_frame_sequence(workflow.field, out/'frames', horizontal='x', vertical='y',
                              times=times, color_policy='global')
    if args.energy_case or args.distribution_case:
        if args.checkpoint is None:
            parser.error('--checkpoint is required for reduced products')
    if args.energy_case:
        reduced = kga.KGlobalCase(args.energy_case)
        views = plot_energy_overlay({s: reduced.energy_spectrum(s, checkpoint=args.checkpoint)
                                     for s in ('electron', 'ion')})
        view = views[0]
        view.figure.savefig(out/'energy_estimators.png')
        view.figure.clear()
    if args.distribution_case:
        reduced = kga.KGlobalCase(args.distribution_case)
        products = {
            'parperp': reduced.parallel_perpendicular_velocity_distribution(args.species, checkpoint=args.checkpoint),
            'positionpar': reduced.position_parallel_velocity_distribution(args.species, position_axis='x', checkpoint=args.checkpoint),
        }
        for name, product in products.items():
            view = plot_distribution(product)
            view.figure.savefig(out/f'{name}_bin_mass.png')
            view.figure.clear()


if __name__ == '__main__':
    main()
