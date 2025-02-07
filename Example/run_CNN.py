from autoXRD import spectrum_analysis, visualizer, quantifier
import sys
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':

    start = time.time()

    max_phases = 4 # default: a maximum 4 phases in each mixture
    cutoff_intensity = 5 # default: ID all peaks with I >= 5% maximum spectrum intensity
    min_conf = 25.0 # Minimum confidence included in predictions
    wavelength = 'CuKa' # default: spectra was measured using Cu K_alpha radiation
    unknown_threshold = 25.0 # default: raise warning when peaks with >= 25% intensity are unknown
    show_reduced = False # Whether to plot reduced spectrum (after subtraction of known phases)
    min_angle, max_angle = 10.0, 80.0
    for arg in sys.argv:
        if '--max_phases' in arg:
            max_phases = int(arg.split('=')[1])
        if '--cutoff_intensity' in arg:
            cutoff_intensity = int(arg.split('=')[1])
        if '--min_conf' in arg:
            min_conf = int(arg.split('=')[1])
        if '--wavelength' in arg:
            wavelength = float(arg.split('=')[1])
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])
        if '--unkown_thresh' in arg:
            unknown_threshold = float(arg.split('=')[1])
        if '--show_reduced' in arg:
            show_reduced = True

    spectrum_names, predicted_phases, confidences, backup_phases, scale_factors, reduced_spectra = spectrum_analysis.main('Spectra', 'References',
        max_phases, cutoff_intensity, min_conf, wavelength, min_angle, max_angle)

    for (spectrum_fname, phase_set, confidence, backup_set, heights, final_spectrum) in zip(spectrum_names, predicted_phases, confidences, backup_phases, scale_factors, reduced_spectra):

        # Print phase ID info
        print('Filename: %s' % spectrum_fname)
        print('Predicted phases: %s' % phase_set)
        print('Confidence: %s' % confidence)

        # If there are unknown peaks with intensity > threshold, raise warning
        if 'None' not in phase_set:
            remaining_I = max(final_spectrum)
            if remaining_I > unknown_threshold:
                print('WARNING: some peaks (I ~ %s%%) were not identified.' % int(remaining_I))
        else:
            print('WARNING: no phases were identified')

        # If this option is specified, show backup predictions (2nd-most probable phases)
        if '--show_backups' in sys.argv:
            print('Alternative phases: %s' % backup_set)

        if ('--plot' in sys.argv) and ('None' not in phase_set):

            save = False
            if '--save' in sys.argv:
                save = True

            # Format predicted phases into a list of their CIF filenames
            phasenames = ['%s.cif' % phase for phase in phase_set]

            # Plot measured spectrum with line profiles of predicted phases
            visualizer.main('Spectra', spectrum_fname, phasenames, heights, final_spectrum, min_angle, max_angle, wavelength, save, show_reduced)

        if ('--weights' in sys.argv) and ('None' not in phase_set):

            # Format predicted phases into a list of their CIF filenames
            phasenames = ['%s.cif' % phase for phase in phase_set]

            # Get weight fractions
            weights = quantifier.main('Spectra', spectrum_fname, phasenames, heights, min_angle, max_angle, wavelength)
            weights = [round(val, 2) for val in weights]
            print('Weight fractions: %s' % weights)

    end = time.time()

    elapsed_time = round(end - start, 1)
    print('Total time: %s sec' % elapsed_time)
