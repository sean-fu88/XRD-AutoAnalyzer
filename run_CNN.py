import re
from autoXRD import spectrum_analysis, visualizer, quantifier
import sys
import numpy as np
import pymatgen as mg
import matplotlib.pyplot as plt
import time


def merge_predictions(preds, confs, cutoff=50.0, max_phases=3):
    """
    Aggregate predictions through an ensemble approach
    whereby each phase is weighted by its confidence.
    """
    avg_soln = {}
    for cmpd, cf in zip(preds, confs):
        if cmpd not in avg_soln.keys():
            avg_soln[cmpd] = [cf]
        else:
            avg_soln[cmpd].append(cf)

    unique_preds, avg_confs = [], []
    for cmpd in avg_soln.keys():
        unique_preds.append(cmpd)
        num_zeros = 2 - len(avg_soln[cmpd])
        avg_soln[cmpd] += [0.0]*num_zeros
        avg_confs.append(np.mean(avg_soln[cmpd]))

    info = zip(unique_preds, avg_confs)
    info = sorted(info, key=lambda x: x[1])
    info.reverse()

    unique_cmpds, unique_confs = [], []
    for cmpd, cf in info:
        if (len(unique_cmpds) < max_phases) and (cf > cutoff):
            unique_cmpds.append(cmpd)
            unique_confs.append(cf)

    return unique_cmpds, unique_confs


if __name__ == '__main__':

    start = time.time()

    max_phases = 4 # default: a maximum 4 phases in each mixture
    cutoff_intensity = 5 # default: ID all peaks with I >= 5% maximum spectrum intensity
    wavelength = 'CuKa' # default: spectra was measured using Cu K_alpha radiation
    min_angle, max_angle = 10.0, 100.0
    is_pdf = False
    spectra_directory = 'Spectra'
    reference_directory = 'References'
    model_path_pdf = "PDF_Model.h5"
    model_path_xrd = "XRD_Model.h5"
    for arg in sys.argv:
        if '--max_phases' in arg:
            max_phases = int(arg.split('=')[1])
        if '--cutoff_intensity' in arg:
            cutoff_intensity = int(arg.split('=')[1])
        if '--wavelength' in arg:
            wavelength = float(arg.split('=')[1])
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])
        if '--inc_pdf' in arg:
            is_pdf = True
        if '--spectra_directory' in arg:
            spectra_directory = str(arg.split('=')[1])
        if '--reference_directory' in arg:
            reference_directory = str(arg.split('=')[1])
        if '--model_path' in arg:
            model_name = str(arg.split('=')[1])
            model_path_pdf = f"PDF_{model_name}"
            model_path_xrd = f"XRD_{model_name}"
    pdf_spectrum_names, pdf_predicted_phases, pdf_confidences, pdf_backup_phases, pdf_scale_factors, pdf_reduced_spectra = spectrum_analysis.main(spectra_directory = spectra_directory,
                                                                                           reference_directory=reference_directory,
                                                                                           max_phases=max_phases,
                                                                                           cutoff_intensity=cutoff_intensity,
                                                                                           wavelength =wavelength,
                                                                                           min_angle=min_angle,
                                                                                           max_angle=max_angle,
                                                                                           model_path =model_path_pdf,
                                                                                           is_pdf=True)
    xrd_spectrum_names, xrd_predicted_phases, xrd_confidences, xrd_backup_phases, xrd_scale_factors, xrd_reduced_spectra = spectrum_analysis.main(spectra_directory = spectra_directory,
                                                                                           reference_directory=reference_directory,
                                                                                           max_phases=max_phases,
                                                                                           cutoff_intensity=cutoff_intensity,
                                                                                           wavelength=wavelength,
                                                                                           min_angle=min_angle,
                                                                                           max_angle=max_angle,
                                                                                           model_path=model_path_xrd)

    final_combined_phases, final_combined_confidences = [], []
    zippedXRD = zip(xrd_spectrum_names, xrd_predicted_phases, xrd_confidences)
    zippedPDF = zip(pdf_spectrum_names, pdf_predicted_phases, pdf_confidences)
    sortedZipXRD = sorted(zippedXRD, key = lambda x : x[0])
    sortedZipPDF = sorted(zippedPDF, key = lambda x : x[0])
    sorted_xrd_spectrum_names, sorted_xrd_predicted_phases, sorted_xrd_confidences = list(zip(*sortedZipXRD))
    sorted_pdf_spectrum_names, sorted_pdf_predicted_phases, sorted_pdf_confidences = list(zip(*sortedZipPDF))
    for (xrd_spectrum_fname, xrd_phase_set, xrd_confidence, pdf_spectrum_fname, pdf_phase_set, pdf_confidence) in zip(sorted_xrd_spectrum_names,
                                                                                                                      sorted_xrd_predicted_phases,
                                                                                                                      sorted_xrd_confidences,
                                                                                                                      sorted_pdf_spectrum_names,
                                                                                                                      sorted_pdf_predicted_phases,
                                                                                                                      sorted_pdf_confidences):
        final_combined_phases, final_combined_confidences = [],[]
        if xrd_spectrum_fname != pdf_spectrum_fname:
            break
        if '--all' not in sys.argv:  # By default: only include phases with a confidence > 50%
            final_combined_phases, final_combined_confidences = merge_predictions(xrd_phase_set+pdf_phase_set, xrd_confidence+pdf_confidence, 25.0)
            print('Filename: %s' % pdf_spectrum_fname)
            print('XRD predicted phases: %s' % xrd_phase_set)
            print('XRD Confidence: %s' % xrd_confidence)
            print('PDF predicted phases: %s' % pdf_phase_set)
            print('PDF Confidence: %s' % pdf_confidence)
            print('Merged predicted phases: %s' % final_combined_phases)
            print('Confidence: %s' % final_combined_confidences)

        else:  # If --all is specified, print *all* suspected phases
            final_combined_phases, final_combined_confidences = merge_predictions(xrd_phase_set+pdf_phase_set, xrd_confidence+pdf_confidence, 0, 1000 )
            print('Filename: %s' % pdf_spectrum_fname)
            print('XRD predicted phases: %s' % xrd_phase_set)
            print('XRD Confidence: %s' % xrd_confidence)
            print('PDF predicted phases: %s' % pdf_phase_set)
            print('PDF Confidence: %s' % pdf_confidence)
            print('Merged predicted phases: %s' % final_combined_phases)
            print('Confidence: %s' % final_combined_confidences)

    if ('--plot' in sys.argv) and (pdf_phase_set != 'None'):

            # Format predicted phases into a list of their CIF filenames
        final_phasenames = ['%s.cif' % phase for phase in final_combined_phases]

            # Plot measured spectrum with line profiles of predicted phases
        visualizer.main('Spectra', pdf_spectrum_fname, final_phasenames, min_angle, max_angle, wavelength)

    if ('--weights' in sys.argv) and (pdf_phase_set != 'None'):

            # Format predicted phases into a list of their CIF filenames
        final_phasenames = ['%s.cif' % phase for phase in final_combined_phases]

            # Get weight fractions
        weights = quantifier.main('Spectra', pdf_spectrum_fname, final_phasenames, min_angle, max_angle, wavelength)
        weights = [round(val, 2) for val in weights]
        print('Weight fractions: %s' % weights)

    end = time.time()

    elapsed_time = round(end - start, 1)
    print('Total time: %s sec' % elapsed_time)