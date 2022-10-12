from autoXRD.spectrum_generation import strain_shifts, uniform_shifts, intensity_changes, peak_broadening, impurity_peaks, mixed
from autoXRD.Combined_Analysis import XRDtoPDF
import pymatgen as mg
import numpy as np
import os
import multiprocessing
from multiprocessing import Pool, Manager
from pymatgen.core import Structure


class SpectraGenerator(object):
    """
    Class used to generate augmented xrd spectra
    for all reference phases
    """

    def __init__(self, reference_dir, is_pdf=False, num_spectra=50, max_texture=0.6, min_domain_size=1.0, max_domain_size=100.0, max_strain=0.04, max_shift=0.25, impur_amt=70.0, min_angle=10.0, max_angle=80.0, separate=True):
        """
        Args:
            reference_dir: path to directory containing
                CIFs associated with the reference phases
        """
        self.num_cpu = multiprocessing.cpu_count()
        self.ref_dir = reference_dir
        self.num_spectra = num_spectra
        self.max_texture = max_texture
        self.min_domain_size = min_domain_size
        self.max_domain_size = max_domain_size
        self.max_strain = max_strain
        self.max_shift = max_shift
        self.impur_amt = impur_amt
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.separate = separate
        self.is_pdf = is_pdf

    def augment(self, phase_info):
        """
        For a given phase, produce a list of augmented XRD spectra.
        By default, 50 spectra are generated per artifact, including
        peak shifts (strain), peak intensity change (texture), and
        peak broadening (small domain size).

        Args:
            phase_info: a list containing the pymatgen structure object
                and filename of that structure respectively.
        Returns:
            patterns: augmented XRD spectra
            filename: filename of the reference phase
        """
        print("got here?")
        struc, filename = phase_info[0], phase_info[1]
        patterns = []
        print("start_augment")
        if self.separate:
            patterns += strain_shifts.main(struc, self.num_spectra, self.max_strain, self.min_angle, self.max_angle)
            patterns += uniform_shifts.main(struc, self.num_spectra, self.max_shift, self.min_angle, self.max_angle)
            patterns += peak_broadening.main(struc, self.num_spectra, self.min_domain_size, self.max_domain_size, self.min_angle, self.max_angle)
            patterns += intensity_changes.main(struc, self.num_spectra, self.max_texture, self.min_angle, self.max_angle)
            patterns += impurity_peaks.main(struc, self.num_spectra, self.impur_amt, self.min_angle, self.max_angle)
        else:
            patterns += mixed.main(struc, 5*self.num_spectra, self.max_shift, self.max_strain, self.min_domain_size, self.max_domain_size,  self.max_texture, self.impur_amt, self.min_angle, self.max_angle)
        print("done with patterns")
        if self.is_pdf:
            print("is a pdf")
            pdf_specs = []
            i = 1
            # getting 250 xrd spectrums
            # flattened_pdf = []
            # for xrd_pattern in patterns:
            #     xrd_pattern = np.array(xrd_pattern).flatten()
            #     flattened_pdf = flattened_pdf.append(xrd_pattern)
            # xrd_patterns = [xrd_pattern.flatten() for xrd_pattern in patterns]
            # #giving XRDtoPDF a list 
            # pdf_patterns = XRDtoPDF(xrd_patterns, self.min_angle, self.max_angle) 
            for xrd_pattern in patterns:
                print("enter for loop" +str(i) )
                xrd_pattern = np.array(xrd_pattern).flatten()
                print("to combined")
                pdf = XRDtoPDF(xrd_pattern, self.min_angle, self.max_angle)
                print("done xrd2pdf")
                pdf = [[val] for val in pdf]
                pdf_specs.append(pdf)
                i+=1
            return (pdf_specs, filename)
        return (patterns, filename)

    @property
    def augmented_spectra(self):
        print("augmented time")
        phases = []
        for filename in sorted(os.listdir(self.ref_dir)):
            phases.append([Structure.from_file('%s/%s' % (self.ref_dir, filename)), filename])
        print(phases)
        print("stage1")
        grouped_xrd = []
        for ph in phases:
            grouped_xrd.append(self.augment(ph))
        print("stage2")
        sorted_xrd = sorted(grouped_xrd, key=lambda x: x[1]) ## Sort by filename
        print("stage3")
        sorted_spectra = [group[0] for group in sorted_xrd]
        print("done with augmented_spectra")
        return np.array(sorted_spectra)


