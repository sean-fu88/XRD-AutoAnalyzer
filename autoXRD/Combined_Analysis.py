import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.fftpack import dst
import numpy as np
import math
from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from pymatgen.core import Structure
from scipy.fft import rfft
import pymatgen as mg
import warnings
from time import time
np.random.seed(1)

def structureToXRD(strucName):

    for theta2 in [60,100,140,180]:
        angles = np.linspace(10, theta2, 4501)

        cmpd = strucName + '.cif'
        calculator = xrd.XRDCalculator()

        ref_dir = 'References'
        min_angle, max_angle = 10, theta2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # don't print occupancy-related warnings
            struct = Structure.from_file('%s/%s' % (ref_dir, cmpd))
        equil_vol = struct.volume
        pattern = calculator.get_pattern(struct, two_theta_range=(min_angle, max_angle))
        anglesf = pattern.x
        intensitiesf = pattern.y

        steps = np.linspace(min_angle, max_angle, 4501)

        signals = np.zeros([len(anglesf), steps.shape[0]])

        for i, ang in enumerate(anglesf):
            # Map angle to closest datapoint step
            idx = np.argmin(np.abs(ang - steps))
            signals[i, idx] = intensitiesf[i]

        # Convolute every row with unique kernel
        # Iterate over rows; not vectorizable, changing kernel for every row
        domain_size = 10.0
        step_size = (max_angle - min_angle) / 4501
        for i in range(signals.shape[0]):
            row = signals[i, :]
            ang = steps[np.argmax(row)]
            calculator = xrd.XRDCalculator()

            ## Calculate FWHM based on the Scherrer equation
            K = 0.9  ## shape factor
            wavelength = calculator.wavelength * 0.1  ## angstrom to nm
            theta = np.radians(ang / 2.)  ## Bragg angle in radians
            beta = (K * wavelength) / (np.cos(theta) * domain_size)  # in radians

            ## Convert FWHM to std deviation of gaussian
            sigma = np.sqrt(1 / (2 * np.log(2))) * 0.5 * np.degrees(beta)
            std_dev = sigma ** 2
            # Gaussian kernel expects step size 1 -> adapt std_dev
            signals[i, :] = gaussian_filter1d(row, np.sqrt(std_dev) * 1 / step_size,
                                              mode='constant')

        # Combine signals
        signal = np.sum(signals, axis=0)

        # Normalize signal
        norm_signal = 1.0 * signal / max(signal)

        noise = np.random.normal(0, 0.001, 4501)
        norm_signal = norm_signal + noise
        intensities = norm_signal
        #writing to xy
        with open(strucName + str(theta2) + '.xy', 'w+') as f:
            for (xval, yval) in zip(angles, intensities):
                f.write('%s %s\n' % (xval, yval))
        #convert to pdf



def XRDtoPDF(pattern, min_angle, max_angle):
    """
    r: an instance of a radius in real space (float)
    S: full scattering function (list)
    Q: full span of reciprocal space (list)
    """
    print("got here")
    st_time = time()
    x= np.linspace(min_angle, max_angle, 4501)
    Q, S = [], []
    for i in range(len(pattern)):
        two_theta = float(x[i])
        theta = two_theta / 2.
        q = 4 * math.pi * math.sin(math.radians(theta)) / 1.5406
        Q.append(q)
        S.append(float(pattern[i]))
    pdf = []
    ed_time = time()
    print("inside time", ed_time-st_time)
    R = np.linspace(0, 20, 4501)
    for r in R:
        integrand = []
        for (s, q) in zip(S, Q):
            integrand.append(q * (s - 0.0) * math.sin(q * r))
        pdf.append(2 * simps(integrand, Q) / math.pi)
    print("check two xrd2pdf")
    struc = pdf.copy()
    return Q, S, struc
