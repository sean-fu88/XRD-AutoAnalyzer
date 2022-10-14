from run_CNN import *
import sys
if __name__ == '__main__':
    run_CNNs(model_name="ModelTheta60.0.h5",reference_directory = "References",spectra_directory="Spectra", max_angle=60.0 )