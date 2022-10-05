from run_CNN import *
import sys
if __name__ == '__main__':
    for i in range(60, 110, 10):
        model_name = "ModelTheta" + str(i) + ".h5"
        run_CNNs(sys_args= sys.argv, spectra_directory="Patterns/1-Phase", model_name=model_name, max_angle=float(i))
        run_CNNs(sys_args= sys.argv, spectra_directory="Patterns/2-Phase", model_name=model_name, max_angle=float(i))
        run_CNNs(sys_args= sys.argv, spectra_directory="Patterns/3-Phase", model_name=model_name, max_angle=float(i))