from construct_model import *
import sys
if __name__ == '__main__':
    name = "ModelTheta" + str(60.0) + ".h5"
    make_model(sys_args= sys.argv, is_pdf=True ,max_angle=float(60.0), model_name=name, skip_filter=True)
    print("finished " + name)