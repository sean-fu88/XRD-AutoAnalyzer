from construct_model import *
import sys
if __name__ == '__main__':
    for x in range(60,110,10):
        name = "ModelTheta" + str(x) + ".h5"
        make_model(sys_args= sys.argv, is_pdf = False, max_angle=float(x), model_name=name, skip_filter=True)
        make_model(sys_args= sys.argv, is_pdf=True ,max_angle=float(x), model_name=name, skip_filter=True)
        print("finished " + name)