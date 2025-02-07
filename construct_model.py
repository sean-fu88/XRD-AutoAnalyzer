from importlib.util import module_for_loader
from autoXRD import cnn, spectrum_generation, solid_solns, tabulate_cifs
import numpy as np
import os
import sys
import pymatgen as mg

def make_model(model_name, is_pdf=False, max_texture = 0.5,
                min_domain_size=5.0, max_domain_size = 30.0,
                max_strain = 0.03, max_shift = 0.5, num_spectra = 50,
                min_angle=10.0, max_angle = 100.0, num_epochs = 2,
                separate = True, skip_filter = False, include_elems = True,
                sys_args=None):
    
    # max_texture = 0.5 # default: texture associated with up to +/- 50% changes in peak intensities
    # min_domain_size, max_domain_size = 5.0, 30.0 # default: domain sizes ranging from 5 to 30 nm
    # max_strain = 0.03 # default: up to +/- 3% strain
    # max_shift = 0.5 # default: up to +/- 0.5 degrees shift in two-theta
    # num_spectra = 50 # Number of spectra to simulate per phase
    # min_angle, max_angle = 10.0, 80.0
    # num_epochs = 2
    # separate = True
    # skip_filter = False
    # include_elems = True
    for arg in sys_args:
        if '--ignore_elems' in arg:
            include_elems = False
        if '--mixed_artifacts' in arg:
            separate = False

    if not skip_filter:
        # Filter CIF files to create unique reference phases
        assert 'All_CIFs' in os.listdir('.'), 'No All_CIFs directory was provided. Please create or use --skip_filter'
        assert 'References' not in os.listdir('.'), 'References directory already exists. Please remove or use --skip_filter'
        tabulate_cifs.main('All_CIFs', 'References', include_elems)

    else:
        assert 'References' in os.listdir('.'), '--skip_filter was specified, but no References directory was provided'

    if '--include_ns' in sys.argv:
        # Generate hypothetical solid solutions
        solid_solns.main('References')
    print("got to spectrum")
    # Simulate and save augmented XRD spectra
    xrd_obj = spectrum_generation.SpectraGenerator(reference_dir= 'References',num_spectra= num_spectra,max_texture= max_texture,min_domain_size = min_domain_size, max_domain_size = max_domain_size, max_strain = max_strain, max_shift = max_shift, min_angle = min_angle, max_angle= max_angle, separate= separate, is_pdf=is_pdf)
    print("initiated")
    xrd_specs = xrd_obj.augmented_spectra
    np.save('XRD', xrd_specs)
    print("onto cnn")
    # Train, test, and save the CNN
    cnn.main(xrd_specs, is_pdf=is_pdf, fmodel=model_name, num_epochs=num_epochs, testing_fraction=0.2)

if __name__ == '__main__':

    max_texture = 0.5 # default: texture associated with up to +/- 50% changes in peak intensities
    min_domain_size, max_domain_size = 5.0, 30.0 # default: domain sizes ranging from 5 to 30 nm
    max_strain = 0.03 # default: up to +/- 3% strain
    max_shift = 0.5 # default: up to +/- 0.5 degrees shift in two-theta
    num_spectra = 50 # Number of spectra to simulate per phase
    min_angle, max_angle = 10.0, 80.0
    num_epochs = 2
    separate = True
    skip_filter = False
    include_elems = True
    for arg in sys.argv:
        if '--max_texture' in arg:
            max_texture = float(arg.split('=')[1])
        if '--min_domain_size' in arg:
            min_domain_size = float(arg.split('=')[1])
        if '--max_domain_size' in arg:
            max_domain_size = float(arg.split('=')[1])
        if '--max_strain' in arg:
            max_strain = float(arg.split('=')[1])
        if '--max_shift' in arg:
            max_shift = float(arg.split('=')[1])
        if '--num_spectra' in arg:
            num_spectra = int(arg.split('=')[1])
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])
        if '--num_epochs' in arg:
            num_epochs = int(arg.split('=')[1])
        if '--skip_filter' in arg:
            skip_filter = True
        if '--ignore_elems' in arg:
            include_elems = False
        if '--mixed_artifacts' in arg:
            separate = False

    if not skip_filter:
        # Filter CIF files to create unique reference phases
        assert 'All_CIFs' in os.listdir('.'), 'No All_CIFs directory was provided. Please create or use --skip_filter'
        assert 'References' not in os.listdir('.'), 'References directory already exists. Please remove or use --skip_filter'
        tabulate_cifs.main('All_CIFs', 'References', include_elems)

    else:
        assert 'References' in os.listdir('.'), '--skip_filter was specified, but no References directory was provided'

    if '--include_ns' in sys.argv:
        # Generate hypothetical solid solutions
        solid_solns.main('References')

    # Simulate and save augmented XRD spectra
    xrd_obj = spectrum_generation.SpectraGenerator('References', num_spectra, max_texture, min_domain_size, max_domain_size, max_strain, max_shift, min_angle, max_angle, separate)
    print("done with generation")
    xrd_specs = xrd_obj.augmented_spectra
    np.save('XRD', xrd_specs)

    # Train, test, and save the CNN
    cnn.main(xrd_specs, num_epochs=num_epochs, testing_fraction=0.2)
