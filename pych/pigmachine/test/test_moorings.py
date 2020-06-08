
from pych.pigmachine.moorings import *
from rosypig import Observable, ObservingSystem
from rosypig.test.test_common import get_pig_grid

_test_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.abspath(os.path.join(_test_dir,'../../'))
_write_dir = os.path.abspath(os.path.join(_test_dir,'mooring_test'))
#_read_dir = os.path.abspath(os.path.join(_test_dir,'observable_expected'))

def test_get_loc_from_mask(get_pig_grid):
    """make sure input mask and resulting observable are equal"""


    ds = get_pig_grid
    
    # --- Define moorings as in rand_array/ study...
    mask_expected = make_mooring_array_mask(ds)
    lon,lat,depth = get_loc_from_mask(ds,mask_expected)

    # make test observable
    tobs = Observable('temp',ds)
    for x,y,z in zip(lon,lat,depth):
        tobs.add_loc_to_masks(x,y,z)

    assert tobs.maskC.reset_coords(drop=True).equals(mask_expected.reset_coords(drop=True))

def test_observingsystem_write_binaries(get_pig_grid):
    """make sure written file is same as original"""

    ds = get_pig_grid
    tobs = make_mooring_obs(ds)
    obsys = ObservingSystem([tobs])

    # write binaries
    obsys.set_write_dir(_WRITE_DIR)
    obsys.make_datafile(nrecs=2)
    obsys.write_binaries(use_mds=True)

    # read and check each file
    for fbase,expected_arr in zip(
    [tobs.uncertainty_filename,tobs.mult_filename,tobs.data_filename],
    [tobs.sigma.values,tobs.mult.values,tobs.data]):

        test_arr = rdmds(f'{obsys.write_dir}/{fbase}')
        assert np.all(test_arr == expected_arr)

        # clean up
        for suffix in ['.meta','.data','']:
            os.remove(f'{obsys.write_dir}/{fbase}{suffix}')

def test_observingsystem_write_updated_uncertainty(get_pig_grid):
    """make sure when an observable's sigma field gets updated,
    ObservingSystem writes out that updated field"""

    ds = get_pig_grid
    
    # --- Define moorings as in rand_array/ study...
    mask_expected = make_mooring_array_mask(ds)
    lon,lat,depth = get_loc_from_mask(ds,mask_expected)

    # make test observable
    tobs = Observable('temp',ds)
    obsys = ObservingSystem([tobs])

    for x,y,z in zip(lon,lat,depth):
        tobs.add_loc_to_masks(x,y,z)


    # for fun, make sigma something crazy
    sigma_expected = tobs.unpack(np.random.rand(tobs.get_n_wet())) 
    tobs.sigma = sigma_expected

    # Make Observing system, write binaries, 
    obsys.set_write_dir(_write_dir)
    obsys.make_datafile(nrecs=1)
    obsys.write_binaries(use_mds=True)

    # Read in uncertainties and verify
    sigma_test = rdmds(f'{obsys.write_dir}/{tobs.uncertainty_filename}')

    assert np.all(sigma_test == sigma_expected.values)
    assert np.all(sigma_test == tobs.sigma.values)

    # cleanup
    for fbase in [tobs.uncertainty_filename,tobs.data_filename]:
        for suffix in ['.meta','.data','']:
            os.remove(f'{obsys.write_dir}/{fbase}{suffix}')

