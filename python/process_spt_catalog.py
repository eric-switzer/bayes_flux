"""Load the published SPT point source raw flux catalogs, deboost and compare
"""
import numpy as np
import process_catalog as process
import self_describing as sd
import utilities as utils
import shelve

# SPT deboosting parameters
# num_X is number of grid points
# cov_calbeam is the covariance contribution due to beam
# and calibration uncertainty plus covariance from background
# sources (found from simulation)
param = {
  "percentiles": [0.16, 0.5, 0.84],
  "num_flux": 800,
  "num_alpha": 800,
  "prior_alpha": [-5., 5.],
  "range_alpha": [-5., 5.],
  "cov_calbeam": [[0.00165700, 0.00129600], [0.00129600, 0.00799300]],
  "freq1": 152.,
  "freq2": 219.5,
  "omega_prior": 1.,
  "flux1name": "flux150",
  "flux2name": "flux220",
  "sigma1name": "sigma150",
  "sigma2name": "sigma220",
  "keyfield_name": "index",
  #"catalog_filename": "../data/source_catalog_vieira09_3sigma.dat",
  "catalog_filename": "../data/source_catalog_vieira09_first200.dat",
  "neglect_calerror": False,
  "verbose": True
}

# suggestions for calculating cov_calbeam for other surveys:
# * use flux recovery simulations to estimate the calibration/beam error
#    covariance
#    -- use a set of PSFs drawn from the error model and look at the scatter
#       in the flux.
#    -- may also do this in Fourier space or multipole space
# * Find the covariance of background fluctuations by applying the matched
#   finding/photometry filter, and calculating the covariance between bands.
# The map could either be:
#    -- empirical noise estimate (e.g. difference map) + CMB etc. realization
#        + source realization up to the threshold
#    -- the real map with sources removed down to the threshold

# rename some fields from the self-describing input ascii catalog
translate = {'ID': param['keyfield_name'],
             'S_150': param['flux1name'],
             'noise_150': param['sigma1name'],
             'S_220': param['flux2name'],
             'noise_220': param['sigma2name']}


def augment_catalog(outfilename):
    """calculate deboosting for the two-band catalog and write an output
    shelve to outfilename"""
    input_catalog = sd.load_selfdescribing_numpy(param['catalog_filename'],
                                                 swaps=translate,
                                                 verbose=param['verbose'])

    # convert flux150, sigma150, flux220, sigma220 from mJy to Jy
    input_catalog[:][param['flux1name']] /= 1000.
    input_catalog[:][param['sigma1name']] /= 1000.
    input_catalog[:][param['flux2name']] /= 1000.
    input_catalog[:][param['sigma2name']] /= 1000.

    augmented_catalog = process.process_ptsrc_catalog_alpha(input_catalog,
                                                            param)

    # TODO: delete shelve if it already exists
    outputshelve = shelve.open(outfilename)
    outputshelve.update(augmented_catalog)
    outputshelve.close()


def compare_catalogs(outfilename, compfilename, septol=1e-3):
    """compare a deboosted catalog with one in literature"""
    augmented_catalog = shelve.open(outfilename)
    comp_catalog = sd.load_selfdescribing_numpy(compfilename,
                                                swaps=translate,
                                                verbose=param['verbose'])

    (cra, cdec) = (comp_catalog[:]["ra"], comp_catalog[:]["dec"])
    for srcname in augmented_catalog:
        (ra, dec) = (augmented_catalog[srcname]["ra"],
                     augmented_catalog[srcname]["dec"])
        dra = cra - ra
        ddec = cdec - dec
        delta = np.sqrt(dra * dra + ddec * ddec)
        if (np.min(delta) > septol):
            print "no associated source in comparison catalog for index:" + \
                  repr(srcname)
            break

        comp_index = np.where(delta == np.min(delta))[0][0]
        comp_entry = comp_catalog[comp_index]
        print "-" * 80
        print srcname, comp_index
        print utils.pm_error(augmented_catalog[srcname][param['flux1name'] + "_posterior"] * 1000., "%5.3g")
        print utils.pm_error(augmented_catalog[srcname][param['flux1name'] + "_posterior_swap"] * 1000., "%5.3g")
        print comp_entry["S_150d"], comp_entry["S_150d_down"], comp_entry["S_150d_up"]
        print utils.pm_error(augmented_catalog[srcname][param['flux2name'] + "_posterior"] * 1000., "%5.3g")
        print utils.pm_error(augmented_catalog[srcname][param['flux2name'] + "_posterior_swap"] * 1000., "%5.3g")
        print comp_entry["S_220d"], comp_entry["S_220d_down"], comp_entry["S_220d_up"]

        # do an inefficient search over a published catalog (assuming no common
        # indices exist

    augmented_catalog.close()


if __name__ == '__main__':
    output_catalog = "augmented_spt_catalog.shelve"
    #augment_catalog(output_catalog)
    compare_catalogs(output_catalog,
                     "../data/source_table_vieira09_3sigma.dat")
