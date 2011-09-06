"""Load the published SPT point source raw flux catalogs, deboost and compare.
"""
import numpy as np
import process_catalog as process
import self_describing as sd
import utilities as utils
import shelve

# SPT deboosting parameters
# num_X is number of grid points
# cov_calibration is the fractional covariance contribution due to beam and
# calibration uncertainty plus covariance from background sources.
param = {
  "percentiles": [0.16, 0.5, 0.84],
  "num_flux": 800,  # 800 seems like reasonable binning, 2000 fine
  "num_alpha": 800,
  "prior_alpha": [-3., 5.],
  "range_alpha": [-5., 5.],
  "cov_calibration": [[0.00165700, 0.00129600], [0.00129600, 0.00799300]],
  "freq1": 152.,
  "freq2": 219.5,
  "omega_prior": 1.,
  "spectral_threshold": 1.66,
  "flux1name": "flux150",
  "flux2name": "flux220",
  "sigma1name": "sigma150",
  "sigma2name": "sigma220",
  "sigma12name": None,
  "keyfield_name": "index",
  #"catalog_filename": "../data/source_catalog_vieira09_3sigma.dat",
  "catalog_filename": "../data/source_catalog_vieira09_first200.dat",
  "neglect_calerror": False,
  "verbose": True
}

# suggestions for calculating cov_calibration for other surveys:
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
                                                            param,
                                                        use_spt_model=False)

    outputshelve = shelve.open(outfilename, flag="n")
    outputshelve.update(augmented_catalog)
    outputshelve.close()


def compare_catalogs(outfilename, compfilename, septol=1e-3):
    """compare a deboosted catalog with one in literature"""
    augmented_catalog = shelve.open(outfilename)
    comp_catalog = sd.load_selfdescribing_numpy(compfilename,
                                                swaps=translate,
                                                verbose=param['verbose'])

    (cra, cdec, csnr150, csnr220) = (comp_catalog[:]["ra"],
                                     comp_catalog[:]["dec"],
                                     comp_catalog[:]["SNR150"],
                                     comp_catalog[:]["SNR220"])

    for srcname in augmented_catalog:
        entry = augmented_catalog[srcname]
        (ra, dec) = (entry["ra"],
                     entry["dec"])
        dra = cra - ra
        ddec = cdec - dec
        delta = np.sqrt(dra * dra + ddec * ddec)
        minsep = np.min(delta)
        if (np.min(delta) > septol):
            print "no associated source in comparison catalog for index:" + \
                  repr(srcname)
            break

        comp_index = np.where(delta == minsep)[0][0]
        orig = comp_catalog[comp_index]
        fp1 = param['flux1name']
        fp2 = param['flux2name']
        orig_flux1 = np.array([orig["S_150d"],
                               orig["S_150d_up"],
                               orig["S_150d_down"]])

        orig_flux2 = np.array([orig["S_220d"],
                               orig["S_220d_up"],
                               orig["S_220d_down"]])

        orig_alpha = np.array([orig["d_alpha"],
                               orig["d_alpha_up"],
                               orig["d_alpha_down"]])

        print "-" * 80
        # can optionally print "_posterior" and "_posterior_swap" for checking
        print srcname, comp_index, cra[comp_index], ra, \
              cdec[comp_index], dec, minsep, \
              csnr150[comp_index], csnr220[comp_index]

        print "S1" + "-" * 60
        print utils.pm_error(entry[fp1 + "_posterior_det"] * 1000., "%5.3g")
        print orig_flux1
        if (np.all(orig_flux1 > 0)):
            print (np.array(utils.pm_vector(entry[fp1 + "_posterior_det"] * \
                    1000.)) - orig_flux1) / orig_flux1 * 100.

        print "S2" + "-" * 60
        print utils.pm_error(entry[fp2 + "_posterior_det"] * 1000., "%5.3g")
        print orig_flux2
        if (np.all(orig_flux2 > 0)):
            print (np.array(utils.pm_vector(entry[fp2 + "_posterior_det"] * \
                    1000.)) - orig_flux2) / orig_flux2 * 100.

        print "ind" + "-" * 69
        print utils.pm_error(entry["alpha_posterior"], "%5.3g")
        print orig_alpha
        if (np.all(orig_alpha > 0)):
            print (np.array(utils.pm_vector(entry["alpha_posterior"])) -
                  orig_alpha) / orig_alpha * 100.

        print "P(a>t) new: " + repr(entry["prob_exceed_det"]) + \
              " old: " + repr(orig["palphagt1"])

    augmented_catalog.close()


if __name__ == '__main__':
    output_catalog = "augmented_spt_catalog.shelve"
    augment_catalog(output_catalog)
    compare_catalogs(output_catalog,
                     "../data/source_table_vieira09_3sigma.dat")
