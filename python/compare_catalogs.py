"""Load the published SPT point source raw flux catalogs, deboost and compare.
"""
import numpy as np
import process_catalog as process
import self_describing as sd
import utilities as utils
import shelve


def compare_catalogs(params, translate, septol=1e-3):
    """compare a deboosted catalog with one in literature"""

    augmented_catalog = shelve.open(params['augmented_catalog'], flag="r")
    comp_catalog = sd.load_selfdescribing_numpy(params['comparison_catalog'],
                                                swaps=translate,
                                                verbose=params['verbose'])

    (cra, cdec, csnr150, csnr220) = (comp_catalog[:]["ra"],
                                     comp_catalog[:]["dec"],
                                     comp_catalog[:]["SNR150"],
                                     comp_catalog[:]["SNR220"])

    for srcname in augmented_catalog:
        entry = augmented_catalog[srcname]
        (ra, dec) = (entry["ra"], entry["dec"])
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

        orig_flux1 = np.array([orig["S_150d"],
                               orig["S_150d_up"],
                               orig["S_150d_down"]])

        orig_flux2 = np.array([orig["S_220d"],
                               orig["S_220d_up"],
                               orig["S_220d_down"]])

        orig_alpha = np.array([orig["d_alpha"],
                               orig["d_alpha_up"],
                               orig["d_alpha_down"]])

        fp1 = params['flux1name']
        fp2 = params['flux2name']

        print "=" * 80
        # can optionally print "_posterior" and "_posterior_swap" for checking
        print srcname, comp_index, cra[comp_index], ra, \
              cdec[comp_index], dec, minsep, \
              csnr150[comp_index], csnr220[comp_index]

        print "S1" + "-" * 78
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
