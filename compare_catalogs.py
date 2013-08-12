"""Load the published SPT point source raw flux catalogs, deboost and compare.
"""
import numpy as np
import self_describing as sd
import utilities as utils
import shelve
# ACT: 149.0 GHz, SPT: 152.
# ACT: 219.6 GHz, SPT: 219.5

def compare_catalogs(params, translate, septol=1e-2):
    """compare a deboosted catalog with one in literature
    plot "comparison.dat" 2:8:1:3:7:9 ..., 5:11:4:6:10:12 with xyerrorbars
    """

    augmented_catalog = shelve.open(params['augmented_catalog'], flag="r")
    comp_catalog = sd.load_selfdescribing_numpy(params['comparison_catalog'],
                                                swaps=translate,
                                                verbose=params['verbose'])

    (cra, cdec, csnr150, csnr220) = (comp_catalog[:]["ra"],
                                     comp_catalog[:]["dec"],
                                     comp_catalog[:]["SNR150"],
                                     comp_catalog[:]["SNR220"])

    outfile = open("comparison/comparison.dat", "w")
    raw_outfile = open("comparison/raw_comparison.dat", "w")

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
            continue

        comp_index = np.where(delta == minsep)[0][0]
        orig = comp_catalog[comp_index]

        # find the range of the original catalog's non-deboosted fluxes
        orig_noise_flux1 = orig["S_150r"] / orig["SNR150"]
        orig_noise_flux2 = orig["S_220r"] / orig["SNR220"]

        orig_rawflux1 = np.array([orig["S_150r"] - orig_noise_flux1,
                                  orig["S_150r"],
                                  orig["S_150r"] + orig_noise_flux1])

        orig_rawflux2 = np.array([orig["S_220r"] - orig_noise_flux2,
                                  orig["S_220r"],
                                  orig["S_220r"] + orig_noise_flux2])

        # find the range of the new catalog's non-deboosted fluxes
        rawflux1 = np.array([entry[params['flux1name']] -
                             entry[params['sigma1name']],
                             entry[params['flux1name']],
                             entry[params['flux1name']] +
                             entry[params['sigma1name']]])

        rawflux2 = np.array([entry[params['flux2name']] -
                             entry[params['sigma2name']],
                             entry[params['flux2name']],
                             entry[params['flux2name']] +
                             entry[params['sigma2name']]])

        fluxarray = []
        fluxarray.extend(rawflux1 * 1000.)
        fluxarray.extend(orig_rawflux1)
        fluxarray.extend(rawflux2 * 1000.)
        fluxarray.extend(orig_rawflux2)
        raw_outfile.write(("%5.3g " * 12 + "\n") % tuple(fluxarray))

        # now find the deboosted flux in the original catalog
        orig_flux1 = np.array([orig["S_150d"],
                               orig["S_150d_up"],
                               orig["S_150d_down"]])

        orig_flux2 = np.array([orig["S_220d"],
                               orig["S_220d_up"],
                               orig["S_220d_down"]])

        orig_alpha = np.array([orig["d_alpha"],
                               orig["d_alpha_up"],
                               orig["d_alpha_down"]])

        reband_factor = (152./149.) ** orig_alpha[0]
        print "scaling for index ", orig_alpha, reband_factor

        frange1 = [orig_flux1[0] - orig_flux1[2], orig_flux1[0],
                   orig_flux1[0] + orig_flux1[1]]

        frange1 /= reband_factor

        frange2 = [orig_flux2[0] - orig_flux2[2], orig_flux2[0],
                   orig_flux2[0] + orig_flux2[1]]

        alrange = [orig_alpha[0] - orig_alpha[2], orig_alpha[0],
                   orig_alpha[0] + orig_alpha[1]]

        fluxarray = []
        fluxarray.extend(entry["posterior"]["flux1"] * 1000.)
        fluxarray.extend(frange1)
        fluxarray.extend(entry["posterior"]["flux2"] * 1000.)
        fluxarray.extend(frange2)
        fluxarray.extend(entry["posterior"]["alpha"])
        fluxarray.extend(alrange)
        outfile.write(("%5.3g " * 18 + "\n") % tuple(fluxarray))

        print "=" * 80
        # can optionally print "_posterior" and "_posterior_swap" for checking
        print srcname, comp_index, cra[comp_index], ra, \
              cdec[comp_index], dec, minsep, \
              csnr150[comp_index], csnr220[comp_index]

        print "S1" + "-" * 78
        print utils.pm_error(entry["posterior"]["flux1"] * 1000., "%5.3g")
        print orig_flux1
        if (np.all(orig_flux1 > 0)):
            print (np.array(utils.pm_vector(entry["posterior"]["flux1"] * \
                    1000.)) - orig_flux1) / orig_flux1 * 100.

        print "S2" + "-" * 60
        print utils.pm_error(entry["posterior"]["flux2"] * 1000., "%5.3g")
        print orig_flux2
        if (np.all(orig_flux2 > 0)):
            print (np.array(utils.pm_vector(entry["posterior"]["flux2"] * \
                    1000.)) - orig_flux2) / orig_flux2 * 100.

        print "ind" + "-" * 69
        print utils.pm_error(entry["posterior"]["alpha"], "%5.3g")
        print orig_alpha
        if (np.all(orig_alpha > 0)):
            print (np.array(utils.pm_vector(entry["posterior"]["alpha"])) -
                  orig_alpha) / orig_alpha * 100.

        print "P(a>t) new: " + repr(entry["posterior"]["prob_exceed"]) + \
              " old: " + repr(orig["palphagt1"])

    augmented_catalog.close()
