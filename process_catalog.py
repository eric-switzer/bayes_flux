"""Process the catalog source-by-source, calling the Bayesian code for each."""
import math
import numpy as np
import source_count_models as dnds
import utilities as utils
import two_band_posterior_flux as tbpf


def process_ptsrc_catalog_alpha(catalog, gp):
    r"""
    gp is the dictionary of global parameters
    gp['use_spt_model'] is an internal test flag to use the same counts model
    as in the SPT source release; distributed version uses source_count_models
    """
    # handle all errors as exceptions
    np.seterr(all='raise', divide='raise',
              over='raise', under='raise',
              invalid='raise')

    # get the catalog parameters
    try:
        fielddtype = catalog.dtype.fields
        fields = fielddtype.keys()
        catalog_size = catalog.size
    except:
        pass

    try:
        fields = catalog.cols.keys()
        catalog_size = len(catalog)
        print "catalog fields: ", fields
    except:
        pass

    if not fields or not catalog_size:
        print "could not determine catalog parameters"
        return 0

    print "Starting process_ptsrc_catalog_alpha on " + repr(catalog_size) + \
          " sources with key field name: " + repr(gp['keyfield_name'])

    print "Band1 named " + repr(gp['flux1name']) + " has frequency " + \
           repr(gp['freq1']) + " and 1-sigma error " + repr(gp['sigma1name'])

    print "Band2 named " + repr(gp['flux2name']) + " has frequency " + \
           repr(gp['freq2']) + " and 1-sigma error " + repr(gp['sigma2name'])

    print "Taking a flat spectral index prior over: " + repr(gp['prior_alpha'])

    print "Percentiles over which the posterior is reported: " + \
           utils.fancy_vector(gp['percentiles'], "%5.3g")

    print "Output/input fluxes assumed to be in Jy. (unless specified as mJy)"

    # calculate theory dN/dS
    # TODO: this is really not very generic now
    # frequencies are hard-coded into particular lookup tables
    # TODO: why let this extend beyond the log-stepped axis (1.)?
    input_s_linear = np.linspace(gp['dnds_minflux'],
                                 gp['dnds_maxflux'],
                                 gp['dnds_numflux'], endpoint=False)
    if gp['use_spt_model']:
        import spt_source_count_models as sptdnds

        dnds_tot_linear_band1 = sptdnds.\
            total_SPTmodel_counts(input_s_linear, gp['freq1'])

        dnds_tot_linear_band2 = sptdnds.\
            total_SPTmodel_counts(input_s_linear, gp['freq2'])
    else:
        dnds_tot_linear_band1 = dnds.dnds_radio(input_s_linear, "143GHz") +\
                                dnds.dnds_ir(input_s_linear, "143GHz")
        #dnds_tot_linear_band1 = dnds.dnds_tucci(input_s_linear, "148GHz") +\
        #                        dnds.dnds_ir(input_s_linear, "143GHz")

        dnds_tot_linear_band2 = dnds.dnds_radio(input_s_linear, "217GHz") +\
                                dnds.dnds_ir(input_s_linear, "217GHz")
        #dnds_tot_linear_band2 = dnds.dnds_tucci(input_s_linear, "220GHz") +\
        #                        dnds.dnds_ir(input_s_linear, "217GHz")

    augmented_catalog = {}
    for srcindex in np.arange(catalog_size):
        print "-" * 80
        flux1 = catalog[srcindex][gp['flux1name']]
        flux2 = catalog[srcindex][gp['flux2name']]

        sigma1 = catalog[srcindex][gp['sigma1name']]
        sigma2 = catalog[srcindex][gp['sigma2name']]

        # convert flux150, sigma150, flux220, sigma220 from mJy to Jy
        if gp['catalog_in_mjy']:
            flux1 /= 1000.
            flux2 /= 1000.
            sigma1 /= 1000.
            sigma2 /= 1000.

        # estimate and remove bias due to maximizing source flux
        # see Vanderlinde et al. cluster paper (there DOF=3)
        # if the SNR^2 is less than the DOF, retain the original value
        if gp['sourcefinder_dof'] is not None:
            snr1 = (flux1 / sigma1) ** 2.
            snr2 = (flux2 / sigma2) ** 2.

            if (snr1 > gp['sourcefinder_dof']) and (flux1 > 0.):
                print "flux1 before positional deboosting: ", flux1 * 1000.
                flux1 = sigma1 * np.sqrt(snr1 - gp['sourcefinder_dof'])
            else:
                print "WARNING: band1 has SNR < DOF, no positional deboosting"

            if (snr2 > gp['sourcefinder_dof']) and (flux2 > 0.):
                print "flux2 before positional deboosting: ", flux2 * 1000.
                flux2 = sigma2 * np.sqrt(snr2 - gp['sourcefinder_dof'])
            else:
                print "WARNING: band2 has SNR < DOF, no positional deboosting"

        # if the catalog has off-diagonal covariance for the flux
        #if gp['sigma12name'] != None:
        #    sigma12 = catalog[srcindex][gp['sigma12name']]
        #else:
        #    sigma12 = 0.

        if gp['sigma12'] != None:
            sigma12 = float(gp['sigma12'])
            print "using sigma12 (Jy): %10.15g" % sigma12

        srcname = repr(catalog[srcindex][gp['keyfield_name']])

        print "Starting analysis on source %s (# %d), " % \
                    (srcname, srcindex) + \
               "flux1 is [mJy]: %5.3g +/- %5.3g, " % \
                    (flux1 * 1000., sigma1 * 1000.) + \
               "flux2 is [mJy]: %5.3g +/- %5.3g, " % \
                    (flux2 * 1000., sigma2 * 1000.)

        posterior = tbpf.two_band_posterior_flux(srcname, flux1, flux2,
                                                 sigma1, sigma2,
                                                 sigma12, input_s_linear,
                                                 dnds_tot_linear_band1,
                                                 dnds_tot_linear_band2,
                                                 gp, swap_flux=False)

        posterior_swap = tbpf.two_band_posterior_flux(srcname, flux1, flux2,
                                                      sigma1, sigma2,
                                                      sigma12, input_s_linear,
                                                      dnds_tot_linear_band1,
                                                      dnds_tot_linear_band2,
                                                      gp, swap_flux=True)

        # copy the source data from the input catalog to the output dict
        source_entry = {}
        for name in fields:
            source_entry[name] = catalog[srcindex][name]

        source_entry["srcindex"] = srcindex

        # find the spectral index of the raw fluxes between bands
        try:
            source_entry["raw_simple_alpha"] = \
                 np.log10(flux1 / flux2) / np.log10(gp['freq1'] / gp['freq2'])
        except FloatingPointError:
            print "source " + srcname + " has no defined raw index (S < 0)"
            source_entry["raw_simple_alpha"] = np.nan

        source_entry["posterior_flux1det"] = posterior
        source_entry["posterior_flux2det"] = posterior_swap
        # assign the posterior flux based on the detection band
        if ((flux1 / sigma1) > (flux2 / sigma2)):
            source_entry["posterior"] = posterior
        else:
            source_entry["posterior"] = posterior_swap

        augmented_catalog[srcname] = source_entry

        prefix = "The percentile points of the posterior "
        print prefix + "band 1 flux are [mJy]: " + \
               utils.pm_error(posterior["flux1"] * 1000., "%5.3g") + \
               " swapped detection: " + \
               utils.pm_error(posterior_swap["flux1"] * 1000., "%5.3g")

        print prefix + "band 2 flux are [mJy]: " + \
               utils.pm_error(posterior["flux2"] * 1000., "%5.3g") + \
               " swapped detection: " + \
               utils.pm_error(posterior_swap["flux2"] * 1000., "%5.3g")

        print prefix + "spectral indices are: " + \
               utils.pm_error(posterior["alpha"], "%5.3g") + \
               " swapped detection: " + \
               utils.pm_error(posterior_swap["alpha"], "%5.3g")

        print prefix + "probability that the index exceeds " + \
               repr(gp['spectral_threshold']) + ": " + \
               repr(posterior["prob_exceed"]) + \
               " swapped detection: " + \
               repr(posterior_swap["prob_exceed"])

    return augmented_catalog
