"""Process the catalog source-by-source, calling the Bayesian code for each."""
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

    fielddtype = catalog.dtype.fields

    print "Starting process_ptsrc_catalog_alpha on " + repr(catalog.size) + \
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
        dnds_tot_linear_band1 = dnds.dnds_total(input_s_linear, "143GHz")
        dnds_tot_linear_band2 = dnds.dnds_total(input_s_linear, "217GHz")

    augmented_catalog = {}
    for srcindex in np.arange(catalog.size):
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

        # if the catalog has off-diagonal covariance for the flux
        #if gp['sigma12name'] != None:
        #    sigma12 = catalog[srcindex][gp['sigma12name']]
        #else:
        #    sigma12 = 0.

        if gp['sigma12'] != None:
            sigma12 = float(gp['sigma12'])
            print "using sigma12 (Jy): %10.15g" % sigma12

        srcname = repr(catalog[srcindex][gp['keyfield_name']])

        print "-" * 80
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
        for name in fielddtype.keys():
            source_entry[name] = catalog[srcindex][name]

        # find the spectral index of the raw fluxes between bands
        try:
            source_entry["raw_simple_alpha"] = \
                 np.log10(flux1 / flux2) / np.log10(gp['freq1'] / gp['freq2'])
        except FloatingPointError:
            print "source " + srcname + " has no defined raw index (S < 0)"
            source_entry["raw_simple_alpha"] = np.nan

        source_entry[gp['flux1name'] + "_posterior"] = posterior[0]
        source_entry[gp['flux2name'] + "_posterior"] = posterior[1]
        source_entry["alpha_posterior"] = posterior[2]
        source_entry["prob_exceed"] = posterior[3]
        source_entry[gp['flux1name'] + "_posterior_swap"] = posterior_swap[0]
        source_entry[gp['flux2name'] + "_posterior_swap"] = posterior_swap[1]
        source_entry["alpha_posterior_swap"] = posterior_swap[2]
        source_entry["prob_exceed_swap"] = posterior_swap[3]
        # assign the posterior flux based on the detection band
        if ((flux1 / sigma1) > (flux2 / sigma2)):
            source_entry[gp['flux1name'] + "_posterior_det"] = posterior[0]
            source_entry[gp['flux2name'] + "_posterior_det"] = posterior[1]
            source_entry["alpha_posterior_det"] = posterior[2]
            source_entry["prob_exceed_det"] = posterior[3]
        else:
            source_entry[gp['flux1name'] + "_posterior_det"] = \
                                                             posterior_swap[0]
            source_entry[gp['flux2name'] + "_posterior_det"] = \
                                                             posterior_swap[1]
            source_entry["alpha_posterior_det"] = posterior_swap[2]
            source_entry["prob_exceed_det"] = posterior_swap[3]

        augmented_catalog[srcname] = source_entry

        prefix = "The percentile points of the posterior "
        print prefix + "band 1 flux are [mJy]: " + \
               utils.pm_error(posterior[0] * 1000., "%5.3g") + \
               " swapped detection: " + \
               utils.pm_error(posterior_swap[0] * 1000., "%5.3g")

        print prefix + "band 2 flux are [mJy]: " + \
               utils.pm_error(posterior[1] * 1000., "%5.3g") + \
               " swapped detection: " + \
               utils.pm_error(posterior_swap[1] * 1000., "%5.3g")

        print prefix + "spectral indices are: " + \
               utils.pm_error(posterior[2], "%5.3g") + \
               " swapped detection: " + \
               utils.pm_error(posterior_swap[2], "%5.3g")

        print prefix + "probability that the index exceeds " + \
               repr(gp['spectral_threshold']) + ": " + repr(posterior[3]) + \
               " swapped detection: " + repr(posterior_swap[3])

    return augmented_catalog
