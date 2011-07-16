"""Process the catalog source-by-source, calling the Bayesian code for each"""
import numpy as np
import source_count_models as dnds
import utilities as utils
import bayesian_flux


def process_ptsrc_catalog_alpha(catalog, gp, use_spt_model=False):
    '''
    gp is the dictionary of global parameters
    use_spt_model is an internal test flag to use the same counts model as in
    the SPT source release; distributed version uses source_count_models
    '''
    # handle all errors as exceptions
    np.seterr(all='raise', divide='raise',
              over='raise', under='raise',
              invalid='raise')

    fielddtype = catalog.dtype.fields

    print "Starting process_ptsrc_catalog_alpha on " + repr(catalog.size) + \
          " sources with key field name: " + repr(gp['keyfield_name'])

    #print "process_catalog writing catalog to database file: " + \
    #       gp['catalog_file_out']
    #print "process_catalog writing resampled fluxes \
    #       and distributions to database file: " + \
    #       gp['distributions_file_out']

    print "Band1 named " + repr(gp['flux1name']) + " has frequency " + \
           repr(gp['freq1']) + " and 1-sigma error " + repr(gp['sigma1name'])

    print "Band2 named " + repr(gp['flux2name']) + " has frequency " + \
           repr(gp['freq2']) + " and 1-sigma error " + repr(gp['sigma2name'])

    print "Taking a flat spectral index prior over: " + repr(gp['prior_alpha'])

    print "Percentiles over which the posterior is reported: " + \
           utils.fancy_vector(gp['percentiles'], "%5.3g")

    print "Output/input fluxes assumed to be in Jy. (unless specified as mJy)"

    # find the type for the flux and errors and make fields for the output
    flux1type = fielddtype[gp['flux1name']][0]
    flux2type = fielddtype[gp['flux2name']][0]

    if flux1type != flux2type:
        print "ERROR: input fluxes do not have the same type"
        return None

    # note: this is actually a dictproxy object
    augmented_fielddtype = dict(fielddtype)

    # each output has nposterior points along the CDF of the posterior
    nposterior = repr(len(gp['percentiles']))
    augmented_fielddtype[gp['flux1name'] + "_posterior"] = \
        (nposterior + flux1type.name, 0)

    augmented_fielddtype[gp['flux2name'] + "_posterior"] = \
        (nposterior + flux1type.name, 0)

    augmented_fielddtype["alpha_posterior"] = \
        (nposterior + flux1type.name, 0)

    augmented_fielddtype[gp['flux1name'] + "_posterior_swap"] = \
        (nposterior + flux1type.name, 0)

    augmented_fielddtype[gp['flux2name'] + "_posterior_swap"] = \
        (nposterior + flux1type.name, 0)

    augmented_fielddtype["alpha_posterior_swap"] = \
        (nposterior + flux1type.name, 0)

    # number of drawings and raw simple alpha
    augmented_fielddtype["numalphadist"] = (flux1type.name, 0)
    augmented_fielddtype["raw_simple_alpha"] = (flux1type.name, 0)
    augmented_fielddtype["alpha_probexceed"] = (flux1type.name, 0)

    # make a new numpy array which is augmented to include outputs
    # and copy input data into this
    input_names = fielddtype.keys()
    names = augmented_fielddtype.keys()
    formats = []

    for name in names:
        formats.append(augmented_fielddtype[name][0])

    augmented_catalog = np.zeros(catalog.size,
                                 dtype={'names': names,
                                        'formats': formats})

    for name in input_names:
        augmented_catalog[:][name] = catalog[:][name]

    # calculate theory dN/dS
    # TODO: this is really not very generic now
    # frequencies are hard-coded into particular lookup tables
    # TODO: why let this extend beyond the log-stepped axis (1.)?
    input_s_linear = np.linspace(1.e-8, 1.5, 1e5, endpoint=False)
    if use_spt_model:
        import spt_source_count_models as sptdnds

        dnds_tot_linear_band1 = sptdnds.\
            total_SPTmodel_counts(input_s_linear, gp['freq1'])

        dnds_tot_linear_band2 = sptdnds.\
            total_SPTmodel_counts(input_s_linear, gp['freq2'])
    else:
        dnds_tot_linear_band1 = dnds.dnds_total(input_s_linear, "143GHz")
        dnds_tot_linear_band2 = dnds.dnds_total(input_s_linear, "217GHz")

    for srcindex in np.arange(catalog.size):
        flux1 = augmented_catalog[srcindex][gp['flux1name']]
        flux2 = augmented_catalog[srcindex][gp['flux2name']]
        sigma1 = augmented_catalog[srcindex][gp['sigma1name']]
        sigma2 = augmented_catalog[srcindex][gp['sigma2name']]
        srcname = augmented_catalog[srcindex][gp['keyfield_name']]
        ra = augmented_catalog[srcindex]['ra']
        dec = augmented_catalog[srcindex]['dec']

        print "-" * 80
        print "Starting analysis on source %s (# %d), flux1 is [mJy]: \
               %5.3g +/- %5.3g, flux2 is [mJy]: %5.3g +/- %5.3g ; \
               RA/dec: %6.4g/%6.4g" \
                    % (srcname, srcindex, flux1 * 1000., sigma1 * 1000.,
                       flux2 * 1000., sigma2 * 1000., ra, dec)

        posterior = two_band_posterior_flux(flux1, flux2, sigma1, sigma2,
                                            input_s_linear,
                                            dnds_tot_linear_band1,
                                            dnds_tot_linear_band2, gp,
                                            swap_flux=False)

        posterior_swap = two_band_posterior_flux(flux1, flux2, sigma1, sigma2,
                                                 input_s_linear,
                                                 dnds_tot_linear_band1,
                                                 dnds_tot_linear_band2, gp,
                                                 swap_flux=True)

        # find the spectral index of the raw fluxes between bands
        try:
            augmented_catalog[srcindex]["raw_simple_alpha"] = \
            np.log10(flux1 / flux2) / np.log10(gp['freq1'] / gp['freq2'])

        except FloatingPointError:
            #print "source " + augmented_catalog[srcindex][keyfield_name] + \
            #      " had negative flux (noise)"
            augmented_catalog[srcindex]["raw_simple_alpha"] = np.nan

        augmented_catalog[srcindex][gp['flux1name'] + "_posterior"] = \
                                                          posterior[0]

        augmented_catalog[srcindex][gp['flux2name'] + "_posterior"] = \
                                                          posterior[1]

        augmented_catalog[srcindex]["alpha_posterior"] = posterior[2]

        augmented_catalog[srcindex][gp['flux1name'] + "_posterior_swap"] = \
                                                          posterior_swap[0]

        augmented_catalog[srcindex][gp['flux2name'] + "_posterior_swap"] = \
                                                          posterior_swap[1]

        augmented_catalog[srcindex]["alpha_posterior_swap"] = posterior_swap[2]

        print "The percentile points of the posterior \
               band 1 flux are [mJy]: " + \
               utils.pm_error(posterior[0] * 1000., "%5.3g") + \
               " swapped detection: " + \
               utils.pm_error(posterior_swap[0] * 1000., "%5.3g")

        print "The percentile points of the posterior \
               band 2 flux are [mJy]: " + \
               utils.pm_error(posterior[1] * 1000., "%5.3g") + \
               " swapped detection: " + \
               utils.pm_error(posterior_swap[1] * 1000., "%5.3g")

        print "The percentile points of the spectral are: " + \
               utils.pm_error(posterior[2], "%5.3g") + \
               " swapped detection: " + \
               utils.pm_error(posterior_swap[2], "%5.3g")

    catalog_dict = utils.numpy_recarray_to_dict(augmented_catalog,
                                                gp['keyfield_name'])
    utils.store_json(catalog_dict, "catalog_human.dat")
    return augmented_catalog


def two_band_posterior_flux(flux1, flux2, sigma1, sigma2, s_in, dnds1,
                            dnds2, gp, swap_flux=False):
    '''
    A wrapper to two band posterior flux methods which returns the
    marginalized S_1, S_2 and alpha distributions flux and sigma for each
    band are required inputs.
    inputs:
        --flux and errors (1-sigma) in bands 1 and 2
        --flux vector and source counts vector to assume
        --gp is the global list of parameters to run with
        --swap_flux sets 2 to be the detection band; default is band 1
    '''
    bands = np.array([gp['freq1'], gp['freq2']])

    # make the alpha prior grid
    if gp['verbose']:
        print "two_band_posterior_flux using alpha prior: " + \
             repr(gp['prior_alpha'])

    prior_alpha_total_range = gp['range_alpha']
    alpha_prior_range = gp['prior_alpha']
    alphavec = np.linspace(prior_alpha_total_range[0],
                           prior_alpha_total_range[1],
                           gp['num_alpha'], endpoint=False)
    alpha_prior = np.zeros(gp['num_alpha'])
    # now set the prior to be flat over that range
    alpha_prior[np.logical_and(alphavec >= alpha_prior_range[0],
                               alphavec <= alpha_prior_range[1])] = 1.

    # uncorrelated noise in each band (in Jy)
    cov_noise_jy = np.zeros((2, 2))
    cov_noise_jy[0, 0] = sigma1 ** 2.
    cov_noise_jy[1, 1] = sigma2 ** 2.

    # convert the fractional beam/cal covariance to a flux covariance
    fluxes = np.array([flux1, flux2])

    fluxvec = (np.arange(0, gp['num_flux']) + 0.5) / float(gp['num_flux']) * \
              1.5 * max(fluxes)

    # total covariance in units of Jy^2
    corr_calbeam = utils.cov_to_corr(gp['cov_calbeam'])
    cov_calbeam_jy = (np.outer(fluxes, fluxes)) * gp['cov_calbeam']

    cov_calbeam_jy[1, 0] = np.sqrt(cov_calbeam_jy[0, 0] * \
                           cov_calbeam_jy[1, 1]) * corr_calbeam[1, 0]

    cov_calbeam_jy[0, 1] = cov_calbeam_jy[1, 0]

    if gp['neglect_calerror']:
        print "Important: ignoring calibration error"
        total_covariance = cov_noise_jy
    else:
        total_covariance = cov_noise_jy + cov_calbeam_jy

    if gp['verbose']:
        print "joint code using noise covariance:" + repr(cov_noise_jy)
        print "joint code using cal covariance:" + repr(cov_calbeam_jy)
        print "joint code using covariance:" + repr(total_covariance)

    # TODO: is this the correct ordering? seems swapped
    # TODO: make this more elegant!
    if swap_flux:
        if gp['verbose']:
            print "running band-2 selected source case (swapped)"
        total_covariance_swap = np.zeros((2, 2))
        total_covariance_swap[0, 0] = total_covariance[1, 1]
        total_covariance_swap[1, 1] = total_covariance[0, 0]

        (posterior_fluxindex, posterior_fluxflux) = bayesian_flux.\
                              posterior_twoband_gaussian(flux2, flux1,
                                                         total_covariance_swap,
                                                         bands[::-1], fluxvec,
                                                         alphavec, s_in, dnds2,
                                                         gp['omega_prior'],
                                                         alpha_prior)

        alpha_dist = np.sum(posterior_fluxindex, axis=0)
        #alpha_dist = np.sum(posterior_fluxindex, axis=1)
        flux_pdf2d = posterior_fluxflux
        flux1_dist = np.sum(posterior_fluxflux, axis=0)
        flux2_dist = np.sum(posterior_fluxflux, axis=1)
    else:
        if gp['verbose']:
            print "running band-1 selected source case"

        (posterior_fluxindex, posterior_fluxflux) = bayesian_flux.\
                              posterior_twoband_gaussian(flux1, flux2,
                                                         total_covariance,
                                                         bands, fluxvec,
                                                         alphavec, s_in,
                                                         dnds1,
                                                         gp['omega_prior'],
                                                         alpha_prior)

        alpha_dist = np.sum(posterior_fluxindex, axis=0)
        #alpha_dist = np.sum(posterior_fluxindex, axis=1)
        flux_pdf2d = posterior_fluxflux
        flux1_dist = np.sum(posterior_fluxflux, axis=1)
        flux2_dist = np.sum(posterior_fluxflux, axis=0)

    # calculate the summaries of the various output PDFs
    flux1_percentiles = utils.percentile_points(fluxvec, flux1_dist,
                                                gp['percentiles'])

    flux2_percentiles = utils.percentile_points(fluxvec, flux2_dist,
                                                gp['percentiles'])

    alpha_percentiles = utils.percentile_points(fluxvec, alpha_dist,
                                                gp['percentiles'])

    return (flux1_percentiles, flux2_percentiles, alpha_percentiles)
