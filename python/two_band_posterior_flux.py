r"""Calculate the two-band posterior flux"""
import numpy as np
import utilities as utils
from scipy import interpolate
from numpy import linalg as la
import plot_2d_pdf


def two_band_posterior_flux(srcname, flux1, flux2, sigma1, sigma2, sigma12, s_in, dnds1,
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
    freq_bands = np.array([gp['freq1'], gp['freq2']])

    # make the alpha prior grid
    if gp['verbose']:
        print "two_band_posterior_flux using alpha prior: " + \
             repr(gp['prior_alpha'])

    prior_alpha_total_range = gp['range_alpha']
    alpha_prior_range = gp['prior_alpha']
    alpha_axis = np.linspace(prior_alpha_total_range[0],
                           prior_alpha_total_range[1],
                           gp['num_alpha'], endpoint=False)
    alpha_prior = np.zeros(gp['num_alpha'])
    # now set the prior to be flat over that range
    alpha_prior[np.logical_and(alpha_axis >= alpha_prior_range[0],
                               alpha_axis <= alpha_prior_range[1])] = 1.

    # uncorrelated noise in each band (in Jy)
    cov_noise_jy = np.zeros((2, 2))
    cov_noise_jy[0, 0] = sigma1 ** 2.
    cov_noise_jy[1, 1] = sigma2 ** 2.
    cov_noise_jy[0, 1] = sigma12 ** 2.
    cov_noise_jy[1, 0] = cov_noise_jy[0, 1]

    # convert the fractional beam/cal covariance to a flux covariance
    fluxes = np.array([flux1, flux2])

    minflux = min([flux1 - 6. * sigma1, flux2 - 6. * sigma2])
    # if the N-sigma point is < 0, reset it to be a little above 0
    # so the log-space calculations do not have errors
    if minflux < 0.:
        minflux = flux1/100.

    maxflux = max([flux1 + 6. * sigma1, flux2 + 6. * sigma2])

    #flux_axis = np.linspace(minflux, maxflux, num=gp['num_flux'])
    # alternate
    flux_axis = (np.arange(0, gp['num_flux']) + 0.5)
    flux_axis *= 1. / float(gp['num_flux']) * 1.5 * np.max(fluxes)

    # if given, convert the fractional calibration error into Jy^2
    if (gp['cov_calibration'] != None):
        corr_calibration = utils.cov_to_corr(gp['cov_calibration'])
        cov_calibration_jy = (np.outer(fluxes, fluxes)) * gp['cov_calibration']

        cov_calibration_jy[1, 0] = np.sqrt(cov_calibration_jy[0, 0] * \
                            cov_calibration_jy[1, 1]) * corr_calibration[1, 0]

        cov_calibration_jy[0, 1] = cov_calibration_jy[1, 0]
    else:
        cov_calibration_jy = np.zeros((2, 2))

    if gp['neglect_calerror']:
        print "Important: ignoring calibration error"
        total_covariance = cov_noise_jy
    else:
        total_covariance = cov_noise_jy + cov_calibration_jy

    if gp['verbose']:
        print "joint code using noise covariance:" + repr(cov_noise_jy)
        #print "joint code using cal covariance:" + repr(cov_calibration_jy)
        print "joint code using total covariance:" + repr(total_covariance)

    if swap_flux:
        if gp['verbose']:
            print "running band-2 selected source case (swapped)"
        total_covariance_swap = np.zeros((2, 2))
        total_covariance_swap[0, 0] = total_covariance[1, 1]
        total_covariance_swap[1, 1] = total_covariance[0, 0]

        (posterior_fluxindex, posterior_fluxflux) = \
                              posterior_twoband_gaussian(flux2, flux1,
                                                         total_covariance_swap,
                                                         freq_bands[::-1],
                                                         flux_axis,
                                                         alpha_axis,
                                                         s_in, dnds2,
                                                         gp['omega_prior'],
                                                         alpha_prior)

        alpha_dist = np.sum(posterior_fluxindex, axis=0)
        #flux2_dist = np.sum(posterior_fluxindex, axis=1)
        flux1_dist = np.sum(posterior_fluxflux, axis=0)
        flux2_dist = np.sum(posterior_fluxflux, axis=1)

    else:
        if gp['verbose']:
            print "running band-1 selected source case"

        (posterior_fluxindex, posterior_fluxflux) = \
                              posterior_twoband_gaussian(flux1, flux2,
                                                         total_covariance,
                                                         freq_bands,
                                                         flux_axis,
                                                         alpha_axis,
                                                         s_in, dnds1,
                                                         gp['omega_prior'],
                                                         alpha_prior)

        alpha_dist = np.sum(posterior_fluxindex, axis=0)
        #flux1_dist = np.sum(posterior_fluxindex, axis=1)
        flux1_dist = np.sum(posterior_fluxflux, axis=1)
        flux2_dist = np.sum(posterior_fluxflux, axis=0)

    # plot the 2D posteriors
    if gp['make_2dplot']:
        logscale = True
        if swap_flux:
            filename = "../plots/%s_fluxflux_swap.png" % srcname
        else:
            filename = "../plots/%s_fluxflux.png" % srcname

        plot_2d_pdf.gnuplot_2D(filename, posterior_fluxflux,
                               flux_axis * 1000., flux_axis * 1000.,
                               ["148 GHz Flux (mJy)", "220 GHz Flux (mJy)"],
                                1., srcname, "", logscale=logscale)

    # calculate the summaries of the various output PDFs
    summary = {}
    summary["flux1"] = utils.percentile_points(flux_axis, flux1_dist,
                                                gp['percentiles'])

    summary["flux2"] = utils.percentile_points(flux_axis, flux2_dist,
                                                gp['percentiles'])

    summary["alpha"] = utils.percentile_points(alpha_axis, alpha_dist,
                                                gp['percentiles'])

    summary["prob_exceed"] = utils.prob_exceed(alpha_axis, alpha_dist,
                                   gp['spectral_threshold'])

    # TODO optionally output: flux_axis, alpha_axis, posterior_fluxindex
    # and posterior_fluxflux to get a full record of the posterior space

    return summary


# TODO: numpy.einsum may be able to do some operations faster
def posterior_twoband_gaussian(s_measured1, s_measured2,
                               covmatrix, freq_bands, flux_axis, alpha_axis,
                               s_dnds_model, dnds_model,
                               solid_angle_arcminsq, index_prior,
                               flat_flux_prior=False, debug=False):
    """
    Input:
    * The measured fluxes in band 1 (s_measured1) and band 2 (s_measured2)
    * The covariance of the error model of the measured fluxes (covmatrix)
    * The frequencies of the two band (freq_bands)
    * Vector specifying the axis of flux (flux_axis), linear spacing
    * Vector specifying the spectral index axis (alpha_axis), linear spacing
    * The differential source counts n #/Jy/deg^2 in band 1 (dnds_model)
    * Flux axis of the dN/dS model (s_dnds_model)
    * The resolution element solid angle in arcmin^2 (solid_angle_arcminsq)
    * The prior along the spectral index axis (index_prior)
    * An option to force a flat prior in flux (flat_flux_prior)
    Output:
    P(S_i1, index | S_m1, S_m2) : posterior_fluxindex
    P(S_i1, S_i2 | S_m1, S_m2) : posterior_fluxflux

    Variables can either be in terms of flux1, flux2 (named fluxflux)
    or flux1 and the spectral index (fluxindex)
    """
    # let small numbers in the exp round down to 0
    np.seterr(under='ignore')

    freq_ratio = freq_bands[1] / freq_bands[0]
    (n_flux_axis, n_alpha_axis) = (flux_axis.size, alpha_axis.size)
    covinv = la.inv(covmatrix)

    # find the likelihood P(S_1m, S_2m | S_1i, index)
    # assuming Gaussian noise with covmatrix
    residual_fluxindex = np.zeros((2, n_flux_axis, n_alpha_axis))
    residual_fluxindex[0, :, :] = np.repeat(flux_axis[:, None],
                                            n_alpha_axis, 1)
    # S_2 = S_1 (nu2/nu1)^index
    index_multiplier = (freq_ratio) ** alpha_axis
    residual_fluxindex[1, :, :] = residual_fluxindex[0, :, :] * \
                                    index_multiplier[None, :]
    residual_fluxindex[0, :, :] -= s_measured1
    residual_fluxindex[1, :, :] -= s_measured2

    likelihood_fluxindex = np.exp(-np.sum(residual_fluxindex * \
                             np.tensordot(covinv, residual_fluxindex, (0, 0)),
                             axis=0) / 2.)

    # find the likelihood P(S_1m, S_2m | S_1i, S_2i)
    # assuming Gaussian noise with covmatrix
    residual_fluxflux = np.zeros((2, n_flux_axis, n_flux_axis))
    residual_fluxflux[0, :, :] = np.repeat(flux_axis[:, None],
                                           n_flux_axis, 1)
    residual_fluxflux[1, :, :] = residual_fluxflux[0, :, :].transpose()
    residual_fluxflux[0, :, :] -= s_measured1
    residual_fluxflux[1, :, :] -= s_measured2

    likelihood_fluxflux = np.exp(-np.sum(residual_fluxflux * \
                            np.tensordot(covinv, residual_fluxflux, (0, 0)),
                            axis=0) / 2.)

    # find P(S_1i) and make a matrix version = P(S_1i) repeating along the
    # flux2 or spectral index direction
    pdf_prior_flux = solid_angle_arcminsq / 3600. * dnds_model * \
                     np.exp(-utils.dnds_to_ngts(s_dnds_model, dnds_model) * \
                     solid_angle_arcminsq / 3600.)

    #interpolant = interpolate.splrep(s_dnds_model, pdf_prior_flux)
    interpolant = interpolate.interp1d(s_dnds_model, pdf_prior_flux,
                                       bounds_error=False, fill_value=0.)

    # is this correct, or its transpose?
    #pdf_prior_alpha_int = interpolant(flux_axis * index_multiplier)
    #pdf_prior_flux_int = interpolate.splev(flux_axis, interpolant)
    pdf_prior_flux_int = interpolant(flux_axis)
    fluxprior_fluxindex = np.repeat(pdf_prior_flux_int[:, None],
                                    n_alpha_axis, 1)

    fluxprior_fluxflux = np.repeat(pdf_prior_flux_int[:, None],
                                   n_flux_axis, 1)

    # find the prior
    # one can either specify the prior in flux or spectral index, and we
    # report the posterior in both flux flux and flux index, so need to convert
    # either of these prior conditions into the axes in the posterior
    flux_prior = np.zeros((n_flux_axis, n_flux_axis))
    index_prior_matrix = np.zeros((n_flux_axis, n_alpha_axis))
    ln_freq_ratio = np.log(freq_ratio)
    min_index = np.min(alpha_axis)
    delta_index = alpha_axis[1] - alpha_axis[0]
    if flat_flux_prior:
        flux_prior += 1.
        # transform this to an index prior
        # S_2 = S_1 (nu2/nu1)^index
        index_multiplier = (freq_ratio) ** alpha_axis * np.abs(ln_freq_ratio)
        # is n_alpha_axis x n_flux_axis
        index_prior_matrix = np.repeat(flux_axis[:, None],
                                        n_alpha_axis, 1) * \
                                        index_multiplier[None, :]
    else:
        for i in np.arange(n_flux_axis):
            index_prior_matrix[i, :] = index_prior

            # find the flux prior from the index prior
            # transform index prior into flux prior.
            # first, calculate effective index for each value of band2 flux.
            index_smax2 = np.log(flux_axis / flux_axis[i]) / ln_freq_ratio
            index_indices = np.round((index_smax2 - min_index) / delta_index)

            whneg = (index_indices < 0)
            index_indices[np.where(whneg)] = 0
            whtb = (index_indices >= n_alpha_axis)
            index_indices[np.where(whtb)] = n_alpha_axis - 1

            index_to_flux = index_prior[index_indices.astype(int)]
            flux_prior[i, :] = index_to_flux / (flux_axis * abs(ln_freq_ratio))

            flux_prior[i, np.where(whneg)] = 0.
            flux_prior[i, np.where(whtb)] = 0.

    # P(S_i1, index | S_m1, S_m2) = P(S_m1, S_m2 | S_i1, index) P(S_i1, index)
    # P(S_i1, index) = P(S_i1) P(index)
    posterior_fluxindex = likelihood_fluxindex * \
                          fluxprior_fluxindex * \
                          index_prior_matrix

    # P(S_i1, S_i2 | S_m1, S_m2) = P(S_m1, S_m2 | S_i1, S_i2) P(S_i1, S_i2)
    posterior_fluxflux = likelihood_fluxflux * fluxprior_fluxflux * flux_prior
    #posterior_fluxflux = fluxprior_fluxflux * flux_prior

    np.seterr(under='raise')
    if debug:
        import shelve
        d = shelve.open("posterior.shelve")
        d["likelihood_fluxindex"] = likelihood_fluxindex
        d["fluxprior_fluxindex"] = fluxprior_fluxindex
        d["index_prior_matrix"] = index_prior_matrix
        d["likelihood_fluxflux"] = likelihood_fluxflux
        d["fluxprior_fluxflux"] = fluxprior_fluxflux
        d["flux_prior"] = flux_prior
        d["dnds_in"] = dnds_model
        d["s_in"] = s_dnds_model
        d["ngts_in"] = utils.dnds_to_ngts(s_dnds_model, dnds_model)
        d["psmax"] = pdf_prior_flux
        d["alpha_axis"] = alpha_axis
        d["flux_axis"] = flux_axis
        d["posterior_fluxindex"] = posterior_fluxindex
        d["posterior_fluxflux"] = posterior_fluxflux
        d.close()
        quit()

    return (posterior_fluxindex, posterior_fluxflux)
