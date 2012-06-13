"""Functions that provide dN/dS for radio and IR populations"""
import scipy as sp
import numpy as np
import math
import self_describing as sd


# TODO: write code to translate to other freq. from table's base freq
# TODO: flags to get just flat, steep, etc.
def dnds_radio(input_s, freqtag, replacenan=0.):
    """
    return radio counts as dN/dS per Jy per deg^2
    model: http://web.oapd.inaf.it/rstools/srccnt/srccnt_tables.html
    input_s is flux vector in Jy
    freqtag is the suffix of the counts file to load, e.g. '143GHz'
    """
    table_filename = "counts_data/Radio_model_counts_" + freqtag + ".txt"

    # read log10(S^2.5*dN/dS) against logs
    logdnds_model = sd.load_selfdescribing_numpy(table_filename)

    interpolant = sp.interpolate.interp1d(logdnds_model['logs'],
                                          logdnds_model['dnds_total'],
                                          bounds_error=False,
                                          fill_value=np.nan)
    dnds = 10. ** interpolant(np.log10(input_s)) / input_s ** 2.5
    dnds *= (math.pi / 180.) ** 2.
    dnds[np.isnan(dnds)] = replacenan

    return dnds


# TODO: write code to translate to other freq. from table's base freq;
# TODO: flags to get various components
def dnds_ir(input_s, freqtag, replacenan=0.):
    """
    return IR counts as dN/dS per Jy per deg^2
    http://www.ias.u-psud.fr/irgalaxies/model.php
    input_s is flux vector in Jy
    freqtag is the suffix of the counts file to load, e.g. '143GHz'
    """
    table_filename = "counts_data/Bethermin_model_counts_" + \
                     freqtag + ".txt"

    # read dN/dS(per Jy per sr) against S(Jy)
    logdnds_model = sd.load_selfdescribing_numpy(table_filename)

    interpolant = sp.interpolate.interp1d(np.log10(logdnds_model['flux']),
                                 np.log10(logdnds_model['dnds_median']),
                                 bounds_error=False, fill_value=np.nan)
    dnds = 10. ** interpolant(np.log10(input_s)) * (math.pi / 180.) ** 2.
    dnds[np.isnan(dnds)] = replacenan

    return dnds


# TODO: now we rely on the radio and IR have the same frequency tag
# e.g. 217GHz, but this is not very general
def dnds_total(input_s, freqtag):
    """dN/dS model of IR + radio source contributions"""
    dndsmodel_agn = dnds_radio(input_s, freqtag)
    dndsmodel_ir = dnds_ir(input_s, freqtag)
    return dndsmodel_ir + dndsmodel_agn
