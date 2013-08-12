"""
Provide several support functions for the deboosting such as
comparison of matrices, data conversions, percentile point calculation and
output formatting.
"""

import os
import numpy as np
from scipy import integrate
from scipy import interpolate
import matplotlib.pyplot as plt
import ConfigParser
import json
import pyfits
import ast


def iniparse(filename, flat=True):
    r"""Note that this calls literal_eval. If this fails, just use the string
    """
    config = ConfigParser.RawConfigParser()
    config.read(filename)

    params = {}
    for section in config.sections():
        if not flat:
            params[section] = {}
        for key, value in config.items(section):
            try:
                eval_val = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                eval_val = value

            if flat:
                params[key] = eval_val
            else:
                params[section][key] = eval_val

    return params


def load_fits_primary(filename, transpose=True):
    """load and transpose a fits file's primary unit using pyfits"""
    output = pyfits.open(filename)
    output = np.array(output[0].data)
    if transpose:
        output = output.transpose()

    return output


def compare_matrices_fits(matrix1, matrix2, filename, fractional=False,
                          replacenan=0.):
    """write out a fits file that makes it easy to compare matrices in ds9:
    ds9 -multiframe filename.fits
    """

    if fractional:
        err = (matrix1 - matrix2) / matrix1
    else:
        err = matrix1 - matrix2

    err[np.isnan(err)] = replacenan
    #np.set_printoptions(threshold=800 * 800 * 2, precision=2)
    #print err
    #pl.imshow(np.log10(np.abs(err)))
    #pl.show()

    # write out the comparison fits
    hduf = pyfits.PrimaryHDU(err)
    hdua = pyfits.ImageHDU(matrix1)
    hdub = pyfits.ImageHDU(matrix2)
    thdulist = pyfits.HDUList([hduf, hdua, hdub])
    try:
        os.remove(filename)
        print "overwriting existing fits file: " + filename
    except OSError:
        print "writing new fits file: " + filename

    thdulist.writeto(filename)

    return err


def compare_matrices(matrix1, matrix2, logplot=False):
    """Make a plot comparing two matrices"""
    fig = plt.figure(1, [6, 3])

    # first subplot
    subplot1 = fig.add_subplot(121)
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins1 = inset_axes(subplot1,
                        width="50%",
                        height="5%",
                        loc=1)

    if (logplot == True):
        image1 = subplot1.imshow(np.log10(matrix1))
    else:
        image1 = subplot1.imshow(matrix1)

    plt.colorbar(image1, cax=axins1, orientation="horizontal",
                 ticks=[1, 2, 3])
    axins1.xaxis.set_ticks_position("bottom")

    # second subplot
    subplot2 = fig.add_subplot(122)

    axins = inset_axes(subplot2,
                       width="5%",
                       height="50%",
                       loc=3,
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=subplot2.transAxes,
                       borderpad=0,
                       )

    # Controlling the placement of the inset axes is basically same as that
    # of the legend.  you may want to play with the borderpad value and
    # the bbox_to_anchor coordinate.

    if (logplot == True):
        image2 = subplot2.imshow(np.log10(matrix2))
    else:
        image2 = subplot2.imshow(matrix2)
    plt.colorbar(image2, cax=axins, ticks=[1, 2, 3])

    plt.draw()
    plt.show()


# TODO check that the input arrays are numpy
# TODO make this a more rigorous or well-behaved CDF; currently cheap
def percentile_points(axis, pdf, percentiles):
    """Coarse way to find percentile points of a distribution given
    as a numpy array
    """
    cdf_points = np.zeros(len(percentiles), dtype=float)

    for index, percentile in enumerate(percentiles):
        try:
            # oritinal
            #cdf = np.cumsum(pdf) / float(np.sum(pdf))
            #cdf = np.cumsum(pdf) / np.sum(pdf)
            #cdf_points[index] = np.nanmin(axis[cdf >= percentile])
            percentile_range = (np.cumsum(pdf) >= (percentile * np.sum(pdf)))
            cdf_points[index] = np.nanmin(axis[percentile_range])
        except FloatingPointError:
            print "percentile_points: invalid PDF"
            cdf = np.zeros_like(pdf)
            cdf_points[index] = -1.

    return cdf_points


# TODO: test this more
def fraction_exceeds(vector, threshold):
    """Find the fraction of vector that exceeds some threshold"""
    return float(len(np.where(vector > threshold)[0])) / float(len(vector))


def prob_exceed(axis, probability, threshold):
    """given x and P(x), find P(x>t)"""
    try:
        exceeding = np.where(axis > threshold)
        integral = integrate.simps(probability[exceeding], axis[exceeding])
        retval = integral / integrate.simps(probability, axis)
    except FloatingPointError:
        print "prob_exceed: invalid PDF"
        retval = -1.

    return retval


# TODO: remove trailing space
def fancy_vector(vector, format_string):
    """print a numpy vector with a format string"""
    output = ""
    for entry in vector:
        output += (format_string + " ") % entry

    return output


def pm_vector(vector):
    """convert a vector of e.g. 16, 50 and 84th percentiles into:
    value (plus) upper error (minus) lower error
    """
    return (vector[1], vector[2] - vector[1], vector[1] - vector[0])


def pm_error(vector, format_string):
    """print a numpy vector with three entries as central value plus errors"""
    return (format_string + " +" + format_string + " -" + format_string) % \
            pm_vector(vector)


def loginterpolate(x_vector, y_vector, xout, replace_nan=np.nan):
    """Do log10-linear interpolation
    (out of range values are _not_ extrapolated)
    The replace_nan flag in the output replaces these (out of bounds)
    with another value
    """
    interpolant = interpolate.interp1d(np.log10(x_vector), np.log10(y_vector),
                                       bounds_error=False,
                                       fill_value=np.nan)
    yout = 10. ** interpolant(np.log10(xout))
    if replace_nan != np.nan:
        yout[np.isnan(yout)] = replace_nan
    return yout


def cov_to_corr(matrix):
    """convert a covariance matrix to a correlation matrix"""
    sqrtdiag = np.sqrt(np.diag(matrix))
    return matrix / np.outer(sqrtdiag, sqrtdiag)


def spline_derivative(x_vector, y_vector):
    """Take a derivative using a spline"""
    tck = interpolate.splrep(x_vector, y_vector, k=3)
    deriv_y = np.array(interpolate.spalde(x_vector, tck))
    deriv_y = deriv_y[:, 1]  # take the first derivative of the spline
    return deriv_y


def store_json(data, filename):
    """save some structure using json"""
    with open(filename, 'w') as fileout:
        fileout.write(json.dumps(data, separators=(', \n', ': ')))


# TODO: rewrite with integrate.simps(s ** moment, dnds); allowing moments
def dnds_to_ngts(s_vector, dnds):
    """Integrate the counts; default is N(>S).
    Uniform (linear) spacing is assumed.
    """
    delta_s = s_vector[1] - s_vector[0]
    revngts = np.cumsum(dnds[::-1]) * delta_s
    return revngts[::-1]


def numpy_recarray_to_dict(data, keyfield_name):
    """convert a numpy structured array to a dictionary
    (easy to write as JSON) eg:

    catalog_dict = utils.numpy_recarray_to_dict(augmented_catalog,
                                                gp['keyfield_name'])
    utils.store_json(catalog_dict, "catalog_human.dat")
    """
    numrec = data.size
    fielddtype = data.dtype.fields
    names = fielddtype.keys()
    formats = []
    output_dict = {}
    for name in names:
        formats.append(fielddtype[name][0])

    for dataindex in np.arange(numrec):
        record_entry = {}
        for name in names:
            # TODO: remove keyfield_name from this list
            record_entry[name] = data[dataindex][name].tolist()
        output_dict[data[dataindex][keyfield_name]] = record_entry
    return output_dict

def print_multicolumn(*args, **kwargs):
    """given a series of arrays of the same size as arguments, write these as
    columns to a file (kwarg "outfile"), each with format string (kwarg format)
    """
    outfile = "multicolumns.dat"
    format = "%10.15g"

    if "outfile" in kwargs:
        outfile = kwargs["outfile"]

    if "format" in kwargs:
        format = kwargs["format"]

    # if there is a root directory that does not yet exist
    rootdir = "/".join(outfile.split("/")[0:-1])
    if len(rootdir) > 0 and rootdir != ".":
        if not os.path.isdir(rootdir):
            print "print_multicolumn: making dir " + rootdir
            os.mkdir(rootdir)

    numarg = len(args)
    fmt_string = (format + " ") * numarg + "\n"
    print "writing %d columns to file %s" % (numarg, outfile)

    outfd = open(outfile, "w")
    for column_data in zip(*args):
        outfd.write(fmt_string % column_data)

    outfd.close()
