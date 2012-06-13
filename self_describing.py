"""
These functions read and write self-describing numpy record arrays.
TODO: extend write function to support string fields
TODO: in unittest generate array internally and check for consistency
TODO: make these a new class with default tokens built in vs. as keywords
"""

import numpy as np
import unittest


def load_selfdescribing_numpy(filename, swaps=None, desctoken="desc:",
                              commenttoken="#", typetoken="@@", verbose=False):
    """Load a self-describing ascii file as a numpy record array.

    Example header and data:
    ---------------------------------------------------------------------------
    # this file has channels name_for_x and name_for_y which are...
    # desc: label@@S20 name_for_x@@float name_for_y@@float name_for_z@@float
    aaaaaaaaaaaaaaaaaaaa 1.3 2.3 3.3
    bbbbbbbbbbbbbbbbbbbb 1.4 2.4 3.4
    ...
    ---------------------------------------------------------------------------
    The header is commented out with # and the code reads until it finds the
    desctoken which indicates the line where the column description can be
    found.

    The data consist of columns of the same length and the description header
    is of the format "column_name@@data_type" where @@ is a reserved separator.

    This format can be any allowed by numpy record arrays, including strings.
    The description is converted into a list of ndtype tuples.

    We may want to rename non-standard column names into standardized ones that
    downstream code will look at -- 'swaps' is a dict of columns to rename,
    e.g. swaps = {"name_for_x":"x", "name_for_y":"y"}

    Use desctoken, commenttoken, or typetoken to change any of these.

    ERS: Apr. 14 '11
    """
    if swaps == None:
        swaps = {}

    inputfile = open(filename, 'r')
    descline = None
    for line in inputfile.xreadlines():
        if (line[0] != commenttoken):
            break

        splitline = line.split()
        if (splitline[1] == desctoken):
            descline = splitline

    inputfile.close()

    if not descline:
        raise NameError(filename + ' had no description line starting with \''
                        + desctoken + '\'')
    desclist = descline[2:]
    if verbose:
        print ("load_selfdescribing_numpy: Description list is "
               + repr(desclist))

    ndtype = []
    for descitem in desclist:
        desctype = descitem.split(typetoken)
        if (desctype[0] in swaps.keys()):
            desctype[0] = swaps[desctype[0]]

        ndtype.append(tuple(desctype))
    if verbose:
        print "load_selfdescribing_numpy: using ndtype " + repr(ndtype)

    output = np.genfromtxt(filename, dtype=ndtype)
    print ("load_selfdescribing_numpy: fields = " + repr(output.dtype.names)
           + ", n_records = " + repr(output.size))

    #print output
    return output


def write_selfdescribing_numpy(filename, data, swaps=None, fmt='%18e',
                               delimiter=' ', newline='\n', comment=None,
                               desctoken="desc:", commenttoken="#",
                               typetoken="@@", verbose=False):
    """Save a record array in a self-describing format.
    Currently uses numpy's savetxt, which does not support string fields.

    ERS: Apr. 17 '11
    """
    if swaps == None:
        swaps = {}

    outputfile = file(filename, 'w')
    # print the comment part of the header
    if comment:
        comment = comment.replace(newline, newline + commenttoken + " ")
        outputfile.write(commenttoken + " " + comment + newline)

    # rename the requested variables and get the type strings
    varnames = data.dtype.names
    # 'fields' is not ordered so,
    vartypes = [data.dtype.fields[x] for x in varnames]
    vartypes = [x[0].str for x in vartypes]
    testswap = lambda x: x in swaps.keys() and swaps[x] or x
    varnames = [testswap(x) for x in varnames]

    # write out the description string
    descstring = commenttoken + " " + desctoken
    for (name, dtype) in zip(varnames, vartypes):
        descstring += " " + name + typetoken + dtype
    if verbose:
        print ("write_selfdescribing_numpy: data description " + descstring)
    descstring += newline
    outputfile.write(descstring)
    outputfile.close()

    # append the data as numpy savetxt ascii
    outputfile = file(filename, 'a')
    np.savetxt(outputfile, data, fmt=fmt, delimiter=delimiter, newline=newline)
    outputfile.close()


class SelfDescribingTest(unittest.TestCase):
    """Unit test class self-describing arrays
    need to be able to write/read test files!
    """
    def test_rw(self):
        """test reading and writing of self-describing arrays"""
        filein = "data/test_self_describing.dat"
        fileout = "data/test_self_describing.out"
        testdata_in = load_selfdescribing_numpy(filein,
                                                swaps={"name_for_x": "x",
                                                       "name_for_y": "y"},
                                                verbose=True)

        write_selfdescribing_numpy(fileout,
                                   testdata_in,
                                   comment="This is a \ntest file",
                                   swaps={"x": "name_for_x",
                                          "y": "name_for_y"},
                                   verbose=True)

        testdata_out = load_selfdescribing_numpy(fileout,
                                         swaps={"name_for_x": "x",
                                         "name_for_y": "y"},
                                         verbose=True)
        print testdata_in.dtype

        # apparently np.array_equal does not know how to compare
        self.assertEqual(testdata_in.dtype, testdata_out.dtype)
        for field in testdata_in.dtype.names:
            self.assertTrue(np.array_equal(testdata_in[field],
                            testdata_out[field]))


# can use "and '--test' in sys.argv" to test
if __name__ == "__main__":
    unittest.main()
