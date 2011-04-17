import numpy as np
import shutil
# TODO: write write_selfdescribing_numpy which writes arrays out

def load_selfdescribing_numpy(filename, swaps={}, desctoken="desc:", commenttoken="#", typetoken="@@", verbose=False):
  '''Load a self-describing ascii file as a numpy record array.

     Example header and data:
     ---------------------------------------------------------------------------
     # this file has channels fancy_name_for_x and fancy_name_for_y which are...
     # desc: label@@S20 fancy_name_for_x@@float fancy_name_for_y@@float fancy_name_for_z@@float
     aaaaaaaaaaaaaaaaaaaa 1.3 2.3 3.3
     bbbbbbbbbbbbbbbbbbbb 1.4 2.4 3.4
     ...
     ---------------------------------------------------------------------------
     The header is commented out with # and the code reads until it finds the desc:
     token which indicates the line where the column description can be found.

     The data consist of columns of the same length and the description header 
     is of the format "column_name@@data_type" where @@ is a reserved separator.

     This format can be any allowed by numpy record arrays, including strings.
     The description is converted into a list of ndtype tuples. 

     We may want to rename non-standard column names into standardized ones that 
     downstream code will look at -- 'swaps' is a dict of columns to rename, e.g.
     swaps = {"fancy_name_for_x":"x", "fancy_name_for_y":"y"}
  
     Use desctoken, commenttoken, or typetoken to change any of these.

     ERS: Apr. 14 '11
  '''
  f = open(filename,'r')
  descline = None
  for line in f.xreadlines():
    if (line[0] != commenttoken): break
    splitline = line.split()
    if (splitline[1] == desctoken): 
      descline = splitline
  f.close()

  if not descline:
    raise NameError(filename+' had no description line starting with \''+desctoken+'\'')
  desclist = descline[2:]
  if verbose: print "load_selfdescribing_numpy: Description list is "+repr(desclist)

  ndtype = []
  for descitem in desclist:
    desctype = descitem.split(typetoken)
    if (desctype[0] in swaps.keys()): desctype[0] = swaps[desctype[0]]  
    ndtype.append(tuple(desctype))
  if verbose: print "load_selfdescribing_numpy: using ndtype "+repr(ndtype)

  output = np.genfromtxt(filename, dtype=ndtype)
  print "load_selfdescribing_numpy: fields = "+repr(output.dtype.names)+", n_records = "+repr(output.size)

  return output

# TODO: extend this to support string fields
def write_selfdescribing_numpy(filename, data, swaps={}, fmt='%18e', delimiter=' ', newline='\n', comment=None, 
                               desctoken="desc:", commenttoken="#", typetoken="@@", verbose=False):
  '''Save a record array in a self-describing format.
     Currently uses numpy's savetxt, which does not support string fields. 

     ERS: Apr. 17 '11
  '''
  fh = file(filename, 'w')
  # print the comment part of the header
  if comment:
    comment=comment.replace(newline, newline+commenttoken+" ")
    fh.write(commenttoken+" "+comment+newline)
 
  # rename the requested variables and get the type strings
  varnames = data.dtype.names
  vartypes = map(lambda x: data.dtype.fields[x], varnames) # 'fields' is not ordered
  vartypes = map(lambda x: x[0].str, vartypes)
  varnames = map(lambda x: x in swaps.keys() and swaps[x] or x, varnames)

  # write out the description string
  descstring = commenttoken+" "+desctoken
  for (name, dtype) in zip(varnames, vartypes):
    descstring += " "+name+typetoken+dtype
  if verbose: print "write_selfdescribing_numpy: data description "+descstring  
  descstring += newline
  fh.write(descstring)
  fh.close()

  # append the data as numpy savetxt ascii 
  fh = file(filename, 'a')
  np.savetxt(fh, data, fmt=fmt, delimiter=delimiter, newline=newline)
  fh.close()

if __name__ == "__main__":
  data = load_selfdescribing_numpy("testfile.dat", swaps={"fancy_name_for_x":"x", "fancy_name_for_y":"y"}, verbose=True)
  print data
  print data.dtype
  print data['x']
  write_selfdescribing_numpy("testfile.out", data, comment="This is a \ntest file", swaps={"x":"fancy_name_for_x", "y":"fancy_name_for_y"}, verbose=True)
  data = load_selfdescribing_numpy("testfile.out", swaps={"fancy_name_for_x":"x", "fancy_name_for_y":"y"}, verbose=True)
  print data
  print data.dtype
