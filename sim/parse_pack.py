"""Parse a captured worker handshake (length-prefixed pickle) from a file and
print the normalized deck region slots."""
import struct
import pickle
import sys
import numpy as np

data = open(sys.argv[1], "rb").read()
if len(data) < 4:
    print(sys.argv[1], ": no handshake captured (worker died before send)")
    sys.exit(1)
n = struct.unpack(">I", data[:4])[0]
pk = pickle.loads(data[4:4 + n])
c = np.rint(np.asarray(pk[0][19:239]) * 7.0)
nz = np.nonzero(c)[0]
print(sys.argv[1], "deck slots:", [(int(i), int(c[i])) for i in nz])
