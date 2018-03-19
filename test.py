import numpy as np

url_Y_train = "https://doc-0o-84-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/9cut65npeuvp9evcfk0a0nq1al25hnan/1521396000000/10970379748800439747/*/1PuENkRYGxw3bJ-m0HQOLT24tFqOPyCf1?e=download"

import urllib.request as urllib2
print(' -- Downloading files --')

with urllib2.urlopen(url_Y_train) as testfile, open('train_yBOOM_remote.csv', 'w') as f:
    f.write(testfile.read().decode())

y = np.loadtxt('train_yBOOM_remote.csv')

print(y.shape)