import numpy as np

url_Y_train = "https://drive.google.com/uc?export=download&id=1kvxmWPdYMO7AhdP68KDtdxCTP35y7iRn"
import urllib.request as urllib2
print(' -- Downloading files --')

with urllib2.urlopen(url_Y_train) as testfile, open('train_xBOOM_remote.csv', 'w') as f:
    f.write(testfile.read().decode())

x = np.loadtxt('train_xBOOM_remote.csv')

print(x.shape)