import numpy as np

a = np.ones(2)

np.savetxt('a.csv',a)

print('Hello hi')

b = np.loadtxt('a.csv')

print(b)