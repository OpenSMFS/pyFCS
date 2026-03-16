import numpy as np
import pyFCS as fcs

data = [np.cumsum(np.random.poisson(100, size=np.random.poisson(200))) for _ in range(2)]
bins = np.logspace(0,3,6).astype(np.int64)
nanos = [np.random.randint(0,10,size=d.size) for d in data]
weightsI = np.array([np.sin(np.arange(10)/10*np.pi), np.sin(np.arange(10)/5*np.pi)])
weightsL = [weightsI[:,n] for n in nanos]
# rcor = np.arange(0,bins.size-1,1).astype(float)

a = fcs.correlate(data, data, bins, max_cores=1)
# a = fcs.correlate(data, data, bins, weightsL, weightsL)
# b = fcs.correlate(data[0], data[0], bins, weightsL[0], weightsL[0])
print(bins)
print(fcs.__doc__)
# b = fcs.normalization_factor(data, data, bins)
dcorrl = np.exp(np.arange(bins.size, -1, 1))
# c = fcs.normalize(dcorrl, data, data, bins)
# b = fcs.normalization_factor(data, data, bins[:-1])
# a = fcs.normalization_factor(data, data, bins)
# del a, data, bins, nanos, weightsI, weightsL
# a = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)
# b = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)
# a = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)
# b = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)

# b = fcs.correlate(data, data, bins, weightsI[0,:], weightsI[1,:],nanos, nanos)
# c = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)

# b = fcs.correlate(data, data, bins, weightsI[0,:], weightsI[1,:],nanos, nanos)
# c = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)

# b = fcs.correlate(data, data, bins, weightsI[0,:], weightsI[1,:],nanos, nanos)
# c = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)
# b = fcs.correlate(data, data, bins, weightsI[0,:], weightsI[1,:],nanos, nanos)
# c = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)
# b = fcs.correlate(data, data, bins, weightsI[0,:], weightsI[1,:],nanos, nanos)
# c = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)
# b = fcs.correlate(data, data, bins, weightsI[0,:], weightsI[1,:],nanos, nanos)
# c = fcs.correlate(data, data, bins, weightsI, weightsI,nanos, nanos)
# print(data, bins, nanos, weightsI, weightsL)
# del data, bins #, nanos, weightsI, weightsL
