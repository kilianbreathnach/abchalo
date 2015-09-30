import abcpmc
from abcpmc import mpi_util
import numpy as np
from multiprocessing import cpu_count
from halotools.empirical_models import Zheng07, model_defaults
from halotools.empirical_models.mock_helpers import (three_dim_pos_bundle,
                                                     infer_mask_from_kwargs)
from halotools.mock_observables.clustering import tpcf
from halotools.sim_manager import supported_sims


clustpar = True
abcpar = False

model = Zheng07(threshold = -20.5)
cat = supported_sims.HaloCatalog()
L = cat.Lbox
rbins = model_defaults.default_rbins
rbin_centers = (rbins[1:] + rbins[0:-1])/2.

zhengnbar = 0.00480192

zhengcorr = np.array([1.40637527e+03, 6.72135032e+02, 3.07970325e+02,
                      1.44010850e+02, 6.88083266e+01, 3.43148073e+01,
                      1.60456345e+01, 7.66339761e+00, 3.91231273e+00,
                      2.15151704e+00, 1.24292105e+00, 7.15933383e-01,
                      3.81631202e-01, 1.91124983e-01])

if clustpar:
    clustN = cpu_count()
else:
    clustN = 1


def dist(dat, mod):

    return np.sum(np.abs((dat - mod) / dat))


def prediction(params):

    model.param_dict['alpha'] = params[0]
    model.param_dict['logM0'] = params[1]
    model.param_dict['logM1'] = params[2]
    model.param_dict['logMmin'] = params[3]
    model.param_dict['sigma_logM'] = params[4]

    model.populate_mock()

    nbar = model.mock.number_density

    mask = infer_mask_from_kwargs(model.mock.galaxy_table)

    pos = three_dim_pos_bundle(table=model.mock.galaxy_table,
                               key1='x', key2='y', key3='z', mask=mask,
                               return_complement=False)

    clustering = tpcf(pos, rbins, period=L, N_threads=clustN)

    return np.array([nbar, clustering[0]])


priormins = np.array([0.1, 8.5, 11.5, 11.0, 0.05])
priormaxs = np.array([5.0, 15.0, 15.0, 13.0, 3.0])

prior = abcpmc.TophatPrior(priormins, priormaxs)

data = np.array([zhengnbar, zhengcorr[0]])

eps = abcpmc.LinearEps(20, 5, 0.075)

if abcpar:

    mpi_pool = mpi_util.MpiPool()
    sampler = abcpmc.Sampler(N=10, Y=data, postfn=postfn, dist=dist, pool=mpi_pool)

    if mpi_pool.isMaster(): print("Start sampling")

else:
    sampler = abcpmc.Sampler(N=10, Y=data, postfn=prediction, dist=dist)

for pool in sampler.sample(prior, eps):
    print("T: {0}, eps: {1:>.4f}, ratio: {2:>.4f}".format(pool.t, pool.eps, pool.ratio))
    for i, (mean, std) in enumerate(zip(np.mean(pool.thetas, axis=0), np.std(pool.thetas, axis=0))):
        print(u"    theta[{0}]: {1:>.4f} \u00B1 {2:>.4f}".format(i, mean,std))
