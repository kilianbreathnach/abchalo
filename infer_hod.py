import abcpmc
from abcpmc import mpi_util
import numpy as np
from scipy.stats import chisquare, chi2_contingency, chi2
from multiprocessing import cpu_count
from halotools.empirical_models import Zheng07, model_defaults
from halotools.empirical_models.mock_helpers import (three_dim_pos_bundle,
                                                     infer_mask_from_kwargs)
from halotools.mock_observables.clustering import tpcf
from halotools.sim_manager import supported_sims


def dist(mod, data):

    clust = np.zeros((2, data.shape[0] - 1))

    clust[0, :] = data[1:]
    clust[1, :] = mod[1:]

    diff = clust[1, :] - clust[0, :]
    chisq = np.dot(diff, np.dot(icov, diff))

#    chix, px, dofx, expx = chi2_contingency(clust)

    nscore = (mod[0] - data[0]) ** 2 / 316.22776601683796

    print nscore, chisq

    return np.array([nscore, chisq])


def prediction(params):

    model.param_dict['alpha'] = params[0]
    model.param_dict['logM0'] = params[1]
    model.param_dict['logM1'] = params[2]
    model.param_dict['logMmin'] = params[3]
    model.param_dict['sigma_logM'] = params[4]

    model.populate_mock()

    nbar = model.mock.number_density

    r, clustering = model.mock.compute_galaxy_clustering()

    return np.append(np.array([nbar]), clustering)


zhengnbar = 0.000931584

zhengcorr = np.array([3.46285183e+03, 1.64749125e+03, 7.80527281e+02,
                      3.30492728e+02, 1.38927882e+02, 5.91026616e+01,
                      2.45091664e+01, 1.10830722e+01, 5.76577829e+00,
                      3.14415063e+00, 1.88664838e+00, 1.07786531e+00,
                      5.54622962e-01, 2.87849970e-01])

data = np.append(np.array([zhengnbar]), zhengcorr)
errors = np.array([39, 37, 34, 31, 27, 22, 18, 16, 14, 11, 9, 7.27, 7.23, 7.52])
icov = np.eye(len(errors)) * (1. / errors)

model = Zheng07(threshold = -21)

priormins = np.array([1.0, 10.5, 12.5, 11.5, 0.3])
priormaxs = np.array([1.26, 13.2, 15.5, 14.2, 0.5])

prior = abcpmc.TophatPrior(priormins, priormaxs)

eps = abcpmc.MultiExponentialEps(4,
                                 np.array([1e-6, 1e7]),
                                 np.array([1e-10, 1e4]))

mpi_pool = mpi_util.MpiPool()
sampler = abcpmc.Sampler(N=10, Y=data, postfn=prediction,
                         dist=dist, pool=mpi_pool)
abcpmc.Sampler.particle_proposal_kwargs = {'k':2}
sampler.particle_proposal_cls = abcpmc.KNNParticleProposal

if mpi_pool.isMaster():
    print("Start sampling")
    pools = []


for pool in sampler.sample(prior, eps):
    eps.eps = mpi_util.mpiBCast(pool.eps)
    print("T: {0}, eps_n: {1:>.4f}, eps_x: {1:>.4f},\
           ratio: {2:>.4f}".format(pool.t, pool.eps[0],
                                   pool.eps[1], pool.ratio))
    for i, (mean, std) in enumerate(zip(np.mean(pool.thetas, axis=0), np.std(pool.thetas, axis=0))):
        print(u"    theta[{0}]: mu = {1:>.4f},   sig = {2:>.4f}".format(i, mean,std))

    if mpi_pool.isMaster():
        pools.append(pool)

if mpi_pool.isMaster():
    thetas = np.vstack([pool.thetas for pool in pools])
    np.savetxt("thetas.dat", thetas)
    weights = np.vstack([pool.ws for pool in pools])
    np.savetxt("weights.dat", weights)
    dists = np.vstack([pool.dists for pool in pools])
    np.savetxt("dists.dat", dists)






