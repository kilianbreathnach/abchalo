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

    nscore = (mod - data) ** 2 / 316.22776601683796

    return nscore


def prediction(params):

    model.param_dict['alpha'] = params[0]
    model.param_dict['logM0'] = params[1]
    model.param_dict['logM1'] = params[2]
    model.param_dict['logMmin'] = params[3]
    model.param_dict['sigma_logM'] = params[4]

    try:
        model.populate_mock()
        nbar = model.mock.number_density
    except:
        nbar = 0

    return nbar


zhengnbar = 0.000931584

data = np.array([zhengnbar])

model = Zheng07(threshold = -21)

priormins = np.array([1.0, 10.5, 12.5, 11.5, 0.3])
priormaxs = np.array([1.26, 13.2, 15.5, 14.2, 0.5])

prior = abcpmc.TophatPrior(priormins, priormaxs)

eps = abcpmc.ExponentialEps(20,
                            1e-6,
                            1e-15)

mpi_pool = mpi_util.MpiPool()
sampler = abcpmc.Sampler(N=100, Y=data, postfn=prediction,
                         dist=dist, pool=mpi_pool)
abcpmc.Sampler.particle_proposal_kwargs = {'k':15}
sampler.particle_proposal_cls = abcpmc.KNNParticleProposal

if mpi_pool.isMaster():
    print("Start sampling")
    pools = []


for pool in sampler.sample(prior, eps):
    eps.eps = mpi_util.mpiBCast(pool.eps)
    if mpi_pool.isMaster():
        print("T: {0}, eps: {1:>.4f},\
               ratio: {2:>.4f}".format(pool.t, pool.eps,
                                       pool.ratio))
    for i, (mean, std) in enumerate(zip(np.mean(pool.thetas, axis=0), np.std(pool.thetas, axis=0))):
        if mpi_pool.isMaster():
            print(u"    theta[{0}]: mu = {1:>.4f},   sig = {2:>.4f}".format(i, mean,std))

    if mpi_pool.isMaster():
        pools.append(pool)

if mpi_pool.isMaster():
    thetas = np.vstack([pool.thetas for pool in pools])
    np.savetxt("nonly_thetas.dat", thetas)
    weights = np.vstack([pool.ws for pool in pools])
    np.savetxt("nonly_weights.dat", weights)
    dists = np.vstack([pool.dists for pool in pools])
    np.savetxt("nonly_dists.dat", dists)






