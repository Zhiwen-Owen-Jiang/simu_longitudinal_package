import os
import time
import logging
import argparse
import numpy as np
import pandas as pd
from utils import GetLogger, sec_to_str

MASTHEAD = "*********************************************************************\n"
MASTHEAD += "* Data generation for imaging genetic data analysis with relatedness\n"
MASTHEAD += "*********************************************************************"


class Simulation:
    """
    Simulating phenotypes with relatedness and population stratification
    
    """
    def __init__(self, heri, snps_array, population, a=1.8, w=0.8):
        """
        heri: a float number between 0 and 1 of heritability
        snps_array (n, m): a np.array of causal variants (centered and normalized)
        population: a pd.DataFrame of FID, IID, and first PC (centered and normalized)
        a: decay rate of lambda, greater than 1. Currently it is only polynomial
        w: true signal proportion
        
        """
        self.heri = heri
        self.snps_array = snps_array
        self.population = population
        self.n_subs, self.n_snps = snps_array.shape
        self.a = a
        self.w = w
        self.logger = logging.getLogger(__name__)

        self._GetBase()
        self._GetLambda()

    def _GetBase(self):
        time_points = np.array([i / 100 for i in range(100)]).reshape(-1, 1)
        self.bases = np.sqrt(2) * np.cos(np.arange(100) * np.pi * time_points)

    def _GetLambda(self):
        self.lam = 2 * (np.arange(100) + 3) ** (-self.a)
        
    def _GetBeta(self):
        se = np.sqrt(np.array(range(4, 100 + 4), dtype=float) ** (-self.a * 1.2)) / 100
        true_b = np.random.normal(0, 1, size=(self.n_snps, 100)) * se
        true_beta = np.dot(true_b, self.bases.T) # (Ng * m) * (m * N)
        return true_b, true_beta
    
    def _GetEta(self, gvar_b):
        # self.theta = gvar_b
        self.theta = self.lam - gvar_b
        self.theta[self.theta < 0] = 0
        xi_eta = np.random.normal(0, 1, (self.n_subs, 100)) * np.sqrt(self.theta)
        xi_eta -= np.mean(xi_eta, axis=0)
        self.eta = np.dot(xi_eta, self.bases.T)

    def _GetCovarEffect(self):
        true_effect = np.random.normal(0, 0.5, 100) * np.arange(1, 101).astype(float) ** -2
        true_effect = np.dot(true_effect, self.bases.T).reshape(1, 100)
        population = self.population[2].values.reshape(-1, 1)
        self.population_effect = np.dot(population, true_effect)

    def _GetEpsilon(self, var):
        if self.w < 0 or self.w > 1:
            raise ValueError('-w should be between 0 and 1')
        epi_var = var * (1 - self.w) / self.w
        epsilon = np.random.normal(0, np.sqrt(epi_var), (self.n_subs, 100)) 
        return epsilon

    def _Adjheri(self):
        gvar = np.diagonal(self.true_gcov)
        etavar = np.var(self.eta, axis=0)
        population_effect_var = np.var(self.population_effect, axis=0)
        cur = gvar / (gvar + etavar + population_effect_var)
        adj_eta = np.sqrt((1 - self.heri) / (1 - cur))
        adj_gcov = np.sqrt(self.heri / cur).reshape(-1, 1)
        self.eta *= adj_eta
        self.population_effect *= adj_eta
        self.true_gcov *= np.outer(adj_gcov, adj_gcov)
        self.Zbeta *= np.sqrt(self.heri / cur)

    def GetSimuData(self):
        self._GetCovarEffect()
        self.true_b, self.true_beta = self._GetBeta()
        self.Zb = np.dot(self.snps_array, self.true_b)
        self.true_bgcov = np.cov(self.Zb.T)
        self.Zbeta = np.dot(self.snps_array, self.true_beta)
        self.true_gcov = np.cov(self.Zbeta.T)
        self._GetEta(np.diag(self.true_bgcov))
        self._Adjheri()
    
        X = self.Zbeta + self.eta + self.population_effect
        true_heri = np.diagonal(self.true_gcov) / np.var(X, axis=0)
        sigmaX = np.cov(X.T)
        epsilon = self._GetEpsilon(np.mean(np.diag(sigmaX)))
        error_data = X + epsilon
        error_data_df = pd.DataFrame(error_data)
        error_data_df = error_data_df.rename({i: f"voxel{i}" for i in range(100)}, axis=1)
        error_data_df.insert(0, 'IID', self.population['IID'])
        error_data_df.insert(0, 'FID', self.population['FID'])

        mean_var_population_effect = np.mean(np.var(self.population_effect, axis=0))
        mean_var_Zbeta = np.mean(np.var(self.Zbeta, axis=0))
        mean_var_eta = np.mean(np.var(self.eta, axis=0))
        mean_var_epsilon = np.mean(np.var(epsilon))

        self.logger.info(f"The empirical variance of Zbeta is {mean_var_population_effect}")
        self.logger.info(f"The empirical variance of Zbeta is {mean_var_Zbeta}")
        self.logger.info(f"The empirical variance of eta is {mean_var_eta}")
        self.logger.info(f"The empirical variance of epsilon is {mean_var_epsilon}")
        self.logger.info(f"The true heritability is {np.mean(true_heri)}")
        self.logger.info(f"The signal-to-noise ratio is {np.mean(np.diag(sigmaX) / np.diag(np.cov(error_data.T)))}")

        return error_data_df


def main(args):
    input_dir = '/work/users/o/w/owenjf/image_genetics/methods/bfiles/relatedness'
    output_dir = '/work/users/o/w/owenjf/image_genetics/methods/simu_rel/data'

    population = pd.read_csv(os.path.join(input_dir, f'ukb_cal_oddchr_cleaned_maf_gene_hwe_white_kinship0.05_{args.percent}percent_10ksub_20pc.eigenvec'), 
                             sep=' ', header=None, usecols=[0, 1, 2])
    population = population.rename({0: 'FID', 1: 'IID'}, axis=1)
    population[2] = (population[2] - np.mean(population[2])) / np.std(population[2])

    snps_array = np.load(os.path.join(input_dir, f'ukb_cal_oddchr_white_kinship0.05_{args.percent}percent_10ksub_40ksnp.npy'))
    snps_array = (snps_array - np.mean(snps_array, axis=0)) / np.std(snps_array, axis=0)

    heri = args.heri
    a = 1.8
    w = args.w
    v = args.v
    c = args.c
    
    if ':' in c:
        start, end = [int(x) for x in c.split(':')]
    else:
        start = int(c)
        end = start

    simulator = Simulation(heri=heri, snps_array=snps_array, population=population, a=a, w=w)

    for i in range(start, end+1):
        phenotype = simulator.GetSimuData()
        phenotype.to_csv(os.path.join(output_dir, f'images_kinship0.05_{args.percent}percent_10ksub_40ksnps_heri{heri}_a{a}_w{w}_100voxels_v{v}_c{i}.txt'), sep='\t', index=None)
        print(os.path.join(output_dir, f'images_kinship0.05_{args.percent}percent_10ksub_40ksnps_heri{heri}_a{a}_w{w}_100voxels_v{v}_c{i}.txt'))


parser = argparse.ArgumentParser()
parser.add_argument('--heri', type=float, default=0.3, help='true heritability of each point')
parser.add_argument('-v', help="version of simulation")
# parser.add_argument('-o', '--out', help='directory for output')
parser.add_argument('-c', help='which replicate it is can be a range, e.g. 1:10')
parser.add_argument('-w', type=float, help='signal to noise ratio')
parser.add_argument('--percent', help='relatedness percentage')


if __name__ == '__main__':
    args = parser.parse_args()
    # logpath = os.path.join(args.out, f"simu_data_{type}_N{str(args.N)}_n{str(args.n)}_m{str(args.m)}_a{str(args.a)}_w{str(args.w)}.log")
    # log = GetLogger(logpath)

    # log.info(MASTHEAD)
    # log.info("Parsed arguments")
    # for arg in vars(args):
    #     log.info(f'--{arg} {getattr(args, arg)}')

    main(args)