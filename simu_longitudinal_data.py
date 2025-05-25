import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from functools import reduce


MASTHEAD = "*********************************************************************\n"
MASTHEAD += "* Data generation for longitudinal GWAS with sample relatedness \n"
MASTHEAD += "*********************************************************************"


def GetLogger(logpath):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(logpath, mode='w')
    log.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    log.addHandler(sh)

    return log


def sec_to_str(t):
    '''Convert seconds to days:hours:minutes:seconds'''
    [d, h, m, s, n] = reduce(lambda ll, b : divmod(ll[0], b) + ll[1:], [(t, 1), 60, 60, 24])
    f = ''
    if d > 0:
        f += '{D}d:'.format(D=d)
    if h > 0:
        f += '{H}h:'.format(H=h)
    if m > 0:
        f += '{M}m:'.format(M=m)

    f += '{S}s'.format(S=s)
    return f


class Simulation:
    """
    Simulating phenotypes with relatedness and population stratification
    
    """
    def __init__(self, heri, snps_array, population, a=1.8, w=0.8, skewed=False):
        """
        heri: a float number between 0 and 1 of heritability
        snps_array (n, m): a np.array of causal variants (centered and normalized)
        population: a pd.DataFrame of FID, IID, and first PC (centered and normalized)
        a: decay rate of lambda, greater than 1. Currently it is only polynomial
        w: true signal proportion
        skewed: if noise distribution is skewed
        
        """
        self.heri = heri
        self.snps_array = snps_array
        self.population = population
        self.n_subs, self.n_snps = snps_array.shape
        self.a = a
        self.w = w
        self.skewed = skewed
        # self.logger = logging.getLogger(__name__)

        self._GetBase()
        self._GetLambda()

    def _GetBase(self):
        time_points = np.array([i / 10 for i in range(10)]).reshape(-1, 1)
        self.bases = np.sqrt(2) * np.cos(np.arange(10) * np.pi * time_points)

    def _GetLambda(self):
        self.lam = 2 * (np.arange(10) + 3) ** (-self.a)
        
    def _GetBeta(self):
        se = np.sqrt(np.array(range(4, 10 + 4), dtype=float) ** (-self.a * 1.2)) / 500
        true_b = np.random.normal(0, 1, size=(self.n_snps, 10)) * se
        true_beta = np.dot(true_b, self.bases.T) # (Ng * m) * (m * N)
        return true_b, true_beta
    
    def _GetEta(self, gvar_b):
        # self.theta = gvar_b
        self.theta = self.lam - gvar_b
        self.theta[self.theta < 0] = 0
        if not self.skewed:
            xi_eta = np.random.normal(0, 1, (self.n_subs, 10)) * np.sqrt(self.theta)
        else:
            xi_eta = np.random.normal(0, 1, (self.n_subs, 10)) ** 2 / np.sqrt(2) * np.sqrt(self.theta)
        xi_eta -= np.mean(xi_eta, axis=0)
        self.eta = np.dot(xi_eta, self.bases.T)

    def _GetCovarEffect(self):
        true_effect = np.random.normal(0, 0.5, 10) * np.arange(1, 11).astype(float) ** -2
        true_effect = np.dot(true_effect, self.bases.T).reshape(1, 10)
        population = self.population[2].values.reshape(-1, 1)
        self.population_effect = np.dot(population, true_effect)

    def _GetEpsilon(self, var):
        if self.w < 0 or self.w > 1:
            raise ValueError('-w should be between 0 and 1')
        epi_var = var * (1 - self.w) / self.w
        epsilon = np.random.normal(0, np.sqrt(epi_var), (self.n_subs, 10)) 
        return epsilon

    def _Adjheri(self):
        gvar = np.diagonal(self.true_gcov)
        etavar = np.var(self.eta, axis=0)
        population_effect_var = np.var(self.population_effect, axis=0)
        # cur = gvar / (gvar + etavar + population_effect_var)
        non_gvar = etavar + population_effect_var
        adj_eta = np.sqrt((1 - self.heri) * gvar / (self.heri * non_gvar))
        self.eta *= adj_eta
        self.population_effect *= adj_eta

    @staticmethod
    def _random_sampling(error_data, id):
        long_format_id = list()
        long_format_data = list()
        long_format_time = list()
        
        for i in range(error_data.shape[0]):
            n_time = np.random.choice(list(range(1,11)), 1)
            time_idx = np.random.choice(10, n_time)
            long_format_id.extend([id[i]] ** n_time)
            long_format_data.append(error_data[i, time_idx])
            long_format_time.append(time_idx / 10)
        
        long_format_data = np.concatenate(long_format_data)
        long_format_time = np.concatenate(long_format_time)
        long_format_df = pd.DataFrame({
            'FID': long_format_id, 'IID': long_format_id, 
            'time': long_format_time, 'pheno': long_format_data
        })

        return long_format_df


    def GetSimuData(self):
        ## covariate effect
        self._GetCovarEffect()
        
        ## common variants effect
        self.true_b, self.true_beta = self._GetBeta()
        self.Zb = np.dot(self.snps_array, self.true_b)
        self.true_bgcov = np.cov(self.Zb.T)
        self.Zbeta = np.dot(self.snps_array, self.true_beta)
        self.true_gcov = np.cov(self.Zbeta.T)
        
        ## unexplained effect
        self._GetEta(np.diag(self.true_bgcov))
        self._Adjheri()
    
        X = self.Zbeta + self.eta + self.population_effect
        true_heri = np.diagonal(self.true_gcov) / np.var(X, axis=0)
        sigmaX = np.cov(X.T)
        epsilon = self._GetEpsilon(np.mean(np.diag(sigmaX)))
        error_data = X + epsilon
        
        ## random sampling
        error_data_df = self._random_sampling(error_data, list(self.population['IID']))

        mean_var_population_effect = np.mean(np.var(self.population_effect, axis=0))
        mean_var_Zbeta = np.mean(np.var(self.Zbeta, axis=0))
        mean_var_eta = np.mean(np.var(self.eta, axis=0))
        mean_var_epsilon = np.mean(np.var(epsilon))

        print(f"The empirical variance of population effect is {mean_var_population_effect}")
        print(f"The empirical variance of Zbeta is {mean_var_Zbeta}")
        print(f"The empirical variance of eta is {mean_var_eta}")
        print(f"The empirical variance of epsilon is {mean_var_epsilon}")
        print(f"The true heritability is {np.mean(true_heri)}")
        print(f"The signal-to-noise ratio is {np.mean(np.diag(sigmaX) / np.diag(np.cov(error_data.T)))}")

        return error_data_df


def main(args):
    input_dir = f'/work/users/o/w/owenjf/image_genetics/methods/bfiles/wgs_0325/{args.percent}percent'
    output_dir = '/work/users/o/w/owenjf/image_genetics/methods/simu_longitudinal/data'

    population = pd.read_csv(os.path.join(input_dir, f'ukb_cal_oddchr_cleaned_maf_gene_hwe_white_kinship0.05_{args.percent}percent_10ksub_20pc_merged.eigenvec'), 
                             sep='\t', header=None, usecols=[0, 1, 2])
    population = population.rename({0: 'FID', 1: 'IID'}, axis=1)
    population[2] = (population[2] - np.mean(population[2])) / np.std(population[2])

    snps_array = np.load(os.path.join(input_dir, f'ukb_cal_oddchr_white_kinship0.05_{args.percent}percent_10ksub_10ksnp_merged_normed.npy'))

    heri = args.heri
    causal = args.causal
    a = 1.8
    w = args.w
    v = args.v
    c = args.c

    n_causal_snps = int(snps_array.shape[0] * causal) + 1
    snps_array = snps_array[:, :n_causal_snps] # fix causal snps across replicates
    
    if args.skewed:
        dist = 'skewed'
    else:
        dist = 'normal'

    if ':' in c:
        start, end = [int(x) for x in c.split(':')]
    else:
        start = int(c)
        end = start

    simulator = Simulation(
        heri=heri, 
        snps_array=snps_array, 
        population=population, 
        a=a, 
        w=w, 
        skewed=args.skewed
    )
    
    for i in range(start, end+1):
        phenotype = simulator.GetSimuData()
        phenotype.to_csv(os.path.join(output_dir, f'longitudinal_data_common_snps_kinship0.05_{args.percent}percent_10ksub_causal{args.causal}_heri{heri}_a{a}_w{w}_{dist}_10times_v{v}_c{i}.txt'), sep='\t', index=None)
        print(os.path.join(output_dir, f'longitudinal_data_common_snps_kinship0.05_{args.percent}percent_10ksub_causal{args.causal}_heri{heri}_a{a}_w{w}_{dist}_10times_v{v}_c{i}.txt'))


parser = argparse.ArgumentParser()
parser.add_argument('--heri', type=float, help='true heritability of each time point')
parser.add_argument('-v', help="version of simulation")
parser.add_argument('-c', help='which replicate it is')
parser.add_argument('-w', type=float, help='percentage of true signal')
parser.add_argument('--percent', help='relatedness percentage')
parser.add_argument('--causal', type=float, help='causal variant percentage')
parser.add_argument('--skewed', action='store_true', help='if the voxel distribution is skewed')


if __name__ == '__main__':
    args = parser.parse_args()
    # logpath = os.path.join(args.out, f"simu_data_{type}_N{str(args.N)}_n{str(args.n)}_m{str(args.m)}_a{str(args.a)}_w{str(args.w)}.log")
    # log = GetLogger(logpath)

    # log.info(MASTHEAD)
    # log.info("Parsed arguments")
    # for arg in vars(args):
    #     log.info(f'--{arg} {getattr(args, arg)}')

    main(args)