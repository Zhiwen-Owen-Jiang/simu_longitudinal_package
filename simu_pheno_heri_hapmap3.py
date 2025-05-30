import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from functools import reduce


MASTHEAD = "*********************************************************************\n"
MASTHEAD += "* Data generation for single phenotypes for heritability\n"
MASTHEAD += "*********************************************************************"


"""
v1: hapmap3 SNPs, fixed beta across replicates, fixed causal variants
v2: hapmap3 SNPs, varying beta across replicates, fixed causal variants
v3: hapmap3 SNPs, varying beta across replicates, varying causal variants
v4: causal variants (0.01 <= maf 0.05)

"""


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
    def __init__(
            self, 
            heri, 
            snps_array, 
            population, 
            ld,
            maf,
            skewed=False, 
            alpha=-1, 
            gamma=0,
            use_covar=False
        ):
        """
        heri: a float number between 0 and 1 of heritability
        snps_array (n, m): a np.array of causal variants (centered and normalized)
        population: a pd.DataFrame of FID, IID, and first PC (centered and normalized)
        ld (m, 1): LD score of causal variants
        maf (m, 1): MAF of causal variants
        skewed: if noise distribution is skewed
        alpha: level of MAF dependence
        gamma: level of LD dependence
        use_covar: including covar?

        """
        self.heri = heri
        self.w = 1 - heri - self.heri / 4
        self.snps_array = snps_array
        self.population = population
        self.ld = ld
        self.maf = maf
        self.n_subs, self.n_snps = snps_array.shape
        self.skewed = skewed
        self.alpha = alpha
        self.gamma = gamma
        self.use_covar = use_covar
        # self.logger = logging.getLogger(__name__)

        self.true_beta = self._GetBeta()
        self.Zbeta = np.dot(self.snps_array, self.true_beta)
        self.true_gcov = np.var(self.Zbeta)
        self.Zbeta *= np.sqrt(self.heri / self.true_gcov)

    def _GetBeta(self):
        se = np.sqrt((2 * self.maf * (1 - self.maf)) ** (1 + self.alpha) * (1 / self.ld) ** self.gamma) / 50
        true_beta = np.random.normal(0, 1, self.n_snps).reshape(-1, 1) * se
        return true_beta

    def _GetCovarEffect(self):
        true_effect = np.random.normal(0, 1) * np.sqrt(self.heri / 4)
        population = self.population[2].values.reshape(-1, 1)
        self.population_effect = population * true_effect

        # remove covariate effects
        if not self.use_covar:
            self.population_effect /= 10000
            self.w = 1 - self.heri

    def _GetEpsilon(self):
        if self.w < 0 or self.w > 1:
            raise ValueError('-w should be between 0 and 1')
        epsilon = np.random.normal(0, 1, (self.n_subs, 1)) * np.sqrt(self.w)
        epsilon *= np.sqrt(self.w / np.var(epsilon))
        return epsilon

    def GetSimuData(self):
        error_data = np.zeros((self.n_subs, 50))

        for i in range(50):
            ## covariate effect
            self._GetCovarEffect()
            
            ## common variants effect
            self.true_beta = self._GetBeta()
            self.Zbeta = np.dot(self.snps_array, self.true_beta)
            self.true_gcov = np.var(self.Zbeta)
            self.Zbeta *= np.sqrt(self.heri / self.true_gcov)
            
            X = self.Zbeta + self.population_effect
            epsilon = self._GetEpsilon()
            error_data[:, i] = (X + epsilon).reshape(-1)
            # true_heri = self.heri / np.var(error_data[:, i])

        error_data_df = pd.DataFrame(error_data)
        error_data_df.insert(0, 'IID', self.population['IID'])
        error_data_df.insert(0, 'FID', self.population['FID']) 
                                
        # mean_var_population_effect = np.var(self.population_effect)
        # mean_var_Zbeta = np.var(self.Zbeta)
        # mean_var_epsilon = np.var(epsilon)

        # print(f"The empirical variance of population effect is {mean_var_population_effect}")
        # print(f"The empirical variance of Zbeta is {mean_var_Zbeta}")
        # print(f"The empirical variance of epsilon is {mean_var_epsilon}")
        # print(f"The true heritability is {np.mean(true_heri)}")

        return error_data_df


def main(args):
    # input_dir = f'/work/users/o/w/owenjf/image_genetics/methods/bfiles/wgs_0325/{args.percent}percent'
    input_dir2 = f'/work/users/o/w/owenjf/image_genetics/methods/bfiles/relatedness'
    output_dir = '/work/users/o/w/owenjf/image_genetics/methods/simu_h2/data'

    population = pd.read_csv(os.path.join(input_dir2, f'ukb_imp_chr14_v3_maf_hwe_INFO_QC_white_kinship0.05_0percent_10ksub.fam'), 
                             sep='\t', header=None, usecols=[0, 1, 4])
    population = population.rename({0: 'FID', 1: 'IID'}, axis=1)
    population[2] = (population[4] - np.mean(population[4])) / np.std(population[4])

    snps_array = np.load(os.path.join(input_dir2, f'ukb_imp_chr14_white_kinship0.05_0percent_10ksub_2ksnp_normed.npy'))
    ld_maf = pd.read_csv(os.path.join(input_dir2, f'ukb_imp_chr14_v3_maf_hwe_INFO_QC_white_kinship0.05_0percent_10ksub_ld.score.ld'), sep=' ')
    ld = ld_maf['ldscore'].values.reshape(-1, 1)
    maf = ld_maf['MAF'].values.reshape(-1, 1)

    heri = args.heri
    causal = 0.15
    v = args.v
    c = args.c
    alpha = args.alpha
    gamma = args.gamma
    use_covar = args.use_covar

    if causal < 1 and causal != 0.15:
        n_causal_snps = int(snps_array.shape[1] * causal) + 1
        causal_idxs = np.random.choice(snps_array.shape[1], n_causal_snps, replace=False)
        snps_array = snps_array[:, causal_idxs] # fix causal snps across replicates
        ld = ld[causal_idxs]
        maf = maf[causal_idxs]
    idxs = (maf <= 0.05).reshape(-1)
    snps_array = snps_array[:, idxs]
    ld = ld[idxs]
    maf = maf[idxs]
    print(f"{snps_array.shape[1]} causal SNPs.")

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
        ld=ld,
        maf=maf,
        skewed=args.skewed,
        alpha=alpha,
        gamma=gamma,
        use_covar=use_covar
    )
    
    for i in range(start, end+1):
        error_data = simulator.GetSimuData()
        error_data.to_csv(os.path.join(output_dir, f'single_data_common_snps_kinship0.05_0percent_10ksub_causal{causal}_heri{heri}_{dist}_alpha{alpha}_gamma{gamma}_v{v}_c{i}.txt'), sep='\t', index=None)
        print(os.path.join(output_dir, f'single_data_common_snps_kinship0.05_0percent_10ksub_causal{causal}_heri{heri}_{dist}_alpha{alpha}_gamma{gamma}_v{v}_c{i}.txt'))


parser = argparse.ArgumentParser()
parser.add_argument('--heri', type=float, help='true heritability of each time point')
parser.add_argument('-v', help="version of simulation")
parser.add_argument('-c', help='which replicate it is')
parser.add_argument('--alpha', type=float, help='dependence of genetic effects on MAF')
parser.add_argument('--gamma', type=float, help='dependence of genetic effects on LD score')
parser.add_argument('--percent', help='relatedness percentage')
parser.add_argument('--causal', type=float, help='causal variant percentage')
parser.add_argument('--skewed', action='store_true', help='if the voxel distribution is skewed')
parser.add_argument('--use-covar', action='store_true', help='including covar')


if __name__ == '__main__':
    args = parser.parse_args()
    # logpath = os.path.join(args.out, f"simu_data_{type}_N{str(args.N)}_n{str(args.n)}_m{str(args.m)}_a{str(args.a)}_w{str(args.w)}.log")
    # log = GetLogger(logpath)

    # log.info(MASTHEAD)
    # log.info("Parsed arguments")
    # for arg in vars(args):
    #     log.info(f'--{arg} {getattr(args, arg)}')

    main(args)