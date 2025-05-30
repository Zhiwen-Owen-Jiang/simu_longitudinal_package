import os
import h5py
import time
import logging
import argparse
import numpy as np
import pandas as pd


MASTHEAD = "*********************************************************************\n"
MASTHEAD += "* Data generation for imaging genetic data analysis with relatedness\n"
MASTHEAD += "*********************************************************************"


"""
v1: hapmap3 SNPs, fixed beta across replicates, fixed causal variants
v2: hapmap3 SNPs, varying beta across replicates, fixed causal variants
v3: hapmap3 SNPs, varying beta across replicates, varying causal variants
v4: causal variants (0.01 <= maf 0.05)

"""


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
            a=1.8, 
            w=0.8, 
            skewed=False, 
            alpha=-1, 
            gamma=0,
            use_covar=False):
        """
        heri: a float number between 0 and 1 of heritability
        snps_array (n, m): a np.array of causal variants (centered and normalized)
        population: a pd.DataFrame of FID, IID, and first PC (centered and normalized)
        a: polynomial decay rate
        w: percentage of smooth signal
        ld (m, 1): LD score of causal variants
        maf (m, 1): MAF of causal variants
        skewed: if noise distribution is skewed
        alpha: level of MAF dependence
        gamma: level of LD dependence
        use_covar: including covar?
        
        """
        self.heri = heri
        self.snps_array = snps_array
        self.population = population
        self.ld = ld
        self.maf = maf
        self.n_subs, self.n_snps = snps_array.shape
        self.skewed = skewed
        self.alpha = alpha
        self.gamma = gamma
        self.a = a
        self.w = w
        self.use_covar = use_covar
        # self.logger = logging.getLogger(__name__)

        self._GetBase()
        self._GetLambda()

    def _GetBase(self):
        self.time_points = np.array([i / 100 for i in range(100)]).reshape(-1, 1)
        self.bases = np.sqrt(2) * np.cos(np.arange(100) * np.pi * self.time_points)

    def _GetLambda(self):
        self.lam = 2 * (np.arange(100) + 3) ** (-self.a)
        
    def _GetBeta(self):
        se = np.sqrt(np.array(range(4, 100 + 4), dtype=float) ** (-self.a * 1.2)) / 100
        true_b = np.random.normal(0, 1, size=(self.n_snps, 100)) * se
        true_b *= np.sqrt((2 * self.maf * (1 - self.maf)) ** (1 + self.alpha) * (1 / self.ld) ** self.gamma)
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
        
        if not self.use_covar:
            self.population_effect /= 10000

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
        # error_data_df = pd.DataFrame(error_data)
        # error_data_df = error_data_df.rename({i: f"voxel{i}" for i in range(100)}, axis=1)
        # error_data_df.insert(0, 'IID', self.population['IID'])
        # error_data_df.insert(0, 'FID', self.population['FID'])

        mean_var_population_effect = np.mean(np.var(self.population_effect, axis=0))
        mean_var_Zbeta = np.mean(np.var(self.Zbeta, axis=0))
        mean_var_eta = np.mean(np.var(self.eta, axis=0))
        mean_var_epsilon = np.mean(np.var(epsilon))

        print(f"The empirical variance of Zbeta is {mean_var_population_effect}")
        print(f"The empirical variance of Zbeta is {mean_var_Zbeta}")
        print(f"The empirical variance of eta is {mean_var_eta}")
        print(f"The empirical variance of epsilon is {mean_var_epsilon}")
        print(f"The true heritability is {np.mean(true_heri)}")
        print(f"The signal-to-noise ratio is {np.mean(np.diag(sigmaX) / np.diag(np.cov(error_data.T)))}")

        return error_data


def main(args):
    input_dir = '/work/users/o/w/owenjf/image_genetics/methods/bfiles/relatedness'
    output_dir = '/work/users/o/w/owenjf/image_genetics/methods/simu_image_h2/data'

    population = pd.read_csv(os.path.join(input_dir, f'ukb_imp_chr14_v3_maf_hwe_INFO_QC_white_kinship0.05_0percent_10ksub.fam'), 
                             sep='\t', header=None, usecols=[0, 1, 4])
    population = population.rename({0: 'FID', 1: 'IID'}, axis=1)
    population[2] = (population[4] - np.mean(population[4])) / np.std(population[4])
    population = population.set_index(['FID', 'IID'])
    ids = population.index.to_list()

    snps_array = np.load(os.path.join(input_dir, f'ukb_imp_chr14_white_kinship0.05_0percent_10ksub_2ksnp_normed.npy'))
    ld_maf = pd.read_csv(os.path.join(input_dir, f'ukb_imp_chr14_v3_maf_hwe_INFO_QC_white_kinship0.05_0percent_10ksub_ld.score.ld'), sep=' ')
    ld = ld_maf['ldscore'].values.reshape(-1, 1)
    maf = ld_maf['MAF'].values.reshape(-1, 1)

    heri = args.heri
    causal = args.causal
    a = 1.8
    w = args.w
    v = args.v
    c = args.c
    alpha = args.alpha
    gamma = args.gamma
    use_covar = args.use_covar

    if causal < 1 and causal != 0.15:
        n_causal_snps = int(snps_array.shape[1] * causal) + 1
        causal_idxs = np.random.choice(snps_array.shape[1], n_causal_snps, replace=False)
        snps_array = snps_array[:, causal_idxs]
        ld = ld[causal_idxs]
        maf = maf[causal_idxs]

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
        phenotype = simulator.GetSimuData()

        with h5py.File(os.path.join(output_dir, f'images_common_snps_kinship0.05_0percent_10ksub_causal{causal}_heri{heri}_a{a}_w{w}_{dist}_alpha{alpha}_gamma{gamma}_100voxels_v{v}_c{i}.h5'), 'w') as file:
            file.create_dataset("images", data=phenotype, dtype="float32")
            file.create_dataset("id", data=ids, dtype="S10")
            file.create_dataset("coord", data=simulator.time_points)

        print(os.path.join(output_dir, f'images_common_snps_kinship0.05_0percent_10ksub_causal{causal}_heri{heri}_a{a}_w{w}_{dist}_alpha{alpha}_gamma{gamma}_100voxels_v{v}_c{i}.h5'))


parser = argparse.ArgumentParser()
parser.add_argument('--heri', type=float, help='true heritability of each time point')
parser.add_argument('-v', help="version of simulation")
parser.add_argument('-c', help='which replicate it is')
parser.add_argument('-w', type=float, help='percentage of true signal')
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