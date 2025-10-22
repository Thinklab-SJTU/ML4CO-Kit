r"""
Base Class for Portfolio Optimization Generators.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from enum import Enum
from typing import Union, List
from ml4co_kit.task.portfolio.base import PortfolioTaskBase
from ml4co_kit.generator.base import GeneratorBase, TASK_TYPE


class PO_TYPE(str, Enum):
    """Define the graph types as an enumeration."""
    GBM = "gbm" # Geometric Brownian Motion
    FACTOR = "factor" # Factor Model
    VAR1 = "var1" # VAR(1) Model
    MVT = "mvt" # Multivariate T Model
    GRACH = "grach" # GARCH Model (typo kept for backward-compatibility)
    JUMP = "jump" # Merton Jump-Diffusion
    REGIME = "regime" # Regime-Switching Model


class PortfolioDistributionArgs(object):
    def __init__(
        self,
        # gbm
        gbm_mu_ann: float = 0.08,
        gbm_sigma_ann: float = 0.2,
        # factor
        factor_k: int = 3,
        factor_mu_B: float = 0.0,
        factor_sigma_B: float = 0.5,
        factor_sigma_eps: float = 0.01,
        factor_mu_f: float = 0.02,
        factor_sigma_f: float = 0.02,
        # var1
        var1_mu_A: float = 0.0,
        var1_sigma_A: float = 0.05,
        var1_sr_A: float = 0.7,
        var1_mu_eps: float = 0.01,
        var1_sigma_eps: float = 0.01,
        # mvt
        mvt_v: int = 5,
        mvt_mu: float = 0.08,
        mvt_scale: float = 0.01,
        # grach
        grach_omega: float = 1e-6,
        grach_alpha: float = 0.05,
        grach_beta: float = 0.9,
        grach_z_scale: float = 1.0,
        # jump
        jump_lambda: float = 0.1,
        jump_mu: float = -0.02,
        jump_sigma: float = 0.05,
        # regime
        regime_num: int = 2,
        regime_trans_prob: float = 0.05,
        regime_mu_list: List[float] = [0.02, -0.02],
        regime_sigma_list: List[float] = [0.01, 0.03],
    ):
        # Special Args for GBM
        self.gbm_mu_ann = gbm_mu_ann                 # Annualized mean return
        self.gbm_sigma_ann = gbm_sigma_ann           # Annualized standard deviation

        # Special Args for Factor
        self.factor_k = factor_k                     # Number of factors
        self.factor_mu_B = factor_mu_B               # mean of factor loading matrix B
        self.factor_sigma_B = factor_sigma_B         # Standard deviation of factor loading matrix B
        self.factor_sigma_eps = factor_sigma_eps     # Standard deviation of noise eps (idiosyncratic term)
        self.factor_mu_f = factor_mu_f               # Mean of factors
        self.factor_sigma_f = factor_sigma_f         # Standard deviation of factors

        # Special Args for Var1
        self.var1_mu_A = var1_mu_A                   # Mean of autoregressive coefficient matrix A
        self.var1_sigma_A = var1_sigma_A             # Standard deviation of A
        self.var1_sr_A = var1_sr_A                   # Sparse ratio of A
        self.var1_mu_eps = var1_mu_eps               # Mean of noise eps
        self.var1_sigma_eps = var1_sigma_eps         # Standard deviation of noise eps

        # Special Args for MVT
        self.mvt_v = mvt_v                           # Degrees of freedom
        self.mvt_mu = mvt_mu                         # Mean expected return
        self.mvt_scale = mvt_scale                   # Scale (covariance-like)

        # Special Args for GARCH
        self.grach_omega = grach_omega               # constant term
        self.grach_alpha = grach_alpha               # previous period return squared
        self.grach_beta = grach_beta                 # previous period conditional variance
        self.grach_z_scale = grach_z_scale           # Scale of z ~ N(0, 1)

        # Special Args for Jump
        self.jump_lambda = jump_lambda               # Annual jump intensity (expected jumps per year)
        self.jump_mu = jump_mu                       # Mean jump size
        self.jump_sigma = jump_sigma                 # Jump size volatility

        # Special Args for Regime
        self.regime_num = regime_num                 # Number of regimes
        self.regime_trans_prob = regime_trans_prob   # Transition probability
        self.regime_mu_list = regime_mu_list         # Mean of each regime
        self.regime_sigma_list = regime_sigma_list   # Standard deviation of each regime
        
        # Check if the length of regime_mu_list and regime_sigma_list match regime_num
        if len(self.regime_mu_list) != self.regime_num:
            raise ValueError("Length of regime_mu_list must match regime_num")
        if len(self.regime_sigma_list) != self.regime_num:
            raise ValueError("Length of regime_sigma_list must match regime_num")


class PortfolioGeneratorBase(GeneratorBase):
    """Base class for portfolio optimization generators."""

    def __init__(
        self, 
        task_type: TASK_TYPE, 
        distribution_type: PO_TYPE = PO_TYPE.GBM,
        precision: Union[np.float32, np.float64] = np.float32,
        assets_num: int = 50,
        time_steps: int = 252,
        distribution_args: PortfolioDistributionArgs = PortfolioDistributionArgs(),
    ):
        # Super Initialization
        super(PortfolioGeneratorBase, self).__init__(
            task_type=task_type, 
            distribution_type=distribution_type,
            precision=precision
        )
        
        # Initialize Attributes
        self.assets_num = assets_num                                   # Number of assets
        self.time_steps = time_steps                                   # Number of time steps

        # Special Args for GBM
        self.gbm_mu_ann = distribution_args.gbm_mu_ann                 # Annualized mean return
        self.gbm_sigma_ann = distribution_args.gbm_sigma_ann           # Annualized standard deviation

        # Special Args for Factor
        self.factor_k = distribution_args.factor_k                     # Number of factors
        self.factor_mu_B = distribution_args.factor_mu_B               # mean of factor loading matrix B
        self.factor_sigma_B = distribution_args.factor_sigma_B         # Standard deviation of factor loading matrix B
        self.factor_sigma_eps = distribution_args.factor_sigma_eps     # Standard deviation of noise eps (idiosyncratic term)
        self.factor_mu_f = distribution_args.factor_mu_f               # Mean of factors
        self.factor_sigma_f = distribution_args.factor_sigma_f         # Standard deviation of factors

        # Special Args for Var1
        self.var1_mu_A = distribution_args.var1_mu_A                   # Mean of autoregressive coefficient matrix A
        self.var1_sigma_A = distribution_args.var1_sigma_A             # Standard deviation of A
        self.var1_sr_A = distribution_args.var1_sr_A                   # Sparse ratio of A
        self.var1_mu_eps = distribution_args.var1_mu_eps               # Mean of noise eps
        self.var1_sigma_eps = distribution_args.var1_sigma_eps         # Standard deviation of noise eps

        # Special Args for MVT
        self.mvt_v = distribution_args.mvt_v                           # Degrees of freedom
        self.mvt_mu = distribution_args.mvt_mu                         # Mean expected return
        self.mvt_scale = distribution_args.mvt_scale                   # Scale (covariance-like)

        # Special Args for GARCH
        self.grach_omega = distribution_args.grach_omega               # constant term
        self.grach_alpha = distribution_args.grach_alpha               # previous period return squared
        self.grach_beta = distribution_args.grach_beta                 # previous period conditional variance
        self.grach_z_scale = distribution_args.grach_z_scale           # Scale of z ~ N(0, 1)

        # Special Args for Jump
        self.jump_lambda = distribution_args.jump_lambda               # Annual jump intensity (expected jumps per year)
        self.jump_mu = distribution_args.jump_mu                       # Mean jump size
        self.jump_sigma = distribution_args.jump_sigma                 # Jump size volatility

        # Special Args for Regime
        self.regime_num = distribution_args.regime_num                 # Number of regimes
        self.regime_trans_prob = distribution_args.regime_trans_prob   # Transition probability
        self.regime_mu_list = distribution_args.regime_mu_list         # Mean of each regime
        self.regime_sigma_list = distribution_args.regime_sigma_list   # Standard deviation of each regime
        
        # Check if the length of regime_mu_list and regime_sigma_list match regime_num
        if len(self.regime_mu_list) != self.regime_num:
            raise ValueError("Length of regime_mu_list must match regime_num")
        if len(self.regime_sigma_list) != self.regime_num:
            raise ValueError("Length of regime_sigma_list must match regime_num")

        # Generation Function Dictionary
        self.generate_func_dict = {
            PO_TYPE.GBM: self._generate_gbm,
            PO_TYPE.FACTOR: self._generate_factor,
            PO_TYPE.VAR1: self._generate_var1,
            PO_TYPE.MVT: self._generate_mvt,
            PO_TYPE.GRACH: self._generate_grach,
            PO_TYPE.JUMP: self._generate_jump,
            PO_TYPE.REGIME: self._generate_regime,
        }

    def _mu(self, N: int, mu: float) -> np.ndarray:
        """
        Create mean vector.
        """
        return np.full(shape=(N,), fill_value=mu, dtype=self.precision)

    def _cov(self, N: int, sigma: float) -> np.ndarray:
        """
        covert standard deviation to covariance matrix.
        """
        return (np.eye(N) * (sigma**2)).astype(self.precision)

    def _generate_gbm(self) -> PortfolioTaskBase:
        """
        [Name]: 
            Geometric Brownian Motion (GBM)
        [Introduction]: 
            It assumes that the log-returns follow a multivariate normal 
            distribution with constant volatility.
        [Suitable for]: 
            Approximately stationary volatility and jump-free market scenarios.
        [Formula]: 
            r_t ~ N(mu_step-0.5*sigma_step**2, sigma_step**2)
        """
        # covert annualized parameters to step parameters
        dt = np.asarray(1.0 / self.time_steps).astype(self.precision)
        mu_step = np.asarray(self.gbm_mu_ann).astype(self.precision) * dt
        sigma_step = np.asarray(self.gbm_sigma_ann).astype(self.precision) * np.sqrt(dt)
        
        # Create mean return and volatility vectors
        mu_vec = np.full(shape=(self.assets_num,), fill_value=mu_step)
        sigma_vec = np.full(shape=(self.assets_num,), fill_value=sigma_step)

        # Create covariance matrix
        cov = np.diag(sigma_vec**2)

        # Generate log-returns
        logR = np.random.multivariate_normal(
            mean=mu_vec - 0.5 * sigma_vec**2, cov=cov, size=self.time_steps
        )

        # Calculate log-returns and covariance
        return self._cal_log_returns_and_cov(logR)

    def _generate_factor(self) -> PortfolioTaskBase:
        """
        [Name]: 
            Factor Model
        [Introduction]: 
            Log-returns are modeled as r_t = B f_t + eps_t, where f_t and eps_t 
            are Gaussian noises. A small number of latent factors capture 
            common risk, while residuals represent idiosyncratic risk.
        [Suitable for]: 
            Capturing cross-asset correlations driven by shared economic factors.
        [Formula]: 
            r_t = B f_t + eps_t, where f_t ~ N(mu_f, sigma_f**2), eps_t ~ N(0, sigma_eps**2)
        """
        # Generate factor loading matrix B (N, K)
        B = np.random.normal(
            loc=self.factor_mu_B,
            scale=self.factor_sigma_B, 
            size=(self.assets_num, self.factor_k)
        ).astype(self.precision) # (N, K)
        
        # Factor covariance matrix (T, K)
        mu_F = self._mu(self.factor_k, self.factor_mu_f) # (K,)
        cov_F = self._cov(self.factor_k, self.factor_sigma_f) # (K, K)
        F = np.random.multivariate_normal(mean=mu_F, cov=cov_F, size=self.time_steps) # (T, K)

        # Idiosyncratic covariance matrix (T, N)
        mu_eps = self._mu(self.assets_num, 0.0) # (N,)
        cov_eps = self._cov(self.assets_num, self.factor_sigma_eps) # (N, N)
        eps = np.random.multivariate_normal(mean=mu_eps, cov=cov_eps, size=self.time_steps) # (T, N)
        
        # Compute log-returns: logR = F @ B^T + eps
        logR = F @ B.T + eps # (T, N)
        
        # Calculate log-returns and covariance
        return self._cal_log_returns_and_cov(logR)

    def _generate_var1(self) -> PortfolioTaskBase:
        """
        [Name]: 
            Vector Autoregression (VAR(1))
        [Introduction]: 
            r_t = A r_{t-1} + eps_t. Captures temporal autocorrelation and 
            cross-asset interactions (linear mean-reversion dynamics).
        [Suitable for]: 
            Modeling momentum, mean-reversion, and lead-lag relationships.
        [Formula]: 
            r_t = A r_{t-1} + eps_t, where eps_t ~ N(0, sigma_e**2)
        """
        # Autoregressive coefficient matrix A (N x N)
        A = np.random.normal(
            loc=self.var1_mu_A,
            scale=self.var1_sigma_A, 
            size=(self.assets_num, self.assets_num)
        ).astype(self.precision)
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig >= 1.0:
            A = A / (1.1 * max_eig)
        mask = np.random.rand(self.assets_num, self.assets_num) < self.var1_sr_A
        A[mask] = 0.0
        
        # Noise matrix (T x N)
        mu_eps = self._mu(self.assets_num, self.var1_mu_eps) # (N,)
        cov_eps = self._cov(self.assets_num, self.var1_sigma_eps) # (N, N)
        eps = np.random.multivariate_normal(mean=mu_eps, cov=cov_eps, size=self.time_steps) # (T, N)

        # Simulate VAR(1) process
        logR = np.zeros((self.time_steps, self.assets_num), dtype=self.precision) # (T, N)
        for t in range(self.time_steps):
            logR[t] = A @ logR[t-1] + eps[t] # (N,)
        
        # Calculate log-returns and covariance
        return self._cal_log_returns_and_cov(logR)

    def _generate_mvt(self) -> PortfolioTaskBase:
        """
        [Name]: 
            Multivariate t-distribution
        [Introduction]: 
            Heavy-tailed distribution to model more frequent extreme returns 
            (tail risk), providing a more conservative risk assessment than 
            the normal distribution.
        [Suitable for]: 
            Markets with fat tails and higher kurtosis (crisis scenarios).
        [Formula]: 
            r_t ~ t_v(μ, Σ), where v controls tail heaviness
            r_t = μ + (L @ z) / np.sqrt(u / v)
            where z ~ N(0, I), u ~ Chi2(v)
        """
        # Mean vector (N,)
        mu = self._mu(self.assets_num, self.mvt_mu) # (N,)
        
        # Scale matrix (covariance-like) (N, N)
        cov_scale = self._cov(self.assets_num, self.mvt_scale) # (N, N)
        
        # Cholesky decomposition for sampling (N, N)
        L = np.linalg.cholesky(cov_scale)
        
        # Initialize log-returns array (T, N)
        logR = np.zeros((self.time_steps, self.assets_num), dtype=self.precision)
        
        # Generate multivariate t samples using Normal/Chi-squared construction
        for t in range(self.time_steps):
            z = np.random.normal(size=self.assets_num)
            u = np.random.chisquare(self.mvt_v)
            logR[t] = mu + (L @ z) / np.sqrt(u / self.mvt_v) # (N,)
        
        # Calculate log-returns and covariance
        return self._cal_log_returns_and_cov(logR)

    def _generate_grach(self) -> PortfolioTaskBase:
        """
        [Name]: 
            Generalized Autoregressive Conditional Heteroskedasticity (GARCH)
        [Introduction]: 
            Each asset's volatility evolves over time (conditional heteroskedasticity). 
            Optionally correlated shocks via Gaussian copula. Captures volatility 
            clustering phenomenon observed in real markets.
        [Suitable for]: 
            Modeling time-varying volatility and volatility clustering.
        [Formula]: 
            r_t = sigma_t * z_t, sigma_t^2 = ω + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        """        
        # Initialize conditional variance (unconditional variance if stationary)
        alpha_beta_sum = self.grach_alpha + self.grach_beta
        if alpha_beta_sum < 1:
            sig2 = np.ones(self.assets_num) * self.grach_omega / alpha_beta_sum
        else:
            sig2 = np.ones(self.assets_num) * self.grach_omega / 0.999999
        
        # Initialize log-returns array (T, N)
        logR = np.zeros((self.time_steps, self.assets_num), dtype=self.precision)
        
        # Simulate GARCH process
        for t in range(self.time_steps):
            z = np.random.normal(size=self.assets_num) * self.grach_z_scale
            rt = np.sqrt(sig2) * z
            logR[t] = rt # (N,)
            sig2 = self.grach_omega + self.grach_alpha * (rt**2) + self.grach_beta * sig2
        
        # Calculate log-returns and covariance
        return self._cal_log_returns_and_cov(logR)

    def _generate_jump(self) -> PortfolioTaskBase:
        """
        [Name]: 
            Merton Jump-Diffusion Model
        [Introduction]: 
            Combines normal diffusion with Poisson jumps to capture sudden 
            large fluctuations (jump risk) in asset prices.
        [Suitable for]: 
            Modeling rare but significant market shocks (e.g., crashes, news events).
        [Formula]: 
            dS/S = mu dt + sigma dW + J dN, where N is a Poisson process, J ~ N(mu_J, signa_J^2)
        """
        # covert annualized parameters to step parameters
        dt = np.asarray(1.0 / self.time_steps).astype(self.precision)
        mu_step = np.asarray(self.gbm_mu_ann).astype(self.precision) * dt
        sigma_step = np.asarray(self.gbm_sigma_ann).astype(self.precision) * np.sqrt(dt)
        
        # Jump correction term
        # The Jump process introduces a non-zero expected return due to the Poisson process.
        # Therefore, we need to subtract this correction term from the original mean return.
        jump_correction = self.jump_lambda * (np.exp(self.jump_mu + 0.5 * self.jump_sigma**2) - 1)
        mu_step = mu_step - jump_correction * dt

        # Create mean return and volatility vectors
        mu_vec = np.full(shape=(self.assets_num,), fill_value=mu_step)
        sigma_vec = np.full(shape=(self.assets_num,), fill_value=sigma_step)
        
        # Create covariance matrix for diffusion part
        cov = np.diag(sigma_vec**2)
        
        # Initialize log-returns array
        logR = np.zeros((self.time_steps, self.assets_num), dtype=self.precision)
        
        # Simulate jump-diffusion process
        for t in range(self.time_steps):
            # Diffusion component (continuous Brownian motion)
            diffusion = np.random.multivariate_normal(
                mean=mu_vec - 0.5 * sigma_vec**2, cov=cov
            )
            
            # Jump component (Poisson arrivals with normal jump sizes)
            nj = np.random.poisson(self.jump_lambda * dt, size=self.assets_num)
            jumps = np.array([
                np.sum(np.random.normal(self.jump_mu, self.jump_sigma, n)) if n > 0 else 0.0
                for n in nj
            ], dtype=self.precision)
            
            # Combined return = diffusion + jumps
            logR[t] = diffusion + jumps

        # Calculate log-returns and covariance
        return self._cal_log_returns_and_cov(logR)

    def _generate_regime(self) -> PortfolioTaskBase:
        """
        [Name]: 
            Regime-Switching (Markov-Switching) Model
        [Introduction]: 
            The market switches among multiple regimes (states), each with 
            different return distributions. Can simulate bull/bear cycles, 
            volatility regime changes, and other macro-structural shifts.
        [Suitable for]: 
            Capturing non-stationary market dynamics and regime changes.
        [Formula]: 
            Step1: Generate regime states s_t ~ MarkovChain(P)
            Step2: Generate log-returns r_t ~ N(μ_{s_t}, Σ_{s_t})
        """
        # Transition probability matrix (regimes x regimes)
        P = np.full((self.regime_num, self.regime_num), self.regime_trans_prob, dtype=self.precision)
        np.fill_diagonal(P, 1.0 - self.regime_trans_prob * (self.regime_num - 1))

        # Mu list and Cov list
        mu_list = [self._mu(self.assets_num, mu) for mu in self.regime_mu_list]
        cov_list = [self._cov(self.assets_num, sigma) for sigma in self.regime_sigma_list]

        # Simulate Markov chain (states)
        state = 0
        states = np.zeros(self.time_steps, dtype=int)
        for t in range(self.time_steps):
            probs = P[state]
            state = np.random.choice(np.arange(self.regime_num), p=probs)
            states[t] = state
        
        # Generate log-returns
        logR = np.zeros((self.time_steps, self.assets_num), dtype=self.precision)
        for t in range(self.time_steps):
            state = states[t]
            logR[t] = np.random.multivariate_normal(mean=mu_list[state], cov=cov_list[state]) # (N,)
        
        # Calculate log-returns and covariance
        return self._cal_log_returns_and_cov(logR)

    def _cal_log_returns_and_cov(self, logR: np.ndarray) -> PortfolioTaskBase:
        cov = np.cov(logR, rowvar=False)
        returns = np.mean(logR, axis=0)
        return self._create_task(returns, cov)

    def _create_task(self, returns: np.ndarray, cov: np.ndarray) -> PortfolioTaskBase:
        raise NotImplementedError("Subclasses of PortfolioGeneratorBase must implement this method.")