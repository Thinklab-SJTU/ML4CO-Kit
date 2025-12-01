r"""
Base classes for ISCO algorithm.
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


import math
import numpy as np
from typing import Callable, Tuple, List


class LinearTemperatureScheduler(object):
    """
    Temperature scheduler for combinatorial optimization.
    """
    
    def __init__(self, tau0: float, total_iterations: int):
        # Temperature List
        t_list = np.arange(total_iterations)
        tau_list = tau0 - t_list * tau0 / (total_iterations - 1)
        self.tau_list = tau_list

    def get_temperature(self, t: int) -> float:
        """
        Get the temperature at the current iteration.
        """
        return self.tau_list[t]


class EnergyFunction(object):
    """
    Energy function for combinatorial optimization.
    """
    
    def __init__(self):
        pass
    
    def compute(self, x: np.ndarray) -> float:
        """
        Compute the energy of the solution.
        """
        raise NotImplementedError(
            "The ``compute`` method is required to implemented in subclasses."
        )
    
    def compute_delta_vector(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the delta vector of the solution.
        """
        raise NotImplementedError(
            "The ``compute_delta_vector`` method is required to implemented in subclasses."
        )


class PASMHSampler(object):
    """
    Path Auxiliary Sampler with Metropolis-Hastings correction.
    
    This sampler implements the ordered without-replacement proposal strategy:
        1. Compute energy change Δ_i for flipping each variable i
        2. Compute sampling weights w_i = g(exp(-Δ_i)) where g is a function
        3. Sample a path length L ~ Poisson(μ)
        4. Sequentially sample L variables without replacement, weighted by w
        5. Apply flips in the sampled order
        6. Compute proposal probabilities for MH correction
    
    Key features:
        - Ordered sampling: The order matters and affects the proposal
        - Without replacement: Each variable flipped at most once per proposal
        - Adaptive path length: μ adapts based on acceptance rate
    """
    
    def __init__(
        self, 
        rng: np.random.RandomState, 
        mu_init: float = 5.0,
        g_func: Callable[[np.ndarray], np.ndarray] = lambda r: np.sqrt(r),
        adapt_mu: bool = True, 
        target_accept_rate: float = 0.574
    ):
        """
        Initialize PAS-MH sampler.
        
        Args:
            rng: Random number generator
            mu_init: Initial mean path length
            g_func: Weight transformation function g(r) where r = exp(-Δ)
            adapt_mu: Whether to adapt μ based on acceptance rate
            target_accept_rate: Target acceptance rate for adaptation (0.574 is optimal for some problems)
        """
        self.rng = rng
        self.mu = mu_init
        self.g_func = g_func
        self.adapt_mu = adapt_mu
        self.target_accept_rate = target_accept_rate
        
    def propose(
        self, 
        x: np.ndarray, 
        energy_func: EnergyFunction, 
    ) -> Tuple[np.ndarray, float, float]:
        """
        Generate PAS-MH proposal.
        
        Process:
            1. Compute delta vector and weights
            2. Sample path length L
            3. Sample ordered sequence of variables
            4. Apply flips to generate proposal
            5. Compute reverse proposal probability
        
        Args:
            x: Current solution vector
            energy_func: Energy function object
            tau: Current temperature (not used in PAS but included for interface)
            
        Returns:
            y: Proposed solution
            q_forward: Forward proposal probability
            q_reverse: Reverse proposal probability
            info: Dictionary with additional information (L, sequence, etc.)
        """
        n = len(x)
        
        # Step 1: Compute delta vector and sampling weights
        # delta[i] = energy change from flipping x[i]
        delta = energy_func.compute_delta_vector(x)
        
        # Weight each variable by g(exp(-Δ_i))
        # Variables with negative Δ (energy decrease) get higher weight
        ratios = np.exp(-delta)  # exp(-Δ_i)
        weights = self.g_func(ratios) + 1e-14  # add epsilon for numerical stability
        
        # Step 2: Sample path length L ~ Poisson(μ) truncated to [1, n]
        L = self.rng.poisson(self.mu)
        L = max(1, min(L, n))  # truncate to valid range
        
        # Step 3: Sample ordered sequence and compute forward probability
        seq_forward, q_forward = self._sample_ordered_sequence(weights, L)
        
        # Step 4: Apply flips in order to generate proposal
        y = self._apply_flips(x, seq_forward)
        
        # Step 5: Compute reverse proposal probability
        # For reversibility, we need q(x|y) computed from state y
        # The reverse sequence is the same sequence in reverse order
        delta_y = energy_func.compute_delta_vector(y)
        ratios_y = np.exp(-delta_y)
        weights_y = self.g_func(ratios_y) + 1e-14
        
        seq_reverse = seq_forward[::-1]  # reverse the sequence
        q_reverse = self._compute_sequence_probability(weights_y, seq_reverse)
        
        return y, q_forward, q_reverse
    
    def _sample_ordered_sequence(self, weights: np.ndarray, L: int) -> Tuple[List[int], float]:
        """
        Sample an ordered sequence of L variables without replacement.
        
        Sequential sampling:
            At each step t, pick variable j with probability w_j / sum(remaining weights)
        
        Args:
            weights: Sampling weights for each variable
            L: Number of variables to sample
            
        Returns:
            sequence: List of sampled variable indices
            probability: Product of sequential sampling probabilities
        """
        n = len(weights)
        L = min(L, n)
        
        remaining = np.ones(n, dtype=bool)  # track which variables are available
        sequence = []
        probability = 1.0
        
        # Handle edge case: all weights are zero
        if weights.sum() <= 0:
            idxs = self.rng.choice(n, size=L, replace=False)
            prob = 1.0 / math.perm(n, L) if n >= L else 1.0
            return list(idxs), prob
        
        # Sequential sampling
        for _  in range(L):
            # Compute probabilities for remaining variables
            probs = weights * remaining  # zero out already-selected variables
            s = probs.sum()
            
            if s <= 0:
                # All remaining variables have zero weight, sample uniformly
                choices = np.nonzero(remaining)[0]
                choice = self.rng.choice(choices)
                p = 1.0 / len(choices)
            else:
                # Sample proportional to weights
                probs = probs / s
                choice = self.rng.choice(n, p=probs)
                p = probs[choice]
            
            sequence.append(int(choice))
            probability *= float(p)
            remaining[choice] = False
        
        return sequence, probability
    
    def _compute_sequence_probability(
        self, weights: np.ndarray, sequence: List[int]
    ) -> float:
        """
        Compute the probability of generating a specific sequence.
        
        This is used to compute the reverse proposal probability q(x|y).
        
        Args:
            weights: Sampling weights
            sequence: Sequence of variable indices
            
        Returns:
            Probability of generating this sequence
        """
        n = len(weights)
        remaining = np.ones(n, dtype=bool)
        probability = 1.0
        
        for j in sequence:
            probs = weights * remaining
            s = probs.sum()
            
            if s <= 0:
                choices = np.nonzero(remaining)[0]
                p = 1.0 / len(choices)
            else:
                # Probability of selecting j given remaining variables
                p = float((weights[j] * remaining[j]) / (s + 1e-14))
            
            probability *= max(p, 1e-14)  # numerical stability
            remaining[j] = False
        
        return probability
    
    def _apply_flips(self, x: np.ndarray, sequence: List[int]) -> np.ndarray:
        """
        Apply variable flips in order.
        
        Args:
            x: Current solution
            sequence: Sequence of variables to flip
            
        Returns:
            New solution after flipping
        """
        y = x.copy()
        for j in sequence:
            y[j] = 1 - y[j]  # flip binary variable
        return y
    
    def adapt(self, accepted: bool):
        """
        Adapt mean path length μ based on acceptance rate.
        
        The paper suggests adapting μ to maintain a target acceptance rate.
        A simple strategy is to nudge μ towards the target.
        
        Args:
            accepted: Whether the last proposal was accepted
        """
        if not self.adapt_mu:
            return
        
        # Simple adaptation: nudge μ based on acceptance
        # If acceptance rate is too high, increase μ (more aggressive proposals)
        # If acceptance rate is too low, decrease μ (more conservative proposals)
        accept_signal = 1.0 if accepted else 0.0
        delta_mu = 0.001 * (accept_signal - self.target_accept_rate)
        self.mu = float(np.clip(self.mu + delta_mu, 1, 100))


def metropolis_hastings_accept(
    deltaE: float, tau: float, q_forward: float, 
    q_reverse: float, rng: np.random.RandomState
) -> bool:
    """
    Metropolis-Hastings acceptance criterion with proposal correction.
    
    Accept with probability:
        alpha = min(1, exp(-ΔE/τ) * q_forward / q_reverse)
    
    where:
        - ΔE = E(y) - E(x): energy change
        - τ: temperature
        - q_forward = q(y|x): forward proposal probability
        - q_reverse = q(x|y): reverse proposal probability
    
    The ratio q_forward/q_reverse corrects for asymmetric proposals.
    When the proposal is symmetric (q_forward = q_reverse), this reduces
    to standard Metropolis criterion.
    
    Args:
        deltaE: Energy change E(y) - E(x)
        tau: Temperature
        q_forward: Forward proposal probability
        q_reverse: Reverse proposal probability
        rng: Random number generator
        
    Returns:
        True if accepted, False if rejected
    """
    # Handle numerical issues
    if q_reverse == 0:
        # Reverse probability is zero (should be rare)
        # This means we can't go back, so reject for safety
        return False
    
    # Compute log acceptance probability for numerical stability
    # log(α) = -ΔE/τ + log(q_forward) - log(q_reverse)
    log_ratio = -deltaE / max(tau, 1e-12)  # Boltzmann factor
    log_ratio += math.log(q_forward + 1e-14) - math.log(q_reverse + 1e-14)
    
    # Accept if log(α) >= 0, otherwise accept with probability exp(log(α))
    if log_ratio >= 0:
        return True
    else:
        return (math.log(rng.rand()) < log_ratio)