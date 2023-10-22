import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pyro
import torch
from pyro import distributions as pdist

import sbibm
from sbibm.tasks.simulator import Simulator
from sbibm.tasks.task import Task
from sbibm.utils.pyro import make_log_prob_grad_fn


class Ornstein_Uhlenbeck(Task):
    def __init__(self):
        """Ornstein_Uhlenbeck"""

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000011,  # observation 1
        ]

        super().__init__(
            dim_parameters=3,
            dim_data=51,
            name=Path(__file__).parent.name,
            name_display="Ornstein_Uhlenbeck",
            num_observations=1,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        self.prior_params = {
            "low": torch.tensor([0., 0., 0.]),
            "high": torch.tensor([10., 5., 3.]),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.simulator_params = {
            "T": 10,
            "n": 50
        }

    def get_prior(self) -> Callable:
        def prior(num_samples=1):
            return pyro.sample("parameters", self.prior_dist.expand_by([num_samples]))

        return prior

    def get_simulator(self, max_calls: Optional[int] = None) -> Simulator:
        """Get function returning samples from simulator given parameters

        Args:
            max_calls: Maximum number of function calls. Additional calls will
                result in SimulationBudgetExceeded exceptions. Defaults to None
                for infinite budget

        Return:
            Simulator callable
        """        
        def simulator(parameters):
            T = self.simulator_params["T"]
            n = self.simulator_params["n"]
            delta = T/n

            alpha = parameters[0,0]
            beta = parameters[0,1]
            sigma = parameters[0,2]

            x0 = 0
            x = torch.zeros(n+1)
            x[0] = x0
            normal_dist = pdist.Normal(loc=0, scale=1)
            for i in range(1,(n+1)):
                x[i] =  alpha + (x[i-1]-alpha)*np.exp(-beta*delta) + np.sqrt((sigma**2/(2*beta))*(1-np.exp(-2*beta*delta))) * normal_dist.sample()
            
            return(x)

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

   

if __name__ == "__main__":
    task = Ornstein_Uhlenbeck()
    task._setup()
