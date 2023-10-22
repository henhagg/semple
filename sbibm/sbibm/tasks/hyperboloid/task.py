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


class Hyperboloid(Task):
    def __init__(self):
        """Hyperboloid"""

        # Observation seeds to use when generating ground truth
        observation_seeds = [
            1000011,  # observation 1
        ]

        super().__init__(
            dim_parameters=2,
            dim_data=10,
            name=Path(__file__).parent.name,
            name_display="Hyperboloid",
            num_observations=1,
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[100, 1000, 10000, 100000, 1000000],
            observation_seeds=observation_seeds,
            path=Path(__file__).parent.absolute(),
        )

        prior_bound = 2.0
        self.prior_params = {
            "low": -prior_bound * torch.ones((self.dim_parameters,)),
            "high": +prior_bound * torch.ones((self.dim_parameters,)),
        }
        self.prior_dist = pdist.Uniform(**self.prior_params).to_event(1)
        self.prior_dist.set_default_validate_args(False)

        self.simulator_params = {
            "micr1": (-0.5,0),
            "micr2": (0.5,0),
            "micr1p": (0,-0.5),
            "micr2p": (0,0.5),
            "sigmaT": 0.01,
            "dofT": 3.0,
            "ny": 10
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
        def simuITDT(x01,m1,m2,sigmaT,dofT,ny):
            micr11=m1[0]
            micr12=m1[1]
            micr21=m2[0]
            micr22=m2[1]
            x0=x01[0,0]
            x1=x01[0,1]

            d1=torch.sqrt((x0-micr11)**2 + (x1-micr12)**2)
            d2=torch.sqrt((x0-micr21)**2 + (x1-micr22)**2)
            
            dofT = torch.tensor(dofT)
            Mutest = torch.ones(ny)*abs(d1-d2)

            Sigmatest = torch.zeros(ny,ny)
            Sigmatest.fill_diagonal_(fill_value=sigmaT)

            t_dist = pdist.MultivariateStudentT(df=dofT, loc=Mutest, scale_tril=Sigmatest)
            return t_dist.sample()
        
        def simulator(parameters):
            num_samples = parameters.shape[0]
            u_dist = pdist.Uniform(low=-1,high=1)
            u = u_dist.sample()

            if(u>0.5):
                ysimu = simuITDT(parameters, self.simulator_params["micr1"], self.simulator_params["micr2"],
                self.simulator_params["sigmaT"], self.simulator_params["dofT"], self.simulator_params["ny"])
            else:
                ysimu = simuITDT(parameters, self.simulator_params["micr1p"], self.simulator_params["micr2p"],
                self.simulator_params["sigmaT"], self.simulator_params["dofT"], self.simulator_params["ny"])
            return ysimu

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

   

if __name__ == "__main__":
    task = Hyperboloid()
    task._setup()
