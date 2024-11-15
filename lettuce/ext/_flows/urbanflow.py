import warnings
from typing import Union, List, Optional

import numpy as np
import torch

from . import ExtFlow
from ... import UnitConversion, Context, Stencil, Equilibrium
from ...util import append_axes
from .. import (EquilibriumBoundaryPU, BounceBackBoundary,
                EquilibriumOutletP, AntiBounceBackOutlet)

__all__ = ['UrbanFlow']


class UrbanFlow(ExtFlow):
    """
    Flow class to simulate the flow around a set of buildings in an urban setting (mask).
    It consists of one inflow (equilibrium boundary)
    and one outflow (anti-bounce-back-boundary), leading to a flow in positive
    x direction. The flow in the top of the simulation box adheres to an EQ as well.

    Parameters
    ----------
    resolution : Tuple[int]:
        Grid resolution.
    domain_length_x : float
        Length of the domain in physical units.

    Attributes
    ----------
    _mask : torch.Tensor with dtype = bool
        Boolean mask to define the obstacle. The shape of this object is the
        shape of the grid.
        Initially set to zero (no obstacle).

    Examples
    --------

    >>> 
   """

    def __init__(self, context: Context, resolution: Union[int, List[int]],
                 reynolds_number, mach_number, domain_length_x,
                 char_length=1, char_velocity=1,
                 stencil: Optional[Stencil] = None,
                 equilibrium: Optional[Equilibrium] = None):
        self.char_length_lu = resolution[0] / domain_length_x * char_length
        self.char_length = char_length
        self.char_velocity = char_velocity
        self.resolution = self.make_resolution(resolution, stencil)
        self._mask = torch.zeros(self.resolution, dtype=torch.bool)
        ExtFlow.__init__(self, context, resolution, reynolds_number,
                         mach_number, stencil, equilibrium)

    def make_units(self, reynolds_number, mach_number, resolution: List[int]
                   ) -> 'UnitConversion':
        return UnitConversion(
            reynolds_number=reynolds_number, mach_number=mach_number,
            characteristic_length_lu=self.char_length_lu,
            characteristic_length_pu=self.char_length,
            characteristic_velocity_pu=self.char_velocity
        )

    def make_resolution(self, resolution: Union[int, List[int]],
                        stencil: Optional['Stencil'] = None) -> List[int]:
        if isinstance(resolution, int):
            return [resolution] * (stencil.d or self.stencil.d)
        else:
            return resolution

    @property
    def solid_mask(self):
        if not hasattr(self.collision_data, 'solid_mask'):
            self.calculate_points_inside()
        return self.collision_data.solid_mask
    
    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, m):
        assert ((isinstance(m, np.ndarray) or isinstance(m, torch.Tensor)) and
                all(m.shape[dim] == self.resolution[dim] for dim in range(
                    self.stencil.d)))
        self._mask = self.context.convert_to_tensor(m, dtype=torch.bool)    

    def initial_pu(self) -> (float, Union[np.array, torch.Tensor]):
        p = np.zeros_like(self.grid[0], dtype=float)[None, ...]

        # free-flow velocity profile according to
        # https://www.simscale.com/knowledge-base/atmospheric-boundary-layer-abl/.
        z = self.grid[2]        
        K = 0.4
        z0 = 2 # z0 should be expected around 1 according to descriptions, but in long-term the profile is much slower
        u_ref = 0.99
        H_ref = 2
        u_dash = K * u_ref / np.log((H_ref + z0) / z0)
        h_cartesian = z
        z_solid = torch.max(torch.where(self.mask, z, z.min() - 1), dim=2)[0]
        h = h_cartesian - z_solid[:, :, None]
        self.ux = u_dash / K * torch.log((torch.where(h > 0, h, 0) + z0) / z0)
        u = torch.stack((self.ux, torch.zeros_like(z), torch.zeros_like(z)), 0)
        
        return p, u

    @property
    def grid(self):
        xyz = tuple(self.units.convert_length_to_pu(torch.arange(n)) for n in
                    self.resolution)
        return torch.meshgrid(*xyz, indexing='ij')

    @property
    def boundaries(self):
        x = self.grid[0]
        z = self.grid[2]
        ux_in = self.ux[0, :, :][None, :, :]
        u_in = torch.stack((ux_in, torch.zeros_like(ux_in), torch.zeros_like(ux_in)), 0)
        
        ux_top_in = self.ux[-1, :, :][None, :, :]
        u_top_in = torch.stack((ux_top_in, torch.zeros_like(ux_in), torch.zeros_like(ux_in)), 0)
        
        return [
            EquilibriumBoundaryPU( # inlet boundary
                                  context=self.context,
                                  mask=torch.abs(x) < 1e-6,
                                  velocity=u_in
                                  ),
            EquilibriumBoundaryPU(  # top boundary
                                context = self.context,
                                mask= z >= z.max(),
                                velocity=u_top_in
                                ),
            AntiBounceBackOutlet(self._unit_vector().tolist(),
                                 self),                                 
            BounceBackBoundary(self.mask)
        ]

    def _unit_vector(self, i=0):
        return torch.eye(self.stencil.d)[i]

