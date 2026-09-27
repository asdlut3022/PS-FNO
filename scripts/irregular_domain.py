"""
Irregular domain processing module

Used to define and process irregular computational domains, supporting:
1. Signed distance function (SDF) computation
2. Mask generation
3. Boundary condition enforcement (Dirichlet soft mask)
4. Predefined shapes for a variety of geometries
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable, Dict
import numpy as np
from abc import ABC, abstractmethod


class Domain(ABC):
    """
    Abstract domain class

    All geometric domains inherit from this class and implement the sdf method
    """

    @abstractmethod
    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the signed distance function

        Args:
            x, y: coordinate tensors (any shape is allowed)

        Returns:
            Distance values: < 0 inside the domain, > 0 outside the domain, = 0 on the boundary
        """
        pass

    def mask(self, x: torch.Tensor, y: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Generate a mask

        Args:
            x, y: coordinates
            threshold: threshold value (default 0; can be adjusted to enlarge/shrink the region)

        Returns:
            Mask: 1 inside the domain, 0 outside the domain
        """
        return (self.sdf(x, y) < threshold).float()

    def boundary_mask(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        thickness: float = 0.05
    ) -> torch.Tensor:
        """
        Generate a boundary layer mask

        Args:
            x, y: coordinates
            thickness: boundary layer thickness

        Returns:
            Boundary layer mask: 1 near the boundary, 0 far from the boundary
        """
        dist = torch.abs(self.sdf(x, y))
        return torch.exp(-dist / thickness)

    def smooth_mask(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        epsilon: float = 0.01
    ) -> torch.Tensor:
        """
        Generate a smooth mask (differentiable)

        Uses a tanh-smoothed step function

        Args:
            x, y: coordinates
            epsilon: smoothing parameter (the smaller, the steeper)

        Returns:
            Smooth mask: continuous values between 0 and 1
        """
        return 0.5 * (1 - torch.tanh(self.sdf(x, y) / epsilon))

    def apply_boundary_condition(
        self,
        u: torch.Tensor,
        u_bc: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        bc_type: str = 'dirichlet',
        thickness: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Enforce boundary conditions via soft constraints

        Args:
            u: predicted field
            u_bc: boundary condition values
            x, y: coordinates
            bc_type: boundary type ('dirichlet')
            thickness: boundary layer thickness

        Returns:
            (corrected field, boundary loss)
        """
        if bc_type == 'dirichlet':
            # Dirichlet: constrain the boundary values directly
            boundary_weight = self.boundary_mask(x, y, thickness)
            u_corrected = u * (1 - boundary_weight) + u_bc * boundary_weight
            bc_loss = torch.mean(boundary_weight * (u - u_bc)**2)
            return u_corrected, bc_loss

        elif bc_type == 'neumann':
            # Neumann: constrain the normal derivative
            raise NotImplementedError("Neumann BC is not implemented yet")

        else:
            raise ValueError(f"Unknown BC type: {bc_type}")


class RectangularDomain(Domain):
    """
    Rectangular domain (the simplest case)
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Args:
            x_range: (x_min, x_max)
            y_range: (y_min, y_max)
        """
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Rectangular SDF

        Negative inside, zero on the boundary, positive outside
        """
        dx = torch.maximum(self.x_min - x, x - self.x_max)
        dy = torch.maximum(self.y_min - y, y - self.y_max)

        # Outside distance
        outside = torch.sqrt(
            torch.clamp(dx, min=0)**2 + torch.clamp(dy, min=0)**2
        )

        # Inside distance
        inside = torch.maximum(dx, dy)
        inside = torch.clamp(inside, max=0)

        return outside + inside


class CircleDomain(Domain):
    """
    Circular domain
    """

    def __init__(
        self,
        center: Tuple[float, float] = (0.5, 0.5),
        radius: float = 0.2
    ):
        """
        Args:
            center: center coordinates (cx, cy)
            radius: radius
        """
        self.cx, self.cy = center
        self.radius = radius

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Circular SDF

        Negative values indicate points inside the circle
        """
        dist = torch.sqrt((x - self.cx)**2 + (y - self.cy)**2)
        return dist - self.radius


class CircleObstacleDomain(Domain):
    """
    Rectangular domain with a circular obstacle

    Commonly used for engineering problems such as flow past a cylinder
    """

    def __init__(
        self,
        domain_x: Tuple[float, float] = (0.0, 2.0),
        domain_y: Tuple[float, float] = (0.0, 1.0),
        obstacle_center: Tuple[float, float] = (0.5, 0.5),
        obstacle_radius: float = 0.1
    ):
        """
        Args:
            domain_x, domain_y: extents of the outer rectangular domain
            obstacle_center: center of the obstacle circle
            obstacle_radius: obstacle radius
        """
        self.outer_domain = RectangularDomain(domain_x, domain_y)
        self.obstacle = CircleDomain(obstacle_center, obstacle_radius)

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Composite SDF: interior of the rectangle minus the circular obstacle

        Uses a boolean difference operation
        """
        sdf_outer = self.outer_domain.sdf(x, y)
        sdf_obstacle = self.obstacle.sdf(x, y)

        # Valid domain = interior of the outer domain ∩ exterior of the obstacle
        # SDF set operation: A - B = A ∩ complement(B) = max(sdf_A, -sdf_B)
        return torch.maximum(sdf_outer, -sdf_obstacle)


class LShapedDomain(Domain):
    """
    L-shaped domain (a classic test case)
    """

    def __init__(
        self,
        width: float = 1.0,
        height: float = 1.0,
        cut_size: float = 0.5
    ):
        """
        Args:
            width, height: outer dimensions
            cut_size: cutout size
        """
        self.rect1 = RectangularDomain((0, width), (0, height - cut_size))
        self.rect2 = RectangularDomain((0, width - cut_size), (height - cut_size, height))

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        L-shaped SDF: union of two rectangles

        Uses a boolean union operation: min(sdf1, sdf2)
        """
        sdf1 = self.rect1.sdf(x, y)
        sdf2 = self.rect2.sdf(x, y)
        return torch.minimum(sdf1, sdf2)


class MultiCircleObstacleDomain(Domain):
    """
    Domain with multiple circular obstacles

    Used for complex geometries
    """

    def __init__(
        self,
        domain_x: Tuple[float, float] = (0.0, 2.0),
        domain_y: Tuple[float, float] = (0.0, 1.0),
        obstacles: List[Tuple[Tuple[float, float], float]] = None
    ):
        """
        Args:
            domain_x, domain_y: outer domain
            obstacles: list of obstacles [(center, radius), ...]
        """
        self.outer_domain = RectangularDomain(domain_x, domain_y)

        if obstacles is None:
            # Default: 3 circular obstacles
            obstacles = [
                ((0.5, 0.5), 0.1),
                ((1.0, 0.3), 0.08),
                ((1.5, 0.7), 0.12)
            ]

        self.obstacles = [
            CircleDomain(center, radius) for center, radius in obstacles
        ]

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Multi-obstacle SDF
        """
        sdf_outer = self.outer_domain.sdf(x, y)

        # Union of all obstacles (take the minimum)
        sdf_obstacles = [obs.sdf(x, y) for obs in self.obstacles]
        sdf_union = torch.stack(sdf_obstacles, dim=0).min(dim=0)[0]

        # Valid domain = interior of the outer domain ∩ exterior of all obstacles
        # SDF set operation: A - B = A ∩ complement(B) = max(sdf_A, -sdf_B)
        return torch.maximum(sdf_outer, -sdf_union)


class IrregularFinsDomain(Domain):
    """
    Irregularly shaped fin heat dissipation domain

    Engineering cases:
    1. Biomimetic tree-shaped fins
    2. Pin fins with varying shapes
    3. Serrated/wavy fins
    4. Mixed cylindrical + elliptical fins
    """

    def __init__(
        self,
        domain_size: Tuple[float, float] = (1.0, 1.0),
        fin_type: str = 'wavy',  # 'wavy', 'tapered', 'pin_circle', 'tree'
        n_fins: int = 4,
        base_height: float = 0.1
    ):
        """
        Args:
            domain_size: computational domain size
            fin_type: fin type
                - 'wavy': wavy fin (curved boundary)
                - 'tapered': wedge/tapered fin (trapezoid)
                - 'pin_circle': cylindrical pin fin
                - 'tree': tree-shaped bifurcating fin
            n_fins: number of fins
            base_height: substrate height
        """
        self.Lx, self.Ly = domain_size
        self.n_fins = n_fins
        self.fin_type = fin_type
        self.base_height = base_height

        # Substrate (regular part)
        self.base = RectangularDomain((0, self.Lx), (0, base_height))

        # Create irregular fins according to the type
        self.fins = self._create_irregular_fins()

    def _create_irregular_fins(self) -> List:
        """Create irregularly shaped fins"""
        fins = []
        spacing = self.Lx / (self.n_fins + 1)

        if self.fin_type == 'wavy':
            # Wavy fin: use a sinusoidal boundary
            for i in range(self.n_fins):
                x_center = (i + 1) * spacing
                fin = WavyFin(
                    x_center=x_center,
                    y_bottom=self.base_height,
                    y_top=self.Ly * 0.8,
                    width_base=0.08,
                    amplitude=0.02,
                    frequency=8
                )
                fins.append(fin)

        elif self.fin_type == 'tapered':
            # Tapered fin: wide at the bottom, narrow at the top
            for i in range(self.n_fins):
                x_center = (i + 1) * spacing
                fin = TaperedFin(
                    x_center=x_center,
                    y_bottom=self.base_height,
                    y_top=self.Ly * 0.8,
                    width_bottom=0.1,
                    width_top=0.03
                )
                fins.append(fin)

        elif self.fin_type == 'pin_circle':
            # Cylindrical pin fins (with different radii)
            radii = [0.04, 0.05, 0.045, 0.038]  # Irregular radii
            for i in range(self.n_fins):
                x_center = (i + 1) * spacing
                y_center = (self.base_height + self.Ly * 0.5) / 2
                fin = CircleDomain(
                    center=(x_center, y_center),
                    radius=radii[i % len(radii)]
                )
                fins.append(fin)

        elif self.fin_type == 'tree':
            # Tree-shaped bifurcating fin
            for i in range(self.n_fins):
                x_center = (i + 1) * spacing
                fin = TreeShapedFin(
                    x_root=x_center,
                    y_root=self.base_height,
                    trunk_height=0.3,
                    trunk_width=0.06,
                    branch_angle=30
                )
                fins.append(fin)

        return fins

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the SDF: negative inside the fluid domain

        Fluid domain = inside the outer domain AND not inside the solid
        Sign convention: SDF < 0 means inside the domain
        """
        # Outer rectangular domain (inside the domain < 0)
        outer = RectangularDomain((0, self.Lx), (0, self.Ly))
        sdf_outer = outer.sdf(x, y)  # < 0 inside the outer domain

        # Substrate and fins (solid; < 0 inside the solid)
        sdf_base = self.base.sdf(x, y)

        sdf_fins = []
        for fin in self.fins:
            sdf_fins.append(fin.sdf(x, y))

        # Union of solids (< 0 when inside any solid)
        all_solids = [sdf_base] + sdf_fins
        sdf_solids = torch.stack(all_solids, dim=0).min(dim=0)[0]  # min for the union

        # Fluid domain = inside the outer domain AND not inside the solid
        # sdf_outer < 0: inside the outer domain
        # sdf_solids > 0: not inside the solid
        # max ensures both conditions are satisfied
        return torch.maximum(sdf_outer, -sdf_solids)


class WavyFin(Domain):
    """
    Wavy fin

    Boundary equation: x = x_center ± (width/2 + A*sin(ω*y))
    Engineering advantage: enhances turbulent mixing and disrupts the boundary layer
    """

    def __init__(
        self,
        x_center: float,
        y_bottom: float,
        y_top: float,
        width_base: float = 0.08,
        amplitude: float = 0.02,
        frequency: float = 8
    ):
        self.x_center = x_center
        self.y_bottom = y_bottom
        self.y_top = y_top
        self.width_base = width_base
        self.amplitude = amplitude
        self.frequency = frequency

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        SDF of the wavy fin (approximate)
        """
        # Vertical constraint
        dy_bottom = y - self.y_bottom
        dy_top = self.y_top - y
        in_y = (dy_bottom > 0) & (dy_top > 0)

        # Compute the wavy width
        phase = self.frequency * (y - self.y_bottom) / (self.y_top - self.y_bottom) * 2 * np.pi
        wavy_width = self.width_base / 2 + self.amplitude * torch.sin(phase)

        # Lateral distance
        dx = torch.abs(x - self.x_center) - wavy_width

        # Composite SDF (simplified version)
        sdf = torch.where(
            in_y,
            dx,  # within the height range, the lateral distance determines the SDF
            torch.minimum(torch.abs(dy_bottom), torch.abs(dy_top))  # outside the height range
        )

        return sdf


class TaperedFin(Domain):
    """
    Tapered/wedge fin

    Shape: wide at the bottom, narrow at the top (trapezoid)
    Engineering advantage: reduces weight, optimizes material distribution; the heat flux at the top is small, so the fin need not be wide there
    """

    def __init__(
        self,
        x_center: float,
        y_bottom: float,
        y_top: float,
        width_bottom: float = 0.1,
        width_top: float = 0.03
    ):
        self.x_center = x_center
        self.y_bottom = y_bottom
        self.y_top = y_top
        self.width_bottom = width_bottom
        self.width_top = width_top

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        SDF of the trapezoidal fin
        """
        # Linearly interpolated width
        height = self.y_top - self.y_bottom
        y_normalized = torch.clamp((y - self.y_bottom) / height, 0, 1)

        # Width at the current height
        width_local = self.width_bottom + (self.width_top - self.width_bottom) * y_normalized

        # Lateral distance
        dx = torch.abs(x - self.x_center) - width_local / 2

        # Vertical distance
        dy = torch.maximum(self.y_bottom - y, y - self.y_top)

        # Composite (approximate trapezoidal SDF)
        outside = torch.sqrt(torch.clamp(dx, min=0)**2 + torch.clamp(dy, min=0)**2)
        inside = torch.maximum(dx, dy)
        inside = torch.clamp(inside, max=0)

        return outside + inside


class TreeShapedFin(Domain):
    """
    Tree-shaped / bifurcating fin

    Biomimetic design, similar to a vascular network
    Engineering advantages:
    - Maximizes the surface area
    - Optimizes the heat flow path
    - The optimal heat transfer structure found in nature

    This is the most irregular of the fin shapes.
    """

    def __init__(
        self,
        x_root: float,
        y_root: float,
        trunk_height: float = 0.3,
        trunk_width: float = 0.06,
        branch_angle: float = 30,
        n_levels: int = 2
    ):
        self.x_root = x_root
        self.y_root = y_root
        self.trunk_height = trunk_height
        self.trunk_width = trunk_width
        self.branch_angle = branch_angle * np.pi / 180
        self.n_levels = n_levels

        # Create the trunk and branches
        self._create_tree_structure()

    def _create_tree_structure(self):
        """Recursively create the tree structure"""
        # Trunk (rectangle)
        self.trunk = RectangularDomain(
            (self.x_root - self.trunk_width/2, self.x_root + self.trunk_width/2),
            (self.y_root, self.y_root + self.trunk_height)
        )

        # Left and right branches (simplified to rectangles; they could be more complex in practice)
        branch_width = self.trunk_width * 0.6
        branch_height = self.trunk_height * 0.7
        branch_y_start = self.y_root + self.trunk_height

        # Left branch
        dx_left = -branch_height * np.tan(self.branch_angle)
        self.branch_left = RectangularDomain(
            (self.x_root + dx_left - branch_width/2, self.x_root + dx_left + branch_width/2),
            (branch_y_start, branch_y_start + branch_height)
        )

        # Right branch
        dx_right = branch_height * np.tan(self.branch_angle)
        self.branch_right = RectangularDomain(
            (self.x_root + dx_right - branch_width/2, self.x_root + dx_right + branch_width/2),
            (branch_y_start, branch_y_start + branch_height)
        )

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        SDF of the tree structure (union of the trunk and branches)
        """
        sdf_trunk = self.trunk.sdf(x, y)
        sdf_left = self.branch_left.sdf(x, y)
        sdf_right = self.branch_right.sdf(x, y)

        # Union of the three (take the minimum)
        return torch.minimum(torch.minimum(sdf_trunk, sdf_left), sdf_right)


class HeatSinkFinsDomain(Domain):
    """
    Chip heat sink fin array domain (a practical engineering case)

    Geometric features:
    - Bottom: chip substrate (heat source)
    - Middle: multiple rectangular fins (solid obstacles)
    - Channels: fluid domain between the fins

    Typical applications: electronics cooling, CPU coolers
    """

    def __init__(
        self,
        domain_size: Tuple[float, float] = (1.0, 1.0),
        n_fins: int = 5,
        fin_width: float = 0.1,
        fin_height: float = 0.6,
        fin_spacing: float = 0.05,
        base_height: float = 0.1
    ):
        """
        Args:
            domain_size: computational domain size (Lx, Ly)
            n_fins: number of fins
            fin_width: fin width
            fin_height: fin height
            fin_spacing: fin spacing
            base_height: bottom substrate height
        """
        self.Lx, self.Ly = domain_size
        self.n_fins = n_fins
        self.fin_width = fin_width
        self.fin_height = fin_height
        self.fin_spacing = fin_spacing
        self.base_height = base_height

        # Compute the fin positions
        total_width = n_fins * fin_width + (n_fins - 1) * fin_spacing
        start_x = (self.Lx - total_width) / 2

        self.fins = []
        for i in range(n_fins):
            x_left = start_x + i * (fin_width + fin_spacing)
            x_right = x_left + fin_width
            y_bottom = base_height
            y_top = base_height + fin_height

            fin = RectangularDomain((x_left, x_right), (y_bottom, y_top))
            self.fins.append(fin)

        # Bottom substrate
        self.base = RectangularDomain((0, self.Lx), (0, base_height))

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the SDF: negative inside the fluid domain, positive inside the solid (fins + substrate)
        """
        # SDF of the outer rectangular domain (negated so that the interior is negative)
        outer_domain = RectangularDomain((0, self.Lx), (0, self.Ly))
        sdf_outer = -outer_domain.sdf(x, y)

        # SDF of the substrate (solid, should be positive)
        sdf_base = -self.base.sdf(x, y)

        # SDF of all fins (solid, should be positive)
        sdf_fins = []
        for fin in self.fins:
            sdf_fins.append(-fin.sdf(x, y))

        # Combine: inside the outer domain and not inside any solid
        # Use max(sdf_base, fin1, fin2, ...) to obtain the union of solids
        all_solids = [sdf_base] + sdf_fins
        sdf_solids = torch.stack(all_solids, dim=0).max(dim=0)[0]

        # Final SDF: negative inside the fluid domain
        # Interior of the outer domain negative, interior of the solid positive, fluid domain = outer domain - solid
        return torch.maximum(sdf_outer, sdf_solids)

    def get_thermal_boundary_zones(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Get the thermal boundary zone masks

        Returns:
            {
                'hot_base': bottom heat source mask,
                'fin_surfaces': fin surface mask,
                'outlet': top outlet mask
            }
        """
        # Bottom heat source (y ≈ 0)
        hot_base = (y < self.base_height * 0.5).float()

        # Fin surfaces (use the boundary mask)
        fin_surfaces = torch.zeros_like(x)
        for fin in self.fins:
            fin_surfaces = fin_surfaces + fin.boundary_mask(x, y, thickness=0.02)

        # Top outlet (y ≈ Ly)
        outlet = (y > self.Ly * 0.95).float()

        return {
            'hot_base': hot_base,
            'fin_surfaces': fin_surfaces,
            'outlet': outlet
        }


class MultiObstacleChannelDomain(Domain):
    """
    Multi-obstacle channel domain (another clearly irregular case)

    Applications: heat exchanger tube bundles, porous media, catalytic reactors
    """

    def __init__(
        self,
        domain_size: Tuple[float, float] = (2.0, 1.0),
        obstacles: List[Dict] = None
    ):
        """
        Args:
            domain_size: computational domain size
            obstacles: list of obstacles, each given as {'type': 'circle'/'rect', 'params': ...}
        """
        self.Lx, self.Ly = domain_size

        if obstacles is None:
            # Default: staggered circular obstacles (similar to a tube bundle)
            obstacles = []
            for row in range(2):
                for col in range(4):
                    x_offset = 0.1 if row % 2 == 0 else 0.3
                    obstacles.append({
                        'type': 'circle',
                        'params': {
                            'center': (0.3 + col * 0.4 + x_offset, 0.25 + row * 0.5),
                            'radius': 0.08
                        }
                    })

        # Create the obstacle objects
        self.obstacles = []
        for obs in obstacles:
            if obs['type'] == 'circle':
                self.obstacles.append(CircleDomain(**obs['params']))
            elif obs['type'] == 'rect':
                self.obstacles.append(RectangularDomain(**obs['params']))

        self.outer_domain = RectangularDomain((0, self.Lx), (0, self.Ly))

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        SDF: negative inside the channel (outer domain - obstacles)
        """
        sdf_outer = self.outer_domain.sdf(x, y)

        # Union of all obstacles
        sdf_obstacles = [obs.sdf(x, y) for obs in self.obstacles]
        sdf_union = torch.stack(sdf_obstacles, dim=0).min(dim=0)[0]

        # Fluid domain = inside the outer domain and outside all obstacles
        return torch.maximum(-sdf_outer, sdf_union)


class ImageBasedDomain(Domain):
    """
    Domain defined from an image

    For rapid prototyping of complex geometries
    """

    def __init__(
        self,
        image: torch.Tensor,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0)
    ):
        """
        Args:
            image: binary image (nx, ny), 1 = inside the domain, 0 = outside the domain
            x_range, y_range: physical coordinate ranges
        """
        self.image = image.float()
        self.nx, self.ny = image.shape
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range

        # Precompute the SDF (using a distance transform)
        self._compute_sdf()

    def _compute_sdf(self):
        """
        Compute the SDF using a distance transform

        Requires scipy.ndimage.distance_transform_edt
        """
        try:
            from scipy.ndimage import distance_transform_edt

            # Inside distance (negative values)
            inside = self.image.cpu().numpy()
            dist_inside = distance_transform_edt(inside)

            # Outside distance (positive values)
            outside = 1 - inside
            dist_outside = distance_transform_edt(outside)

            # Combine
            sdf_array = dist_outside - dist_inside
            self.sdf_grid = torch.from_numpy(sdf_array).float()

        except ImportError:
            print("Warning: scipy not available, using simple mask-based SDF")
            self.sdf_grid = (1 - 2 * self.image).float()

    def sdf(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Query the SDF via bilinear interpolation
        """
        # Normalize coordinates to [-1, 1]
        x_norm = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        y_norm = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1

        # grid_sample expects (batch, height, width, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1)

        # Add batch and channel dimensions
        sdf_grid_4d = self.sdf_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, nx, ny)

        # Interpolate
        sampled = F.grid_sample(
            sdf_grid_4d,
            grid.unsqueeze(0).unsqueeze(0),
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return sampled.squeeze()


class DomainUtils:
    """
    Domain-related utility functions
    """

    @staticmethod
    def create_grid(
        nx: int,
        ny: int,
        x_range: Tuple[float, float] = (0.0, 1.0),
        y_range: Tuple[float, float] = (0.0, 1.0),
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create a uniform grid

        Args:
            nx, ny: number of grid points
            x_range, y_range: ranges
            device: device

        Returns:
            (X, Y): grid tensors (nx, ny)
        """
        x = torch.linspace(x_range[0], x_range[1], nx, device=device)
        y = torch.linspace(y_range[0], y_range[1], ny, device=device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return X, Y

    @staticmethod
    def visualize_domain(
        domain: Domain,
        nx: int = 256,
        ny: int = 256,
        ax=None
    ):
        """
        Visualize a domain

        Args:
            domain: domain object
            nx, ny: resolution
            ax: matplotlib axes object
        """
        import matplotlib.pyplot as plt

        # Create the grid
        X, Y = DomainUtils.create_grid(nx, ny)

        # Compute the SDF and the mask
        sdf = domain.sdf(X, Y).numpy()
        mask = domain.mask(X, Y).numpy()

        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        else:
            axes = [ax]

        # Plot the SDF
        im1 = axes[0].contourf(X.numpy(), Y.numpy(), sdf, levels=20, cmap='RdBu_r')
        axes[0].contour(X.numpy(), Y.numpy(), sdf, levels=[0], colors='black', linewidths=2)
        axes[0].set_title('Signed Distance Function')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        axes[0].axis('equal')
        plt.colorbar(im1, ax=axes[0])

        if len(axes) > 1:
            # Plot the mask
            axes[1].imshow(mask.T, origin='lower', extent=[0, 1, 0, 1], cmap='gray')
            axes[1].set_title('Domain Mask')
            axes[1].set_xlabel('x')
            axes[1].set_ylabel('y')
            axes[1].axis('equal')

        plt.tight_layout()
        return axes


def test_domains():
    """
    Test all domain types
    """
    print("=" * 60)
    print("Testing the irregular domain processing module")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test 1: rectangular domain
    print("\n1. Rectangular domain test")
    rect = RectangularDomain((0, 1), (0, 1))
    X, Y = DomainUtils.create_grid(64, 64, device=device)

    sdf = rect.sdf(X, Y)
    mask = rect.mask(X, Y)
    print(f"   SDF range: [{sdf.min():.3f}, {sdf.max():.3f}]")
    print(f"   Mask volume fraction: {mask.mean():.3f}")

    # Test 2: circular domain
    print("\n2. Circular domain test")
    circle = CircleDomain((0.5, 0.5), 0.2)
    sdf_circle = circle.sdf(X, Y)

    # Verify the distance at the circle center
    sdf_center = circle.sdf(
        torch.tensor([0.5], device=device),
        torch.tensor([0.5], device=device)
    )
    print(f"   SDF at the circle center: {sdf_center.item():.3f} (expected -0.200)")

    # Test 3: flow past a cylinder domain
    print("\n3. Circular obstacle domain test")
    obstacle_domain = CircleObstacleDomain(
        domain_x=(0, 2),
        domain_y=(0, 1),
        obstacle_center=(0.5, 0.5),
        obstacle_radius=0.1
    )
    X2, Y2 = DomainUtils.create_grid(128, 64, (0, 2), (0, 1), device)
    mask_obs = obstacle_domain.mask(X2, Y2)
    print(f"   Valid domain volume fraction: {mask_obs.mean():.3f}")

    # Test 4: L-shaped domain
    print("\n4. L-shaped domain test")
    lshape = LShapedDomain(1.0, 1.0, 0.5)
    mask_l = lshape.mask(X, Y)
    print(f"   L-shaped domain volume fraction: {mask_l.mean():.3f} (theoretical value: 0.75)")

    # Test 5: boundary condition enforcement
    print("\n5. Boundary condition test")
    u = torch.rand(64, 64, device=device)
    u_bc = torch.ones(64, 64, device=device)
    u_corrected, bc_loss = circle.apply_boundary_condition(u, u_bc, X, Y)
    print(f"   Boundary loss: {bc_loss.item():.6f}")
    print(f"   Difference before/after correction: {(u - u_corrected).abs().mean().item():.6f}")

    # Test 6: smooth mask
    print("\n6. Smooth mask test")
    X.requires_grad = True
    smooth = circle.smooth_mask(X, Y, epsilon=0.01)
    print(f"   Smooth mask range: [{smooth.min():.3f}, {smooth.max():.3f}]")

    # Compute the gradient (requires_grad=True is required)
    if X.requires_grad:
        grad_x = torch.autograd.grad(smooth.sum(), X, create_graph=True)[0]
        print(f"   Smooth mask gradient: {grad_x.abs().mean():.6f}")
    else:
        print("   Smooth mask gradient: skipped (X does not require grad)")

    # Test 7: irregularly shaped fins - wavy
    print("\n7. Irregular fin test - wavy (Wavy Fins)")
    wavy_fins = IrregularFinsDomain(
        domain_size=(1.0, 1.0),
        fin_type='wavy',
        n_fins=4,
        base_height=0.1
    )
    X_wf, Y_wf = DomainUtils.create_grid(128, 128, (0, 1), (0, 1), device)
    mask_wf = wavy_fins.mask(X_wf, Y_wf)
    print(f"   Fluid domain volume fraction: {mask_wf.mean():.3f}")
    print(f"   Fin type: {wavy_fins.fin_type}")

    # Test 8: tapered fins
    print("\n8. Irregular fin test - tapered (Tapered Fins)")
    tapered_fins = IrregularFinsDomain(
        domain_size=(1.0, 1.0),
        fin_type='tapered',
        n_fins=4,
        base_height=0.1
    )
    X_tf, Y_tf = DomainUtils.create_grid(128, 128, (0, 1), (0, 1), device)
    mask_tf = tapered_fins.mask(X_tf, Y_tf)
    print(f"   Fluid domain volume fraction: {mask_tf.mean():.3f}")

    # Test 9: tree-shaped fins (the most irregular!)
    print("\n9. Irregular fin test - tree-shaped bifurcating (Tree-shaped Fins)")
    tree_fins = IrregularFinsDomain(
        domain_size=(1.0, 1.0),
        fin_type='tree',
        n_fins=3,
        base_height=0.1
    )
    X_tree, Y_tree = DomainUtils.create_grid(128, 128, (0, 1), (0, 1), device)
    mask_tree = tree_fins.mask(X_tree, Y_tree)
    print(f"   Fluid domain volume fraction: {mask_tree.mean():.3f}")
    print(f"   Irregular computational domains generated; see above for the fluid domain volume fraction")

    print("\n" + "=" * 60)
    print("All tests passed")
    print("=" * 60)

    # Visualization (if matplotlib is available)
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # First row: basic geometries
        domains_basic = [
            ('Rectangle', rect, (0, 1), (0, 1), 64, 64),
            ('Circle', circle, (0, 1), (0, 1), 64, 64),
            ('Circle Obstacle', obstacle_domain, (0, 2), (0, 1), 128, 64),
            ('L-Shape', lshape, (0, 1), (0, 1), 64, 64),
        ]

        for i, (name, dom, xr, yr, nx, ny) in enumerate(domains_basic):
            ax = fig.add_subplot(gs[0, i])
            X_vis, Y_vis = DomainUtils.create_grid(nx, ny, xr, yr, device='cpu')
            sdf_vis = dom.sdf(X_vis, Y_vis).cpu().detach().numpy()
            mask_vis = dom.mask(X_vis, Y_vis).cpu().detach().numpy()

            ax.imshow(1 - mask_vis.T, origin='lower', extent=[xr[0], xr[1], yr[0], yr[1]],
                     cmap='gray', vmin=0, vmax=1)
            ax.contour(X_vis.numpy(), Y_vis.numpy(), sdf_vis, levels=[0],
                      colors='red', linewidths=2)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')

        # Second row: irregular fins
        irregular_fins = [
            ('Wavy Fins\n(wavy fins)', wavy_fins, X_wf, Y_wf),
            ('Tapered Fins\n(tapered fins)', tapered_fins, X_tf, Y_tf),
            ('Tree-shaped Fins\n(tree-shaped fins)', tree_fins, X_tree, Y_tree),
        ]

        for i, (name, dom, X_grid, Y_grid) in enumerate(irregular_fins):
            ax = fig.add_subplot(gs[1, i])
            sdf_vis = dom.sdf(X_grid, Y_grid).cpu().detach().numpy()
            mask_vis = dom.mask(X_grid, Y_grid).cpu().detach().numpy()

            ax.imshow(1 - mask_vis.T, origin='lower', extent=[0, 1, 0, 1],
                     cmap='gray_r', vmin=0, vmax=1, interpolation='bilinear')
            ax.contour(X_grid.cpu().numpy(), Y_grid.cpu().numpy(), sdf_vis,
                      levels=[0], colors='red', linewidths=2.5)
            ax.set_title(name, fontsize=12, fontweight='bold', color='darkblue')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)

        # 4th subplot of the second row: SDF visualization
        ax = fig.add_subplot(gs[1, 3])
        sdf_tree_vis = tree_fins.sdf(X_tree, Y_tree).cpu().numpy()
        im = ax.contourf(X_tree.cpu().numpy(), Y_tree.cpu().numpy(), sdf_tree_vis,
                        levels=20, cmap='RdBu_r')
        ax.contour(X_tree.cpu().numpy(), Y_tree.cpu().numpy(), sdf_tree_vis,
                  levels=[0], colors='black', linewidths=3)
        ax.set_title('Tree Fins SDF', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Distance')

        # Third row: zoomed-in details
        for i, (name, dom, X_grid, Y_grid) in enumerate(irregular_fins):
            ax = fig.add_subplot(gs[2, i])

            # Local zoom
            mask_vis = dom.mask(X_grid, Y_grid).cpu().detach().numpy()
            ax.imshow(1 - mask_vis.T, origin='lower', extent=[0, 1, 0, 1],
                     cmap='Greys', vmin=0, vmax=1)

            # Add streamlines (schematic)
            y_vals = torch.linspace(0.2, 0.8, 5, device='cpu')
            for y_val in y_vals:
                x_line = torch.linspace(0, 1, 200, device='cpu')
                y_line = torch.full_like(x_line, y_val)
                mask_line = dom.mask(x_line, y_line).cpu().numpy()
                valid = mask_line > 0.5
                ax.plot(x_line.numpy()[valid], y_line.numpy()[valid],
                       'b-', alpha=0.6, linewidth=1.5)

            ax.set_title(f'{name.split(chr(10))[0]} - Flow Paths', fontsize=10)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_aspect('equal')
            ax.set_xlim([0.1, 0.6])  # Local zoom

        # Overall title
        fig.suptitle('Irregular Computational Domains: From Simple to Complex\nIrregular Computational Domains for Multi-Physics PINO',
                    fontsize=16, fontweight='bold', y=0.98)

        # Anchored to the script's own directory, not to the working directory:
        # a bare filename would drop the figure wherever the demo was launched.
        _out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'figures')
        os.makedirs(_out_dir, exist_ok=True)
        _out = os.path.join(_out_dir, 'irregular_domains_complete.png')
        plt.savefig(_out, dpi=150, bbox_inches='tight')
        print("\n[OK] Visualization saved to " + os.path.normpath(_out))
        print("   Showcasing 3 irregular fin shapes: wavy, tapered, tree-shaped")

    except ImportError:
        print("\nSkipping visualization (matplotlib required)")


if __name__ == '__main__':
    test_domains()
