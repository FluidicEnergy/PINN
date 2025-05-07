import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage
import time
#from torch_geometric.nn import GCNConv, knn_graph
import math
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# CSG 관련 클래스 및 함수

class Primitive:
    def __init__(self, f):
        self.f = f

    def __call__(self, x, y):
        return self.f(x, y)

class Union(Primitive):
    def __init__(self, *primitives):
        self.primitives = primitives

    def __call__(self, x, y):
        return torch.min(torch.stack([p(x, y) for p in self.primitives]), dim=0)[0]

class Intersection(Primitive):
    def __init__(self, *primitives):
        self.primitives = primitives

    def __call__(self, x, y):
        return torch.max(torch.stack([p(x, y) for p in self.primitives]), dim=0)[0]

class Difference(Primitive):
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def __call__(self, x, y):
        return torch.max(self.p1(x, y), -self.p2(x, y))

def Rectangle(x1, y1, x2, y2):
    return Primitive(lambda x, y: torch.max(
        torch.max(x1 - x, x - x2),
        torch.max(y1 - y, y - y2)
    ))

class Circular(Primitive):
    def __init__(self, x_center, y_center, radius):
        super().__init__(lambda x, y: torch.sqrt((x - x_center)**2 + (y - y_center)**2)-radius)


# Define KAN Layer
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output



    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        L1 and the entropy loss is for the feature selection, i.e., let the weight of the activation function be small.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor, update_grid=False):
        for index, layer in enumerate(self.layers):
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
            if index < len(self.layers)-1: # 如果不是最后一层，拉到-1到1之间，最后一层不需要tanh
                x = torch.tanh(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )

#class PINN_MLP_unsteady(nn.Module):
#    def __init__(self, domain, layers, lb, ub, mu, rho):
#        super(PINN_MLP_unsteady, self).__init__()
#        self.domain = domain
#        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
#        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
#        self.mu = mu
#        self.rho = rho
#        
#        #MLP
##        self.net = nn.Sequential()
##        for i in range(len(layers) - 2):
##            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
##            self.net.add_module(f'tanh_{i}', nn.Tanh())
##        self.net.add_module(f'linear_out', nn.Linear(layers[-2], layers[-1]))
#
#        #KAN
#        #self.net = KAN(
#            #layers_hidden=layers,
#            #grid_size=5,
#            #spline_order=3,
#            #base_activation=torch.nn.Tanh,
#            #grid_range=[-1, 1]
#        #).to(device)
#
#    def forward(self, x):
#       # x[:, :2] = 2.0 * (x[:, :2] - self.lb) / (self.ub - self.lb) - 1.0 # Scale input
#        #x_norm = 2.0*(x - self.lb)/(self.up - self.lb) -1.0
#        # Add time dimensio:
#        #x = torch.cat([x, t], dim=1)
#        return self.net(x)
#
#    def net_uvp(self, x, y, t):
#        xyt = torch.cat([x, y, t], dim=1)
#        uvp = self.forward(xyt)
#        u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
#        return u, v, p
#
#    def net_f(self, x, y, t):
#        x.requires_grad_(True)
#        y.requires_grad_(True)
#        t.requires_grad_(True)
#        u, v, p = self.net_uvp(x, y, t)
#
#        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
#        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
#
#        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
#        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
#        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
#        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
#
#        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
#        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
#
#        # Time derivatives
#        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
#        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
#
#        # Unsteady Navier-Stokes Equations
#        f_u = self.rho * u_t + (u * u_x + v * u_y) + p_x - self.mu * (u_xx + u_yy)
#        f_v = self.rho * v_t + (u * v_x + v * v_y) + p_y - self.mu * (v_xx + v_yy)
#        f_e = u_x + v_y  # Continuity equation
#
#        return f_u, f_v, f_e

class DeepONet_unsteady(nn.Module):
    def __init__(self, domain, branch_layers, trunk_layers, lb, ub, mu, rho):
        super(DeepONet_unsteady, self).__init__()
        self.domain = domain
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.mu = mu
        self.rho = rho

        # Branch network (input: t)
        branch_net = []
        for i in range(len(branch_layers)-1):
            branch_net.append(nn.Linear(branch_layers[i], branch_layers[i+1]))
            if i < len(branch_layers)-2:
                branch_net.append(nn.Tanh())
        self.branch_net = nn.Sequential(*branch_net)

       # Trunk network (input: x,y)
        trunk_net = []
        for i in range(len(trunk_layers)-1):
            trunk_net.append(nn.Linear(trunk_layers[i], trunk_layers[i+1]))
            if i < len(trunk_layers)-2:
                trunk_net.append(nn.Tanh())
        self.trunk_net = nn.Sequential(*trunk_net)


        #Branch network (KAN 사용)
#        self.branch_net = KAN(
#            layers_hidden=branch_layers,
#            grid_size=5,
#            spline_order=3,
#            base_activation=torch.nn.Tanh,
#            grid_range=[-1, 1]
#        ).to(device)
#
#        # Trunk network (KAN 사용)
#        self.trunk_net = KAN(
#            layers_hidden=trunk_layers,
#            grid_size=5,
#            spline_order=3,
#            base_activation=torch.nn.Tanh,
#            grid_range=[-1, 1]
#        ).to(device)

        # Final output layers for u, v, p
        final_dim = branch_layers[-1]  # branch와 trunk의 출력 차원은 같아야 합니다.
        self.fc_u = nn.Linear(final_dim, 1)
        self.fc_v = nn.Linear(final_dim, 1)
        self.fc_p = nn.Linear(final_dim, 1)

    def forward(self, x, y, t):
        # Input normalization
        x_norm = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        y_norm = 2.0 * (y - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0
        #t_norm = 2.0 * (t - self.lb[2]) / (self.ub[2] - self.lb[2]) - 1.0

        # Branch and trunk outputs
        branch_out = self.branch_net(t)
        trunk_out = self.trunk_net(torch.cat([x_norm, y_norm], dim=1))

        # Element-wise multiplication (DeepONet)
        features = branch_out * trunk_out

        u = self.fc_u(features)
        v = self.fc_v(features)
        p = self.fc_p(features)

        return u, v, p

    def net_f(self, x, y, t):
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True)

        u, v, p = self.forward(x, y, t)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]

        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]

        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        f_u = self.rho * u_t + (u * u_x + v * u_y) + p_x - self.mu * (u_xx + u_yy)
        f_v = self.rho * v_t + (u * v_x + v * v_y) + p_y - self.mu * (v_xx + v_yy)
        f_e = u_x + v_y  # continuity

        return f_u, f_v, f_e

class DirichletBC:
    def __init__(self, boundary_func, value_func):
        self.boundary_func = boundary_func
        self.value_func = value_func

    def apply(self, x, y, u, v, p):
        mask = self.boundary_func(x, y)
        u_bc, v_bc, p_bc = self.value_func(x, y)

        if u_bc is not None:
            u = torch.where(mask, u_bc, u)
        if v_bc is not None:
            v = torch.where(mask, v_bc, v)
        if p_bc is not None:
            p = torch.where(mask, p_bc, p)

        return u, v, p

class NeumannBC:
    def __init__(self, boundary_func, gradient_func):
        """
        boundary_func: A function that returns a mask for boundary points.
        gradient_func: A function that returns a tuple of the gradient type and the target value.
                       The gradient type can be 'du/dx', 'du/dy', 'dv/dx', 'dv/dy', 'dp/dx', 'dp/dy'.
        """
        self.boundary_func = boundary_func
        self.gradient_func = gradient_func

    def apply(self, x, y, u, v, p):
        mask = self.boundary_func(x, y)
        grad_type, target_value = self.gradient_func(x, y)

        if grad_type == 'du/dx':
            x.requires_grad_(True)
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
            if u_x is not None:
                u_x = torch.where(mask, target_value, u_x)
        elif grad_type == 'du/dy':
            y.requires_grad_(True)
            u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True)[0]
            if u_y is not None:
                u_y = torch.where(mask, target_value, u_y)
        elif grad_type == 'dv/dx':
            x.requires_grad_(True)
            v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, allow_unused=True)[0]
            if v_x is not None:
                v_x = torch.where(mask, target_value, v_x)
        elif grad_type == 'dv/dy':
            y.requires_grad_(True)
            v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, allow_unused=True)[0]
            if v_y is not None:
                v_y = torch.where(mask, target_value, v_y)
        elif grad_type == 'dp/dx':
            x.requires_grad_(True)
            p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True, allow_unused=True)[0]
            if p_x is not None:
                p_x = torch.where(mask, target_value, p_x)
        elif grad_type == 'dp/dy':
            y.requires_grad_(True)
            p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True, allow_unused=True)[0]
            if p_y is not None:
                p_y = torch.where(mask, target_value, p_y)
        else:
            raise ValueError(f"Unsupported gradient type: {grad_type}")

        return u, v, p

def generate_boundary_points_random(domain, num_rect, num_cyl, lb, ub, cylinder_center, cylinder_radius):
    """
    Generates boundary points using random sampling on each boundary segment.
    
    Args:
        domain: A function that defines the domain (not directly used here).
        num_rect: Number of random points to sample for each rectangular boundary.
        num_cyl: Number of random points to sample on the cylinder surface.
        lb: Lower bounds of the domain [x_min, y_min].
        ub: Upper bounds of the domain [x_max, y_max].
        cylinder_center: Center of the cylinder [x_c, y_c].
        cylinder_radius: Radius of the cylinder.
        
    Returns:
        x_boundary: A tensor of x-coordinates of the boundary points.
        y_boundary: A tensor of y-coordinates of the boundary points.
    """
    # Left wall: x is fixed at lb[0], y is random in [lb[1], ub[1]]
    x_left = torch.full((num_rect, 1), lb[0], device=device)
    y_left = torch.empty((num_rect, 1), device=device).uniform_(lb[1], ub[1])
    
    # Right wall: x is fixed at ub[0]
    x_right = torch.full((num_rect, 1), ub[0], device=device)
    y_right = torch.empty((num_rect, 1), device=device).uniform_(lb[1], ub[1])
    
    # Top wall: y is fixed at ub[1], x is random in [lb[0], ub[0]]
    x_top = torch.empty((num_rect, 1), device=device).uniform_(lb[0], ub[0])
    y_top = torch.full((num_rect, 1), ub[1], device=device)
    
    # Bottom wall: y is fixed at lb[1]
    x_bottom = torch.empty((num_rect, 1), device=device).uniform_(lb[0], ub[0])
    y_bottom = torch.full((num_rect, 1), lb[1], device=device)
    
    # Cylinder surface: sample random angles uniformly in [0, 2π]
    angles = torch.empty((num_cyl,), device=device).uniform_(0, 2 * np.pi)
    x_cyl = (cylinder_center[0] + cylinder_radius * torch.cos(angles)).unsqueeze(1)
    y_cyl = (cylinder_center[1] + cylinder_radius * torch.sin(angles)).unsqueeze(1)
    
    # Combine all boundary points
    x_boundary = torch.cat([x_left, x_right, x_top, x_bottom, x_cyl], dim=0)
    y_boundary = torch.cat([y_left, y_right, y_top, y_bottom, y_cyl], dim=0)
    
    return x_boundary, y_boundary

def generate_interior_points_random(domain, num_points, lb, ub):
    """
    Generates interior points by randomly sampling the bounding box and filtering 
    with the given domain function.
    
    Args:
        domain: A function that takes x and y tensors and returns a tensor where 
                domain(x,y) <= 0 indicates points inside the domain.
        num_points: Desired number of interior points.
        lb: Lower bounds of the domain [x_min, y_min].
        ub: Upper bounds of the domain [x_max, y_max].
        
    Returns:
        x_interior: A tensor of x-coordinates of the interior points.
        y_interior: A tensor of y-coordinates of the interior points.
    """
    points = []
    # Rejection sampling: continue until enough interior points are collected.
    while len(points) < num_points:
        # Generate a batch (2*num_points is arbitrary to increase yield)
        batch_size = num_points * 2
        x_batch = torch.empty((batch_size, 1), device=device).uniform_(lb[0], ub[0])
        y_batch = torch.empty((batch_size, 1), device=device).uniform_(lb[1], ub[1])
        
        # Filter points based on the domain function
        mask = domain(x_batch, y_batch) <= 0
        x_valid = x_batch[mask]
        y_valid = y_batch[mask]
        
        for i in range(x_valid.shape[0]):
            points.append((x_valid[i].item(), y_valid[i].item()))
            if len(points) >= num_points:
                break

    # Convert list of points to tensors
    points_tensor = torch.tensor(points, device=device)
    x_interior = points_tensor[:, 0].unsqueeze(1)
    y_interior = points_tensor[:, 1].unsqueeze(1)
    return x_interior, y_interior



#def generate_interior_points_cartesian(domain, num_x, num_y, lb, ub):
#    x_vals = torch.linspace(lb[0], ub[0], num_x, device = device)
#    y_vals = torch.linspace(lb[1], ub[1], num_y, device = device)
#
#    x_grid, y_grid = torch.meshgrid(x_vals, y_vals, indexing = 'ij')
#
#    x = x_grid.reshape(-1, 1)
#    y = y_grid.reshape(-1, 1)
#
#    mask = domain(x, y) <= 0
#    x = x[mask.squeeze()]
#    y = y[mask.squeeze()]
#
#    return x, y
#
#def generate_boundary_points_cartesian(domain, num_x, num_y, lb, ub, cylinder_center, cylinder_radius):
#    """
#    Generates boundary points using a Cartesian grid for each boundary segment.
#
#    Args:
#        domain: A function that defines the domain (not directly used here but could be used for filtering).
#        num_x: Number of points in the x-direction for horizontal boundaries.
#        num_y: Number of points in the y-direction for vertical boundaries.
#        lb: Lower bounds of the domain [x_min, y_min].
#        ub: Upper bounds of the domain [x_max, y_max].
#        cylinder_center: Center of the cylinder [x_c, y_c].
#        cylinder_radius: Radius of the cylinder.
#
#    Returns:
#        x_boundary: A tensor of x-coordinates of the boundary points.
#        y_boundary: A tensor of y-coordinates of the boundary points.
#    """
#    # Left wall
#    x_left = torch.full((num_y, 1), lb[0], device=device)
#    y_left = torch.linspace(lb[1], ub[1], num_y, device=device).reshape(-1, 1)
#
#    # Right wall
#    x_right = torch.full((num_y, 1), ub[0], device=device)
#    y_right = torch.linspace(lb[1], ub[1], num_y, device=device).reshape(-1, 1)
#
#    # Upper wall
#    x_top = torch.linspace(lb[0], ub[0], num_x, device=device).reshape(-1, 1)
#    y_top = torch.full((num_x, 1), ub[1], device=device)
#
#    # Bottom wall
#    x_bottom = torch.linspace(lb[0], ub[0], num_x, device=device).reshape(-1, 1)
#    y_bottom = torch.full((num_x, 1), lb[1], device=device)
#
#    # Cylinder Surface (using polar coordinates)
#    num_cyl = int(2 * np.pi * cylinder_radius * (num_x / (ub[0] - lb[0])))  # Estimate based on density
#    angles = torch.linspace(0, 2 * np.pi, num_cyl, device=device)
#    x_cyl = (cylinder_center[0] + cylinder_radius * torch.cos(angles)).reshape(-1, 1)
#    y_cyl = (cylinder_center[1] + cylinder_radius * torch.sin(angles)).reshape(-1, 1)
#
#    # Combine all boundary points
#    x_boundary = torch.cat([x_left, x_right, x_top, x_bottom, x_cyl], dim=0)
#    y_boundary = torch.cat([y_left, y_right, y_top, y_bottom, y_cyl], dim=0)
#
#    return x_boundary, y_boundary

def plot_data_distribution(domain, x_interior, y_interior, x_boundary, y_boundary, lb, ub):
    plt.figure(figsize=(12, 6))

    # Domain shape
    nx, ny = 500, 200
    x = np.linspace(lb[0], ub[0], nx)
    y = np.linspace(lb[1], ub[1], ny)
    X, Y = np.meshgrid(x, y)
    X_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    Z = domain(X_flat, Y_flat).cpu().numpy().reshape(X.shape)

    # Interior points
    plt.scatter(x_interior.cpu(), y_interior.cpu(), c='lightblue', s=1, alpha=0.5, label='Interior points')
    plt.scatter(x_boundary.cpu(), y_boundary.cpu(), c='red', s=1, alpha=0.5, label='Boundary points')

    plt.title('Distribution of Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('distribution_plot.jpg')
    #plt.show()


#def plot_combined_loss(losses_adam, losses_lbfgs, n_epochs_adam, n_epochs_lbfgs):
    #plt.figure(figsize=(12, 8))
#
    ## Adam loss
    #iterations_adam = np.arange(len(losses_adam['total']))
    #plt.semilogy(iterations_adam, losses_adam['total'], label='Total Loss (Adam)', color='blue')
    ##plt.semilogy(iterations_adam, losses_adam['pde'], label='PDE Loss (Adam)', color='cyan')
    ##plt.semilogy(iterations_adam, losses_adam['bc'], label='BC Loss (Adam)', color='navy')
#
    ## L-BFGS loss
    #iterations_lbfgs = np.arange(len(losses_lbfgs['total'])) + len(losses_adam['total'])
    #plt.semilogy(iterations_lbfgs, losses_lbfgs['total'], label='Total Loss (L-BFGS)', color='red')
    #plt.semilogy(iterations_lbfgs, losses_lbfgs['pde'], label='PDE Loss (L-BFGS)', color='orange')
    #plt.semilogy(iterations_lbfgs, losses_lbfgs['bc'], label='BC Loss (L-BFGS)', color='darkred')
#
    #plt.xlim([0, n_epochs_adam+n_epochs_lbfgs])
    #plt.xlabel('Iterations')
    #plt.ylabel('Loss')
    #plt.title('Combined Adam and L-BFGS Optimization Losses')
    #plt.legend()
    #plt.grid(True)
    #plt.tight_layout()
    #plt.savefig('loss_plot.jpg')
    ##plt.show()

def plot_combined_loss(losses_mlp, losses_lbfgs, n_epochs_adam):
    plt.figure(figsize=(12, 8))

    # Combine the losses for MLP-Adam and MLP-LBFGS
    iterations = np.arange(len(losses_mlp['total']) + len(losses_lbfgs['total']))
    total_losses = np.concatenate([losses_mlp['total'], losses_lbfgs['total']])
    pde_losses = np.concatenate([losses_mlp['pde'], losses_lbfgs['pde']])
    bc_losses = np.concatenate([losses_mlp['bc'], losses_lbfgs['bc']])
    ic_losses = np.concatenate([losses_mlp['ic'], losses_lbfgs['ic']])

    # Plot the combined losses
    plt.semilogy(iterations, total_losses, label='Total Loss (MLP)', color='red')
    plt.semilogy(iterations, pde_losses, label='PDE Loss (MLP)', color='orange')
    plt.semilogy(iterations, bc_losses, label='BC Loss (MLP)', color='darkred')
    plt.semilogy(iterations, ic_losses, label='IC Loss (MLP)', color='purple')

    # Add vertical line to indicate where L-BFGS starts
    plt.axvline(x=n_epochs_adam, color='black', linestyle='--', label='Adam/L-BFGS Switch')

    plt.xlim([0, len(iterations)])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('MLP PINN Losses (Adam + L-BFGS)')

    # Combine legend entries for Adam and L-BFGS
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles)) # Remove duplicate labels

    # Manually set the legend order
    legend_order = ['Total Loss (MLP)', 'PDE Loss (MLP)', 'BC Loss (MLP)', 'IC Loss (MLP)', 'Adam/L-BFGS Switch']
    ordered_handles = [by_label[k] for k in legend_order]

    plt.legend(ordered_handles, legend_order, loc='upper right')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparison_loss_plot_KINN.jpg')
    #plt.show()

def train_pinn(
    model,
    optimizer,
    x_interior,
    y_interior,
    x_boundary,
    y_boundary,
    time_values,
    lb,
    ub,
    n_epochs,
    bc_weights,
    inlet_velocity,  # 새로 추가된 파라미터
    batch_size=None,
    model_type='MLP'
):
    losses = {
        'total': [],
        'pde': [],
        'bc': [],
        'inlet_bc': [],
        'top_bc': [],
        'bottom_bc': [],
        'outlet_p_bc': [],
        'outlet_du_dy_bc': [],
        'wall_bc': [],
        'ic': []
    }
    eps = 5e-4

    # 개별 경계 조건 정의
    t_end = time_values[-1]

    # 입구 경계 조건 (Dirichlet: u=U, v=0)
    inlet_bc = DirichletBC(
        lambda x, y: (x <= lb[0] + eps),
        lambda x, y: ((6/ub[1]) * inlet_velocity * (y/ub[1] - (y/ub[1])**2), torch.zeros_like(y), None)
    )

    # 상단 벽 경계 조건 (Dirichlet: u=U, v=0)
    top_bc = DirichletBC(
        lambda x, y: (y >= ub[1] - eps),
        lambda x, y: (torch.zeros_like(x), torch.zeros_like(y), None)
    )

    # 하단 벽 경계 조건 (Dirichlet: u=U, v=0)
    bottom_bc = DirichletBC(
        lambda x, y: (y <= lb[1] + eps),
        lambda x, y: (torch.zeros_like(x), torch.zeros_like(y), None)
    )

    # 출구 압력 경계 조건 (Dirichlet: p=0)
    outlet_p_bc = DirichletBC(
        lambda x, y: (x >= ub[0] - eps),
        lambda x, y: (None, None, torch.zeros_like(x))
    )

    # 출구 속도 기울기 경계 조건 (Neumann: du/dy=0)
    outlet_du_dy_bc = NeumannBC(
        lambda x, y: (x >= ub[0] - eps),
        lambda x, y: ('du/dy', torch.zeros_like(x))
    )

     # 실린더에 대한 노-슬립 조건 (Dirichlet: u=0, v=0)
    wall_bc = DirichletBC(
        lambda x, y: ((torch.sqrt((x - cylinder_center[0])**2 + (y - cylinder_center[1])**2) <= cylinder_radius + eps)),
        lambda x, y: (torch.zeros_like(x), torch.zeros_like(y), None)
    )

    if batch_size is None:
        batch_size = len(x_interior)


    def closure():
        optimizer.zero_grad()

        # Sample time points randomly from the entire time domain [0, t_end]
        t_batch = torch.rand(batch_size, 1, device=device) * t_end

        # Sample spatial points (same as before)
        idx = np.random.choice(len(x_interior), batch_size, replace=False)
        x_batch = x_interior[idx]
        y_batch = y_interior[idx]

        t_bnd = torch.rand(len(x_boundary), 1, device=device) * t_end
        # Calculate PDE loss
        f_u, f_v, f_e = model.net_f(x_batch, y_batch, t_batch)
        loss_pde = torch.mean(f_u**2 + f_v**2 + f_e**2)

        # 입구 경계 조건
        u_pred, v_pred, p_pred = model(x_boundary, y_boundary, t_bnd)
        u_inlet, v_inlet, p_inlet = inlet_bc.apply(x_boundary, y_boundary, u_pred.clone(), v_pred.clone(), p_pred.clone())
        loss_inlet_bc = torch.mean((u_inlet - u_pred)**2) + torch.mean((v_inlet - v_pred)**2)

        # 상단 벽 경계 조건
        u_top, v_top, p_top = top_bc.apply(x_boundary, y_boundary, u_pred, v_pred, p_pred)
        loss_top_bc = torch.mean((u_top - u_pred)**2) + torch.mean((v_top - v_pred)**2)

        # 하단 벽 경계 조건
        u_bottom, v_bottom, p_bottom = bottom_bc.apply(x_boundary, y_boundary, u_pred, v_pred, p_pred)
        loss_bottom_bc = torch.mean((u_bottom - u_pred)**2) + torch.mean((v_bottom - v_pred)**2)

        # 출구 압력 경계 조건
        _, _, p_outlet = outlet_p_bc.apply(x_boundary, y_boundary, u_pred, v_pred, p_pred)
        loss_outlet_p_bc = torch.mean((p_outlet - p_pred)**2)

        # 출구 속도 기울기 경계 조건
        u_b_outlet_dy, v_b_outlet_dy, _ = outlet_du_dy_bc.apply(x_boundary, y_boundary, u_pred, v_pred, p_pred)
        loss_outlet_du_dy_bc = (torch.mean((u_b_outlet_dy - u_pred)**2) +
                                torch.mean((v_b_outlet_dy - v_pred)**2))

        # 실린더 노-슬립 경계 조건 손실
        u_wall, v_wall, p_wall = wall_bc.apply(x_boundary, y_boundary, u_pred.clone(), v_pred.clone(), p_pred.clone())
        loss_wall_bc = torch.mean((u_wall - u_pred)**2) + torch.mean((v_wall - v_pred)**2)


        # 총 경계 손실
        loss_bc = loss_inlet_bc + loss_top_bc + loss_bottom_bc + loss_outlet_p_bc + loss_outlet_du_dy_bc + loss_wall_bc

        # Initial condition loss:
        t_initial = torch.zeros(batch_size, 1, device=device)  # t=0 for initial condition
        u_initial, v_initial, p_initial = model(x_batch, y_batch, t_initial)

        # Assuming you have some way to define the true initial condition:
        u_initial_true = torch.zeros_like(u_initial)#inlet_velocity * torch.ones_like(u_initial)  # Example
        v_initial_true = torch.zeros_like(v_initial)
        p_initial_true = torch.zeros_like(p_initial)

        loss_ic = torch.mean((u_initial - u_initial_true)**2) + \
                  torch.mean((v_initial - v_initial_true)**2) + \
                  torch.mean((p_initial - p_initial_true)**2)

        # 총 손실
        loss = loss_pde + \
               bc_weights.get('inlet', 1.0) * loss_inlet_bc + \
               bc_weights.get('top', 1.0) * loss_top_bc + \
               bc_weights.get('bottom', 1.0) * loss_bottom_bc + \
               bc_weights.get('outlet_p', 1.0) * loss_outlet_p_bc + \
               bc_weights.get('outlet_du_dy', 1.0) * loss_outlet_du_dy_bc + \
               bc_weights.get('wall', 1.0) * loss_wall_bc + \
               loss_ic

        loss.backward()

        # 손실 기록 업데이트
        losses['total'].append(loss.item())
        losses['pde'].append(loss_pde.item())
        losses['bc'].append(loss_bc.item())
        losses['inlet_bc'].append(loss_inlet_bc.item())
        losses['top_bc'].append(loss_top_bc.item())
        losses['bottom_bc'].append(loss_bottom_bc.item())
        losses['outlet_p_bc'].append(loss_outlet_p_bc.item())
        losses['outlet_du_dy_bc'].append(loss_outlet_du_dy_bc.item())
        losses['wall_bc'].append(loss_wall_bc.item())
        losses['ic'].append(loss_ic.item())

        return loss

    # Training loop
    start_time = time.time()
    for epoch in range(n_epochs):
        loss = optimizer.step(closure)

        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch}, Total Loss: {losses["total"][-1]:.4e}, PDE Loss: {losses["pde"][-1]:.4e}, BC Loss: {losses["bc"][-1]:.4e}, Time: {elapsed:.2f}s')
            start_time = time.time()

    return losses


# ... (Rest of the code remains the same, including classes, functions, and training) ...
def create_unsteady_visualization_data(domain, model_mlp, num_x, num_y, lb, ub, time_values):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    nx, ny = num_x, num_y
    x = np.linspace(lb[0], ub[0], nx)
    y = np.linspace(lb[1], ub[1], ny)
    X, Y = np.meshgrid(x, y)

    # Generate u_data, v_data, and p_data for each time step
    u_data = []
    v_data = []
    p_data = []
    for t in time_values:
        x_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        t_flat = torch.full_like(x_flat, t)
        # 모델 예측
        u, v, p = model_mlp(x_flat, y_flat, t_flat)
        # 2D 형태로 reshape 후 리스트에 저장
        u_data.append(u.cpu().detach().numpy().reshape(X.shape))
        v_data.append(v.cpu().detach().numpy().reshape(X.shape))
        p_data.append(p.cpu().detach().numpy().reshape(X.shape))

    # ---------- 초기 프레임(0번째)에 대한 imshow 생성 ----------
    im1 = axs[0].imshow(u_data[0],
                        extent=[lb[0], ub[0], lb[1], ub[1]],
                        origin='lower',
                        cmap='jet', vmax = 1.2, vmin =0)
    fig.colorbar(im1, ax=axs[0], orientation='horizontal')

    im2 = axs[1].imshow(v_data[0],
                        extent=[lb[0], ub[0], lb[1], ub[1]],
                        origin='lower',
                        cmap='jet', vmax = 0.7, vmin = -0.7)
    fig.colorbar(im2, ax=axs[1], orientation='horizontal')

    im3 = axs[2].imshow(p_data[0],
                        extent=[lb[0], ub[0], lb[1], ub[1]],
                        origin='lower',
                        cmap='jet', vmax = 3, vmin = 0)
    fig.colorbar(im3, ax=axs[2], orientation='horizontal')

    # 화면 비율 맞추기
    for ax in axs:
        ax.set_aspect('equal', 'box')

    # 실린더(원) 표시
    c0 = plt.Circle(cylinder_center, cylinder_radius, color='black', zorder=10)
    c1 = plt.Circle(cylinder_center, cylinder_radius, color='black', zorder=10)
    c2 = plt.Circle(cylinder_center, cylinder_radius, color='black', zorder=10)
    axs[0].add_patch(c0)
    axs[1].add_patch(c1)
    axs[2].add_patch(c2)

    # ---------- 애니메이션에 사용할 함수 정의 ----------
    def animate(i):
        # imshow 객체에 새로운 데이터 세팅
        im1.set_data(u_data[i])
        im2.set_data(v_data[i])
        im3.set_data(p_data[i])

        # 필요하다면 색 범위도 업데이트 가능
        # 예) im1.set_clim(vmin=-1.0, vmax=1.0)

        axs[0].set_title(f'u velocity (t={time_values[i]:.2f})')
        axs[1].set_title(f'v velocity (t={time_values[i]:.2f})')
        axs[2].set_title(f'pressure (t={time_values[i]:.2f})')

        # 원(실린더)도 다시 그려줘야 한다면 patch를 새로 업데이트하거나,
        # set_visible(True/False) 등의 방식을 사용할 수도 있음

        # 업데이트된 imshow 객체들을 반환
        return im1, im2, im3

    # ---------- 애니메이션 만들기 ----------
    ani = animation.FuncAnimation(
        fig, animate,
        frames=len(time_values),
        interval=200,
        blit=False
    )

    ani.save('unsteady_flow_DeepONet.gif', writer='pillow', fps=20)

    # plt.show()
    return X, Y, u_data, v_data, p_data

if __name__ == "__main__":
    # ----------------------------
    # 1.HyperParameters
    # ----------------------------
    desired_Re = 20.0
    rho = 1  # kg/m³
    mu = 0.005     # Pa·s

    cylinder_radius = 0.05  # 미터
    D = 2 * cylinder_radius  # 직경 = 0.1 m

    nu = mu / rho  # 운동 점도 m²/s
    U = (desired_Re * nu) / D  # m/s
    print(f"계산된 입구 속도 U = {U:.6f} m/s, Reynolds 수 = {desired_Re}")

    n_epochs_adam = 20000
    #n_epochs_lbgfs =20000
    n_epochs_lbgfs = 20000
    batch_size_adam = 100 

    # ----------------------------
    # 2. 도메인 및 장애물 정의
    # ----------------------------
    lb = np.array([0, 0])     # 도메인 하한 (x_min, y_min)
    ub = np.array([1.1, 0.41])    # 도메인 상한 (x_max, y_max)
    #ub = np.array([1.1, 0.81])    # 도메인 상한 (x_max, y_max)
    lb_2 = np.array([0.1, 0.1])    # subdomain, srround circular cylinder
    ub_2 = np.array([0.31, 0.31])
    lb_3 = np.array([0.3, 0.1])     # subdomain. after circular cylinder
    ub_3 = np.array([0.9, 0.31])

    cylinder_center = np.array([0.2, 0.2])  # 원의 중심

    channel = Rectangle(lb[0], lb[1], ub[0], ub[1])
    cylinder = Circular(cylinder_center[0], cylinder_center[1], cylinder_radius)

    domain = Difference(channel, cylinder)

    # ----------------------------
    # 3. 훈련 데이터 생성
    # ----------------------------

    #x_boundary, y_boundary = generate_boundary_points_cartesian(
    #    domain, num_x_bnd, num_y_bnd, lb, ub, cylinder_center, cylinder_radius
    #)
    #num_x_int = 100 
    #num_y_int = 50
    #x_interior, y_interior = generate_interior_points_cartesian(domain, num_x_int, num_y_int, lb, ub)
    #x_boundary, y_boundary = generate_boundary_points_cartesian(domain,8* num_x_int, 4* num_y_int, lb, ub, cylinder_center, cylinder_radius)
    num_rect = 6000
    num_cyl = 3000
    
    N_domain = 30000
    N_domain_sub1 = 8000 
    N_domain_sub2 = 8000


    x_interior_1, y_interior_1 = generate_interior_points_random(domain, N_domain, lb, ub)
    x_interior_2, y_interior_2 = generate_interior_points_random(domain, N_domain_sub1, lb_2, ub_2)
    x_interior_3, y_interior_3 = generate_interior_points_random(domain, N_domain_sub2, lb_3, ub_3)

    # Integrate data points
    x_interior = torch.cat([x_interior_1, x_interior_2, x_interior_3], dim=0)
    y_interior = torch.cat([y_interior_1, y_interior_2, y_interior_3], dim=0)
    #x_interior, y_interior = generate_interior_points_random(domain, num_int, lb, ub)
    
    x_boundary, y_boundary = generate_boundary_points_random(domain, num_rect, num_cyl, lb, ub, cylinder_center, cylinder_radius)

    # Combine all boundary points
    x_boundary = torch.cat([x_interior, x_boundary], dim=0)
    y_boundary = torch.cat([y_interior, y_boundary], dim=0)

    time_values = torch.linspace(0, 0.5, 500, device=device)
    # 데이터 분포 시각화
    plot_data_distribution(domain, x_interior, y_interior, x_boundary, y_boundary, lb, ub)

    # ----------------------------
    # 4. PINN 모델 초기화
    # ----------------------------

    trunk_layers = [2, 200, 200, 200, 100]
    branch_layers = [1, 200, 200, 200, 200,100]
    #model_mlp = PINN_MLP_unsteady(domain, layers_mlp, lb, ub, mu=mu, rho=rho).to(device)
    model_mlp = DeepONet_unsteady(domain, branch_layers, trunk_layers, lb, ub, mu=mu, rho=rho).to(device)

    # ----------------------------
    # 5. 손실 가중치 정의
    # ----------------------------
    bc_weights = {'inlet': 1.0, 'top': 1.0, 'bottom': 1.0, 'outlet_p': 1.0, 'outlet_du_dy': 0, 'wall': 4.0}
    learning_rate = 1e-4 
    # ----------------------------
    # 6. 옵티마이저 정의
    # ----------------------------
    optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=learning_rate)
    optimizer_lbfgs = optim.LBFGS(model_mlp.parameters(), lr=learning_rate, max_iter=50000, history_size=100, line_search_fn='strong_wolfe')

    # ----------------------------
    # 9. MLP 기반 PINN 훈련
    # ----------------------------
    print("MLP 기반 PINN 훈련 시작...")
    losses_mlp = train_pinn(
        model=model_mlp,
        optimizer=optimizer_mlp,
        x_interior=x_interior,
        y_interior=y_interior,
        x_boundary=x_boundary,
        y_boundary=y_boundary,
        time_values=time_values,
        lb=lb,
        ub=ub,
        n_epochs=n_epochs_adam,
        bc_weights=bc_weights,
        inlet_velocity=U,  # 입구 속도 전달
        batch_size=batch_size_adam,
        model_type='MLP'
    )

    time_values_animation = torch.linspace(0, 0.5, 100, device=device)
    #X, Y, u_data, v_data, p_data = create_unsteady_visualization_data(domain, model_mlp, 100, 50, lb, ub, time_values_animation)

    losses_lbgfs = train_pinn(
        model=model_mlp,
        optimizer=optimizer_lbfgs,
        x_interior=x_interior,
        y_interior=y_interior,
        x_boundary=x_boundary,
        y_boundary=y_boundary,
        time_values=time_values,
        lb=lb,
        ub=ub,
        n_epochs=n_epochs_lbgfs,
        bc_weights=bc_weights,
        inlet_velocity=U,  # 입구 속도 전달
        batch_size=batch_size_adam,
        model_type='MLP'
    )


    # ----------------------------
    # 10. 손실 그래프 시각화
    # ----------------------------
    plot_combined_loss(losses_mlp, losses_lbgfs, n_epochs_adam)

    # ----------------------------
    # 11. 시각화 데이터 생성
    # ----------------------------
    
    X, Y, u_data, v_data, p_data = create_unsteady_visualization_data(domain, model_mlp, 100, 50, lb, ub, time_values_animation)
