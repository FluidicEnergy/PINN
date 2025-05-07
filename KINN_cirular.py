import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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

class PINN_KAN(nn.Module):
    def __init__(self, domain, layers, lb, ub, mu, rho):
        super(PINN_KAN, self).__init__()
        self.domain = domain
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.mu = mu
        self.rho = rho

        self.net = KAN(
            layers_hidden=layers,
            grid_size=5,
            spline_order=3,
            base_activation=torch.nn.Tanh,
            grid_range=[-1, 1]
        ).to(device)

    def forward(self, x):
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0  # Scale input
        return self.net(x)

    def net_uvp(self, x, y):
        xy = torch.cat([x, y], dim=1)
        uvp = self.forward(xy)
        u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
        return u, v, p

    def net_f(self, x, y):
        x.requires_grad_(True)
        y.requires_grad_(True)

        u, v, p = self.net_uvp(x, y)

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

        # Navier-Stokes Equations
        f_u = (u * u_x + v * u_y) + p_x /self.rho - (self.mu / self.rho) * (u_xx + u_yy)
        f_v = (u * v_x + v * v_y) + p_y /self.rho - (self.mu /self.rho) * (v_xx + v_yy)
        f_e = u_x + v_y  # Continuity equation

        return f_u, f_v, f_e



class PINN_MLP(nn.Module):
    def __init__(self, domain, layers, lb, ub, mu, rho):
        super(PINN_MLP, self).__init__()
        self.domain = domain
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.mu = mu
        self.rho = rho

        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f'tanh_{i}', nn.Tanh())
        self.net.add_module(f'linear_out', nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0 # Scale input
        return self.net(x)

    def net_uvp(self, x, y):
        xy = torch.cat([x, y], dim=1)
        uvp = self.forward(xy)
        u, v, p = uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]
        return u, v, p

    def net_f(self, x, y):
        x.requires_grad_(True)
        y.requires_grad_(True)

        u, v, p = self.net_uvp(x, y)

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

        # Navier-Stokes Equations
        f_u = (u * u_x + v * u_y) + p_x /self.rho - (self.mu / self.rho) * (u_xx + u_yy)
        f_v = (u * v_x + v * v_y) + p_y /self.rho - (self.mu /self.rho) * (v_xx + v_yy)
        f_e = u_x + v_y  # Continuity equation

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

def generate_interior_points(domain, N_points, lb, ub):
    x = torch.rand(N_points, 1, device=device) * (ub[0] - lb[0]) + lb[0]
    y = torch.rand(N_points, 1, device=device) * (ub[1] - lb[1]) + lb[1]

    mask = domain(x, y) <= 0
    x = x[mask.squeeze()]
    y = y[mask.squeeze()]

    return x, y

def generate_boundary_points(domain, N_points, lb, ub, cylinder_center, cylinder_radius):
    eps = 1e-5

    N_each = N_points // 4  # 4 boundaries: left, right, top, bottom

    # Left wall
    x_left = torch.full((N_each, 1), lb[0], device=device)
    y_left = torch.rand(N_each, 1, device=device) * (ub[1] - lb[1]) + lb[1]

    # Right wall
    x_right = torch.full((N_each, 1), ub[0], device=device)
    y_right = torch.rand(N_each, 1, device=device) * (ub[1] - lb[1]) + lb[1]

    # Upper wall
    x_top = torch.rand(N_each, 1, device=device) * (ub[0] - lb[0]) + lb[0]
    y_top = torch.full((N_each, 1), ub[1], device=device)

    # Bottom wall
    x_bottom = torch.rand(N_each, 1, device=device) * (ub[0] - lb[0]) + lb[0]
    y_bottom = torch.full((N_each, 1), lb[1], device = device)

    # Cylinder Surface
    angles = torch.rand(N_each//4, 1 ,device = device) * 2 *np.pi
    x_cyl = cylinder_center[0] + cylinder_radius* torch.cos(angles)
    y_cyl = cylinder_center[1] + cylinder_radius * torch.sin(angles)

    # Combine all boundary points
    x_boundary = torch.cat([x_left, x_right, x_top, x_bottom, x_cyl], dim = 0)
    y_boundary = torch.cat([y_left, y_right, y_top, y_bottom, y_cyl])

    return x_boundary, y_boundary

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

    # Boundary points
    plt.scatter(x_boundary.cpu(), y_boundary.cpu(), c='red', s=5, label='Boundary points')

    plt.title('Distribution of Data Points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('distribution_plot.jpg')
    #plt.show()

def create_masked_visualization_data(domain, nx, ny, lb, ub):
    x = np.linspace(lb[0], ub[0], nx)
    y = np.linspace(lb[1], ub[1], ny)
    X, Y = np.meshgrid(x, y)

    X_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

    mask = domain(X_flat, Y_flat).cpu().numpy().reshape(X.shape) < 0

    return X, Y, mask

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

def plot_combined_loss(losses_kan, losses_mlp, n_epochs):
    plt.figure(figsize=(12, 8))

    # KAN-based PINN losses
    iterations_kan = np.arange(len(losses_kan['total']))
    plt.semilogy(iterations_kan, losses_kan['total'], label='Total Loss (KAN)', color='blue')
    #plt.semilogy(iterations_kan, losses_kan['pde'], label='PDE Loss (KAN)', color='cyan')
    #plt.semilogy(iterations_kan, losses_kan['bc'], label='BC Loss (KAN)', color='navy')

    # MLP-based PINN losses
    iterations_mlp = np.arange(len(losses_mlp['total']))
    plt.semilogy(iterations_mlp, losses_mlp['total'], label='Total Loss (MLP)', color='red')
    #plt.semilogy(iterations_mlp, losses_mlp['pde'], label='PDE Loss (MLP)', color='orange')
    #plt.semilogy(iterations_mlp, losses_mlp['bc'], label='BC Loss (MLP)', color='darkred')

    plt.xlim([0, n_epochs])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Comparison of KAN and MLP PINN Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('comparison_loss_plot.jpg')
    #plt.show()

def plot_comparison_results(model_kan, model_mlp, X, Y, mask):
    with torch.no_grad():
        # KAN-based PINN predictions
        x = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        y = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        u_kan, v_kan, p_kan = model_kan.net_uvp(x, y)
        u_kan = u_kan.cpu().numpy().reshape(X.shape)
        v_kan = v_kan.cpu().numpy().reshape(X.shape)
        p_kan = p_kan.cpu().numpy().reshape(X.shape)

        # MLP-based PINN predictions
        u_mlp, v_mlp, p_mlp = model_mlp.net_uvp(x, y)
        u_mlp = u_mlp.cpu().numpy().reshape(X.shape)
        v_mlp = v_mlp.cpu().numpy().reshape(X.shape)
        p_mlp = p_mlp.cpu().numpy().reshape(X.shape)

    # Masking
    u_kan = np.ma.masked_array(u_kan, ~mask)
    v_kan = np.ma.masked_array(v_kan, ~mask)
    p_kan = np.ma.masked_array(p_kan, ~mask)

    u_mlp = np.ma.masked_array(u_mlp, ~mask)
    v_mlp = np.ma.masked_array(v_mlp, ~mask)
    p_mlp = np.ma.masked_array(p_mlp, ~mask)

    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))

    # KAN-based PINN
    im1 = axs[0, 0].imshow(u_kan, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='jet')
    axs[0, 0].set_title('u velocity (KAN)')
    fig.colorbar(im1, ax=axs[0, 0])

    im2 = axs[0, 1].imshow(v_kan, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='jet')
    axs[0, 1].set_title('v velocity (KAN)')
    fig.colorbar(im2, ax=axs[0, 1])

    im3 = axs[0, 2].imshow(p_kan, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='jet')
    axs[0, 2].set_title('pressure (KAN)')
    fig.colorbar(im3, ax=axs[0, 2])

    # MLP-based PINN
    im4 = axs[1, 0].imshow(u_mlp, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='jet')
    axs[1, 0].set_title('u velocity (MLP)')
    fig.colorbar(im4, ax=axs[1, 0])

    im5 = axs[1, 1].imshow(v_mlp, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='jet')
    axs[1, 1].set_title('v velocity (MLP)')
    fig.colorbar(im5, ax=axs[1, 1])

    im6 = axs[1, 2].imshow(p_mlp, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='jet')
    axs[1, 2].set_title('pressure (MLP)')
    fig.colorbar(im6, ax=axs[1, 2])

    plt.tight_layout()
    plt.savefig('comparison_result_plot.jpg')
    #plt.show()



def train_pinn(
    model,
    optimizer,
    x_interior,
    y_interior,
    x_boundary,
    y_boundary,
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
        'wall_bc': []
    }
    eps = 1e-5

    # 개별 경계 조건 정의

    # 입구 경계 조건 (Dirichlet: u=U, v=0)
    inlet_bc = DirichletBC(
        lambda x, y: (x == lb[0]),
        lambda x, y: (inlet_velocity * torch.ones_like(x), torch.zeros_like(y), None)
    )

    # 상단 벽 경계 조건 (Dirichlet: u=U, v=0)
    top_bc = DirichletBC(
        lambda x, y: (y == ub[1]),
        lambda x, y: (inlet_velocity * torch.ones_like(x), torch.zeros_like(y), None)
    )

    # 하단 벽 경계 조건 (Dirichlet: u=U, v=0)
    bottom_bc = DirichletBC(
        lambda x, y: (y == lb[1]),
        lambda x, y: (inlet_velocity * torch.ones_like(x), torch.zeros_like(y), None)
    )

    # 출구 압력 경계 조건 (Dirichlet: p=0)
    outlet_p_bc = DirichletBC(
        lambda x, y: (x == ub[0]),
        lambda x, y: (None, None, torch.zeros_like(x))
    )

    # 출구 속도 기울기 경계 조건 (Neumann: du/dy=0)
    outlet_du_dy_bc = NeumannBC(
        lambda x, y: (x == ub[0]),
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

        # 미니 배치 샘플링
        idx = np.random.choice(len(x_interior), batch_size, replace=False)
        x_batch = x_interior[idx]
        y_batch = y_interior[idx]

        # PDE 손실 계산
        f_u, f_v, f_e = model.net_f(x_batch, y_batch)
        loss_pde = torch.mean(f_u**2 + f_v**2 + f_e**2)

        # 경계 조건 손실 계산

        # 입구 경계 조건
        u_pred, v_pred, p_pred = model.net_uvp(x_boundary, y_boundary)
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

        # 총 손실
        loss = loss_pde + \
               bc_weights.get('inlet', 1.0) * loss_inlet_bc + \
               bc_weights.get('top', 1.0) * loss_top_bc + \
               bc_weights.get('bottom', 1.0) * loss_bottom_bc + \
               bc_weights.get('outlet_p', 1.0) * loss_outlet_p_bc + \
               bc_weights.get('outlet_du_dy', 1.0) * loss_outlet_du_dy_bc + \
               bc_weights.get('wall', 1.0) * loss_wall_bc

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


if __name__ == "__main__":
    # ----------------------------
    # 1.HyperParameters
    # ----------------------------
    desired_Re = 100.0
    rho = 100.0  # kg/m³
    mu = 0.01     # Pa·s

    cylinder_radius = 0.05  # 미터
    D = 2 * cylinder_radius  # 직경 = 0.1 m

    nu = mu / rho  # 운동 점도 m²/s
    U = (desired_Re * nu) / D  # m/s
    print(f"계산된 입구 속도 U = {U:.6f} m/s, Reynolds 수 = {desired_Re}")

    N = 10
    L = 0.1/N
    # ----------------------------
    # 2. 도메인 및 장애물 정의
    # ----------------------------
    lb = np.array([0.0, 0.0])     # 도메인 하한 (x_min, y_min)
    ub = np.array([2.2, 0.41 + L])    # 도메인 상한 (x_max, y_max)

    lb_2 = np.array([0.25, 0.1])    # subdomain, srround circular cylinder 
    ub_2 = np.array([0.76, 0.31 + L])   

    lb_3 = np.array([0.75, 0.1])     # subdomain. after circular cylinder 
    ub_3 = np.array([1.76, 0.31 + L])   
    
    cylinder_center = np.array([2.2, 0.41 + L/2])  # 원의 중심
    cylinder_radius = 0.05

    channel = Rectangle(lb[0], lb[1], ub[0], ub[1])
    cylinder = Circular(cylinder_center[0], cylinder_center[1], cylinder_radius)

    domain = Difference(channel, cylinder)

    # ----------------------------
    # 3. 훈련 데이터 생성
    # ----------------------------
    N_domain = 20000
    N_domain_sub1 = 20000
    N_domain_sub2 = 20000
    N_boundary = 1500

    x_boundary, y_boundary = generate_boundary_points(
        domain, N_boundary, lb, ub, cylinder_center, cylinder_radius
    )
    
    x_interior_1, y_interior_1 = generate_interior_points(domain, N_domain, lb, ub)
    x_interior_2, y_interior_2 = generate_interior_points(domain, N_domain_sub1, lb_2, ub_2)
    x_interior_3, y_interior_3 = generate_interior_points(domain, N_domain_sub2, lb_3, ub_3)
   
    # Integrate data points
    x_interior = np.concatenate([x_interior_1, x_interior_2, x_interior_3], axis=0)
    y_interior = np.concatenate([y_interior_1, y_interior_2, y_interior_3], axis=0) 
    

    # 데이터 분포 시각화
    plot_data_distribution(domain, x_interior, y_interior, x_boundary, y_boundary, lb, ub)

    # ----------------------------
    # 4. PINN 모델 초기화
    # ----------------------------
    layers_kan = [2, 100, 100, 100, 100, 3]
    model_kan = PINN_KAN(domain, layers_kan, lb, ub, mu=mu, rho=rho).to(device)

    layers_mlp = [2, 100, 100, 100, 100, 3]
    model_mlp = PINN_MLP(domain, layers_mlp, lb, ub, mu=mu, rho=rho).to(device)

    # ----------------------------
    # 5. 손실 가중치 정의
    # ----------------------------
    bc_weights = {'inlet': 1.0, 'top': 1.0, 'bottom': 1.0, 'outlet_p': 1.0, 'outlet_du_dy': 1.0, 'wall': 1.0}
    learning_rate = 0.001
    # ----------------------------
    # 6. 옵티마이저 정의
    # ----------------------------
    optimizer_kan = optim.Adam(model_kan.parameters(), lr=learning_rate)
    optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=learning_rate)
    #optimizer_kan = optim.LBFGS(model_kan.parameters(), lr=1.0, max_iter=50000, history_size=100, line_search_fn='strong_wolfe')
    #optimizer_mlp = optim.LBFGS(model_mlp.parameters(), lr=1.0, max_iter=50000, history_size=100, line_search_fn='strong_wolfe')
    # ----------------------------
    # 7. 훈련 파라미터 설정
    # ----------------------------
    n_epochs_adam = 100000
    batch_size_adam = 1000

    # ----------------------------
    # 8. KAN 기반 PINN 훈련
    # ----------------------------
    print("KAN 기반 PINN 훈련 시작...")
    losses_kan = train_pinn(
        model=model_kan,
        optimizer=optimizer_kan,
        x_interior=x_interior,
        y_interior=y_interior,
        x_boundary=x_boundary,
        y_boundary=y_boundary,
        lb=lb,
        ub=ub,
        n_epochs=n_epochs_adam,
        bc_weights=bc_weights,
        inlet_velocity=U,  # 입구 속도 전달
        batch_size=batch_size_adam,
        model_type='KAN'
    )

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
        lb=lb,
        ub=ub,
        n_epochs=n_epochs_adam,
        bc_weights=bc_weights,
        inlet_velocity=U,  # 입구 속도 전달
        batch_size=batch_size_adam,
        model_type='MLP'
    )

    # ----------------------------
    # 10. 손실 그래프 시각화
    # ----------------------------
    plot_combined_loss(losses_kan, losses_mlp, n_epochs_adam)

    # ----------------------------
    # 11. 시각화 데이터 생성
    # ----------------------------
    X, Y, mask = create_masked_visualization_data(domain, 600, 300, lb, ub)

    # ----------------------------
    # 12. 결과 비교 시각화
    # ----------------------------
    plot_comparison_results(model_kan, model_mlp, X, Y, mask)
