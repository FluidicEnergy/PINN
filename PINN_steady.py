import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimag
import time

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

class PINN(nn.Module):
    def __init__(self, domain, layers, lb, ub, mu):
        super(PINN, self).__init__()
        self.domain = domain
        self.lb = torch.tensor(lb, dtype=torch.float32).to(device)
        self.ub = torch.tensor(ub, dtype=torch.float32).to(device)
        self.mu = mu

        # 신경망 정의
        self.net = nn.Sequential()
        for i in range(len(layers) - 2):
            self.net.add_module(f'linear_{i}', nn.Linear(layers[i], layers[i+1]))
            self.net.add_module(f'tanh_{i}', nn.Tanh())
        self.net.add_module(f'linear_out', nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
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
        
        f_u = (u*u_x + v*u_y) + p_x - self.mu*(u_xx + u_yy)
        f_v = (u*v_x + v*v_y) + p_y - self.mu*(v_xx + v_yy)
        f_e = u_x + v_y
        
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

def generate_boundary_points(domain, N_points, lb, ub, lb_obstacle, ub_obstacle):
    eps = 1e-5

    [obstacle_x1, obstacle_y1] = lb_obstacle
    [obstacle_x2, obstacle_y2] = ub_obstacle

    N_each = N_points // 6  # 6개의 경계(좌, 우, 상, 하, 장애물 상단, 장애물 측면)

    # 왼쪽 벽
    x_left = torch.full((N_each, 1), lb[0], device=device)
    y_left = torch.rand(N_each, 1, device=device) * (ub[1] - lb[1]) + lb[1]

    # 오른쪽 벽
    x_right = torch.full((N_each, 1), ub[0], device=device)
    y_right = torch.rand(N_each, 1, device=device) * (ub[1] - lb[1]) + lb[1]

    # 상단 벽
    x_top = torch.rand(N_each, 1, device=device) * (ub[0] - lb[0]) + lb[0]
    y_top = torch.full((N_each, 1), ub[1], device=device)

    # 하단 벽 (장애물 제외)
    x_bottom = torch.cat([
        torch.rand(N_each // 4, 1, device=device) * obstacle_x1,
        torch.rand(N_each //2 - N_each // 4, 1, device=device) * (ub[0] - obstacle_x2) + obstacle_x2
    ])
    y_bottom = torch.full_like(x_bottom, lb[1])

    # 장애물 상단
    x_obstacle_top = torch.rand(N_each, 1, device=device) * (obstacle_x2 - obstacle_x1) + obstacle_x1
    y_obstacle_top = torch.full_like(x_obstacle_top, obstacle_y2)

    # 장애물 측면
    x_obstacle_side = torch.cat([
        torch.full((N_each // 2, 1), obstacle_x1, device=device),
        torch.full((N_each - N_each // 2, 1), obstacle_x2, device=device)
    ])
    y_obstacle_side = torch.rand(N_each, 1, device=device) * obstacle_y2

    # 모든 경계 포인트 결합
    x_boundary = torch.cat([x_left, x_right, x_top, x_bottom, x_obstacle_top, x_obstacle_side])
    y_boundary = torch.cat([y_left, y_right, y_top, y_bottom, y_obstacle_top, y_obstacle_side])

    return x_boundary, y_boundary

def plot_data_distribution(domain, x_interior, y_interior, x_boundary, y_boundary, lb, ub):
    plt.figure(figsize=(12, 6))
    
    # 도메인 형상 그리기
    nx, ny = 1000, 500
    x = np.linspace(lb[0], ub[0], nx)
    y = np.linspace(lb[1], ub[1], ny)
    X, Y = np.meshgrid(x, y)
    X_flat = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    Y_flat = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
    Z = domain(X_flat, Y_flat).cpu().numpy().reshape(X.shape)
    #plt.contourf(X, Y, Z, levels=[float('-inf'), 0, float('inf')], colors=['lightgray', 'white'])
    
    # 내부 포인트 그리기
    plt.scatter(x_interior.cpu(), y_interior.cpu(), c='lightblue', s=1, alpha=0.5, label='Interior points')
    
    # 경계 포인트 그리기
    eps = 1e-5
    inlet_mask = y_boundary.squeeze() > (ub[1] - eps)
    outlet_mask = y_boundary.squeeze() < eps
    wall_mask = ~(inlet_mask | outlet_mask)

    plt.scatter(x_boundary[inlet_mask].cpu(), y_boundary[inlet_mask].cpu(), c='red', s=5, label='Inlet')
    plt.scatter(x_boundary[outlet_mask].cpu(), y_boundary[outlet_mask].cpu(), c='green', s=5, label='Outlet')
    plt.scatter(x_boundary[wall_mask].cpu(), y_boundary[wall_mask].cpu(), c='blue', s=5, label='Wall')

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

def plot_combined_loss(losses_adam, losses_lbfgs, n_epochs_adam, n_epochs_lbfgs):
    plt.figure(figsize=(12, 8))
    
    # Adam loss
    iterations_adam = np.arange(len(losses_adam['total']))
    plt.semilogy(iterations_adam, losses_adam['total'], label='Total Loss (Adam)', color='blue')
    plt.semilogy(iterations_adam, losses_adam['pde'], label='PDE Loss (Adam)', color='cyan')
    plt.semilogy(iterations_adam, losses_adam['bc'], label='BC Loss (Adam)', color='navy')
    
    # L-BFGS loss
    iterations_lbfgs = np.arange(len(losses_lbfgs['total'])) + len(losses_adam['total'])
    plt.semilogy(iterations_lbfgs, losses_lbfgs['total'], label='Total Loss (L-BFGS)', color='red')
    plt.semilogy(iterations_lbfgs, losses_lbfgs['pde'], label='PDE Loss (L-BFGS)', color='orange')
    plt.semilogy(iterations_lbfgs, losses_lbfgs['bc'], label='BC Loss (L-BFGS)', color='darkred')
    
    plt.xlim([0, n_epochs_adam+n_epochs_lbfgs])
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Combined Adam and L-BFGS Optimization Losses')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_plot.jpg')
    #plt.show()

def plot_results(model, X, Y, mask):
    with torch.no_grad():
        x = torch.tensor(X.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        y = torch.tensor(Y.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        u, v, p = model.net_uvp(x, y)
        u = u.cpu().numpy().reshape(X.shape)
        v = v.cpu().numpy().reshape(X.shape)
        p = p.cpu().numpy().reshape(X.shape)

    u = np.ma.masked_array(u, ~mask)
    v = np.ma.masked_array(v, ~mask)
    p = np.ma.masked_array(p, ~mask)

    fig, axs = plt.subplots(3, 1, figsize=(5, 20))
    
    im1 = axs[0].imshow(u, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='jet')
    axs[0].set_title('u velocity')
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(v, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='jet')
    axs[1].set_title('v velocity')
    fig.colorbar(im2, ax=axs[1])

    im3 = axs[2].imshow(p, extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower', cmap='jet')
    axs[2].set_title('pressure')
    fig.colorbar(im3, ax=axs[2])

    plt.tight_layout()
    plt.savefig('result_plot.jpg')
    #plt.show()

def train(model, optimizer, x_interior, y_interior, x_boundary, y_boundary, lb, ub, n_epochs, bc_weights, batch_size=None):
    losses = {'total': [], 'pde': [], 'bc': [], 'wall_bc': [], 'inlet_bc': [], 'outlet_bc': [], 'outlet_dv_dx_bc': []}
    eps = 1e-6

    # 직사각형 장애물의 위치와 크기 정의
    obstacle_x1, obstacle_y1 = 0.05, 0
    obstacle_x2, obstacle_y2 = 0.3, 0.1

    # 경계 조건 정의
    wall_bc = DirichletBC(
        lambda x, y: ((x < lb[0] + eps) | (x > ub[0] - eps) |
                      ((y < lb[1] + eps) & ((x <= obstacle_x1 - eps) | (x >= obstacle_x2 + eps))) |
                      ((x >= obstacle_x1 - eps) & (x <= obstacle_x2 + eps) & (y <= obstacle_y2 + eps))),
        lambda x, y: (torch.zeros_like(x), torch.zeros_like(y), None)
    )
    inlet_bc = DirichletBC(
        lambda x, y: (y > ub[1] - eps),
        lambda x, y: (torch.zeros_like(x), -0.1 *4*(x/ub[0])* (1 - x/ub[0]) /ub[0] * torch.ones_like(y), None)
    )
    outlet_bc = DirichletBC(
        lambda x, y: (y < lb[1] + eps) & ((x <= obstacle_x1 - eps) | (x >= obstacle_x2 + eps)),
        lambda x, y: (None, None, torch.zeros_like(x))
    )

    # dv/dx = 0 at the outlet
    outlet_bc_dv_dx = NeumannBC(
        lambda x, y: (y < lb[1] + eps) & ((x <= obstacle_x1 - eps) | (x >= obstacle_x2 + eps)),
        lambda x, y: ('dv/dx', torch.zeros_like(x))
    )


    if batch_size is None:
        batch_size = len(x_interior)

    def closure():
        optimizer.zero_grad()
        
        # 미니배치 샘플링
        idx = np.random.choice(len(x_interior), batch_size, replace=False)
        x_batch = x_interior[idx]
        y_batch = y_interior[idx]
        
        # PDE 손실
        f_u, f_v, f_e = model.net_f(x_batch, y_batch)
        loss_pde = torch.mean(f_u**2 + f_v**2 + f_e**2)
        
        # 경계 조건 손실 계산
        u_b, v_b, p_b = model.net_uvp(x_boundary, y_boundary)
        
        # Wall BC 적용 및 손실 계산
        u_b_wall, v_b_wall, p_b_wall = wall_bc.apply(x_boundary, y_boundary, u_b.clone(), v_b.clone(), p_b.clone())
        loss_wall_bc = (torch.mean((u_b_wall - u_b)**2) + 
                        torch.mean((v_b_wall - v_b)**2))
        
        # Inlet BC 적용 및 손실 계산
        u_b_inlet, v_b_inlet, p_b_inlet = inlet_bc.apply(x_boundary, y_boundary, u_b.clone(), v_b.clone(), p_b.clone())
        loss_inlet_bc = (torch.mean((u_b_inlet - u_b)**2) + 
                         torch.mean((v_b_inlet - v_b)**2))
        
        # Outlet BC 적용 및 손실 계산
        u_b_outlet, v_b_outlet, p_b_outlet = outlet_bc.apply(x_boundary, y_boundary, u_b.clone(), v_b.clone(), p_b.clone())
        loss_outlet_bc = (torch.mean((u_b_outlet - u_b)**2) + 
                          torch.mean((v_b_outlet - v_b)**2) +
                          torch.mean((p_b_outlet - p_b)**2))

        # Outlet dv/dx BC 적용 및 손실 계산
        u_b_outlet_dx, v_b_outlet_dx, p_b_outlet_dx = outlet_bc_dv_dx.apply(x_boundary, y_boundary, u_b.clone(), v_b.clone(), p_b.clone())
        loss_outlet_dv_dx_bc = (torch.mean((u_b_outlet_dx - u_b)**2) + 
                          torch.mean((v_b_outlet_dx - v_b)**2))
        
        # 전체 손실
        loss = (loss_pde + 
                bc_weights.get('wall', 3.0) * loss_wall_bc + 
                bc_weights.get('inlet', 1.0) * loss_inlet_bc + 
                bc_weights.get('outlet', 1.0) * loss_outlet_bc + 
                bc_weights.get('outlet_dv_dx', 1.0) * loss_outlet_dv_dx_bc)
        
        loss.backward()
        
        # losses 딕셔너리 업데이트
        losses['total'].append(loss.item())
        losses['pde'].append(loss_pde.item())
        losses['bc'].append(loss_wall_bc.item() + loss_inlet_bc.item() + loss_outlet_bc.item() + loss_outlet_dv_dx_bc.item())
        losses['wall_bc'].append(loss_wall_bc.item())
        losses['inlet_bc'].append(loss_inlet_bc.item())
        losses['outlet_bc'].append(loss_outlet_bc.item())
        losses['outlet_dv_dx_bc'].append(loss_outlet_dv_dx_bc.item())
        
        return loss
    

    start_time = time.time()
    for epoch in range(n_epochs):
        loss = optimizer.step(closure)
        
        if epoch % 100 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch}, Total Loss: {losses["total"][-1]:.4e}, PDE Loss: {losses["pde"][-1]:.4e}, BC Loss: {losses["bc"][-1]:.4e}, Time: {elapsed:.2f}s')
            start_time = time.time()

    return losses

if __name__ == "__main__":
    # 도메인 정의
    lb = np.array([0, 0])
    ub = np.array([0.35, 0.2])
    
    lb_obstacle = np.array([0.075, 0])
    ub_obstacle = np.array([0.275, 0.1])

    channel = Rectangle(lb[0], lb[1], ub[0], ub[1])
    obstacle = Rectangle(lb_obstacle[0], lb_obstacle[1], ub_obstacle[0], ub_obstacle[1])
    sub_channel = Rectangle(lb[0], 0.3*ub[1], ub[0], 0.7*ub[1])

    domain = Difference(channel, obstacle)
    sub_domain = Difference(sub_channel, obstacle)
    # 학습 데이터 생성
    N_domain = 8000
    N_boundary = 8000
    N_sub_add= 16000

    x_boundary, y_boundary = generate_boundary_points(domain, N_boundary, lb, ub, lb_obstacle, ub_obstacle)
    x_interior, y_interior = generate_interior_points(domain, N_domain, lb, ub)
    x_sub_interior, y_sub_interior = generate_interior_points(sub_domain, N_sub_add, lb, ub)
    x_interior = torch.cat((x_interior, x_sub_interior), dim=0)
    y_interior = torch.cat((y_interior, y_sub_interior), dim=0)
    
    # 데이터 포인트 분포 시각화
    plot_data_distribution(domain, x_interior, y_interior, x_boundary, y_boundary, lb, ub)

    # PINN 모델 초기화
    layers = [2, 100, 100, 100, 100, 100, 100, 3]  # 레이어 크기 증가
    model = PINN(domain, layers, lb, ub, mu=0.01).to(device)

    # Loss weights
    bc_weights = {'wall': 9, 'inlet': 9, 'outlet': 9, 'outlet_dv_dx': 9}

    # Adam 최적화기 설정
    adam_optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Adam으로 초기 학습
    n_epochs_adam = 10000  # Adam 에폭 수 증가
    batch_size_adam = 1000  # 배치 크기 설정
    losses_adam = train(model, adam_optimizer, x_interior, y_interior, x_boundary, y_boundary, lb, ub, n_epochs_adam, bc_weights, batch_size_adam)

    # L-BFGS 최적화기 설정
    lbfgs_optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=50, max_eval=50, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=50, line_search_fn="strong_wolfe")

    # L-BFGS로 추가 학습
    n_epochs_lbfgs = 2  # L-BFGS 에폭 수 감소
    losses_lbfgs = train(model, lbfgs_optimizer, x_interior, y_interior, x_boundary, y_boundary, lb, ub, n_epochs_lbfgs, bc_weights)

    # 결과 시각화
    plot_combined_loss(losses_adam, losses_lbfgs, n_epochs_adam, n_epochs_lbfgs)
    X, Y, mask = create_masked_visualization_data(domain, 200, 100, lb, ub)
    plot_results(model, X, Y, mask)