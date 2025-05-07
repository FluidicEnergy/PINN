import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# DeepONet 구성 요소
# -------------------------------
class BranchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BranchNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class TrunkNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TrunkNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    """
    DeepONet: branch network와 trunk network의 내적 결과로 operator output을 생성합니다.
    """
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(branch_input_dim, hidden_dim, output_dim)
        self.trunk_net = TrunkNet(trunk_input_dim, hidden_dim, output_dim)
        
    def forward(self, branch_input, trunk_input):
        # branch_input: (batch_size, branch_input_dim)
        # trunk_input: (batch_size, trunk_input_dim) → 예: (x, y, t)
        branch_out = self.branch_net(branch_input)  # (batch_size, output_dim)
        trunk_out = self.trunk_net(trunk_input)       # (batch_size, output_dim)
        # 두 네트워크의 내적 결과 (scalar 예측)
        output = torch.sum(branch_out * trunk_out, dim=1, keepdim=True)
        return output

# -------------------------------
# URANS 문제를 위한 PINN 모델 (각 변수별 DeepONet 사용)
# -------------------------------
class URANS_PINN(nn.Module):
    """
    URANS_PINN: u, v, p 세 개의 물리량을 각각 별도의 DeepONet으로 예측합니다.
    입력으로는 branch_input (예: 초기/경계 조건 정보를 압축한 벡터)와 
    공간-시간 좌표 (x, y, t)를 받습니다.
    """
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim, output_dim):
        super(URANS_PINN, self).__init__()
        self.deeponet_u = DeepONet(branch_input_dim, trunk_input_dim, hidden_dim, output_dim)
        self.deeponet_v = DeepONet(branch_input_dim, trunk_input_dim, hidden_dim, output_dim)
        self.deeponet_p = DeepONet(branch_input_dim, trunk_input_dim, hidden_dim, output_dim)
        
    def forward(self, branch_input, x):
        # x: (batch_size, 3) → (x, y, t)
        u = self.deeponet_u(branch_input, x)
        v = self.deeponet_v(branch_input, x)
        p = self.deeponet_p(branch_input, x)
        return u, v, p

# -------------------------------
# URANS PDE 잔차 (Residual) 계산 함수
# -------------------------------
def compute_residuals(model, branch_input, x):
    """
    URANS의 간단화된 방정식을 사용합니다.
    Continuity: u_x + v_y = 0
    Momentum (x): u_t + u*u_x + v*u_y + p_x/rho - nu*(u_xx+u_yy) = 0
    Momentum (y): v_t + u*v_x + v*v_y + p_y/rho - nu*(v_xx+v_yy) = 0
    """
    x.requires_grad = True
    u, v, p = model(branch_input, x)  # 각 변수 shape: (N, 1)

    # u에 대한 미분
    grad_u = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                 retain_graph=True, create_graph=True)[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, x, grad_outputs=torch.ones_like(u_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]

    # v에 대한 미분
    grad_v = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                 retain_graph=True, create_graph=True)[0]
    v_x = grad_v[:, 0:1]
    v_y = grad_v[:, 1:2]
    v_t = grad_v[:, 2:3]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                               retain_graph=True, create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, x, grad_outputs=torch.ones_like(v_y),
                               retain_graph=True, create_graph=True)[0][:, 1:2]

    # p에 대한 미분 (공간 미분만 고려)
    grad_p = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                                 retain_graph=True, create_graph=True)[0]
    p_x = grad_p[:, 0:1]
    p_y = grad_p[:, 1:2]

    # 상수 (예시)
    rho = 1.0
    nu = 0.01

    # 연속 방정식 잔차
    r_continuity = u_x + v_y

    # x-모멘텀 잔차
    r_momentum_x = u_t + u * u_x + v * u_y + p_x / rho - nu * (u_xx + u_yy)
    # y-모멘텀 잔차
    r_momentum_y = v_t + u * v_x + v * v_y + p_y / rho - nu * (v_xx + v_yy)

    return r_continuity, r_momentum_x, r_momentum_y

# -------------------------------
# 도메인 및 경계 조건 샘플링 함수
# -------------------------------
def generate_collocation_points(N, x_range, y_range, t_range):
    """
    PDE 잔차를 최소화하기 위한 collocation points를 도메인 내에서 무작위 생성.
    """
    x = torch.rand(N, 1) * (x_range[1] - x_range[0]) + x_range[0]
    y = torch.rand(N, 1) * (y_range[1] - y_range[0]) + y_range[0]
    t = torch.rand(N, 1) * (t_range[1] - t_range[0]) + t_range[0]
    return torch.cat([x, y, t], dim=1)

def generate_cylinder_bc_points(N, radius, t_range):
    """
    원통(원)의 경계에서의 no-slip 조건을 위해, 원 둘레 상의 점들과 시간 값을 생성.
    """
    theta = 2 * np.pi * torch.rand(N, 1)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    t = torch.rand(N, 1) * (t_range[1] - t_range[0]) + t_range[0]
    return torch.cat([x, y, t], dim=1)

# -------------------------------
# 모델 학습 및 실행
# -------------------------------
if __name__ == "__main__":
    # 하이퍼파라미터 설정
    branch_input_dim = 10   # 예: 센서 데이터를 통한 초기/경계 조건 압축 벡터 차원
    trunk_input_dim = 3     # (x, y, t)
    hidden_dim = 50
    output_dim = 50         # branch와 trunk에서 출력되는 latent 차원

    # URANS_PINN 모델 생성 (u, v, p 각각 별도의 DeepONet 사용)
    model = URANS_PINN(branch_input_dim, trunk_input_dim, hidden_dim, output_dim)
    
    # Adam 옵티마이저
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 학습에 사용할 도메인 설정: x ∈ [-5,15], y ∈ [-5,5], t ∈ [0,1]
    x_range = (-5, 15)
    y_range = (-5, 5)
    t_range = (0, 1)
    
    # 학습 루프에 사용할 collocation point 개수
    N_collocation = 1000
    
    # Cylinder boundary condition을 위한 점 (원통 표면: 반지름 1)
    N_bc = 200
    
    epochs = 5000
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE 잔차를 위한 collocation points 생성
        x_coll = generate_collocation_points(N_collocation, x_range, y_range, t_range)
        # 각 collocation point마다 branch input은 무작위 벡터 (실제 문제에선 초기/경계 조건 정보를 사용)
        branch_input_coll = torch.randn(x_coll.shape[0], branch_input_dim)
        r_cont, r_mom_x, r_mom_y = compute_residuals(model, branch_input_coll, x_coll)
        loss_pde = torch.mean(r_cont**2) + torch.mean(r_mom_x**2) + torch.mean(r_mom_y**2)
        
        # 원통 경계에서의 no-slip 조건 (u = 0, v = 0)을 위한 loss
        x_bc = generate_cylinder_bc_points(N_bc, radius=1.0, t_range=t_range)
        branch_input_bc = torch.randn(x_bc.shape[0], branch_input_dim)
        u_bc, v_bc, _ = model(branch_input_bc, x_bc)
        loss_bc = torch.mean(u_bc**2) + torch.mean(v_bc**2)
        
        # 전체 loss (경계 조건 loss에 가중치 lambda_bc 적용)
        lambda_bc = 10.0
        loss = loss_pde + lambda_bc * loss_bc
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print("Epoch {}: Total Loss = {:.6f}, PDE Loss = {:.6f}, BC Loss = {:.6f}".format(
                epoch, loss.item(), loss_pde.item(), loss_bc.item()))
    
    # -------------------------------
    # 결과 시각화: t=0.5인 평면에서 (x, y) 좌표에 대한 u, v, p 예측
    # -------------------------------
    grid_nx, grid_ny = 50, 50
    x_vals = np.linspace(x_range[0], x_range[1], grid_nx)
    y_vals = np.linspace(y_range[0], y_range[1], grid_ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    T = 0.5 * np.ones_like(X)  # t=0.5 고정

    # mesh grid의 각 점에 대해 (x, y, t) 좌표 생성
    grid_points = np.stack([X.flatten(), Y.flatten(), T.flatten()], axis=1)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
    
    # 각 grid point마다 동일한 branch input (여기서는 무작위 벡터를 재사용)
    branch_input_grid = torch.randn(grid_points_tensor.shape[0], branch_input_dim)
    
    model.eval()
    with torch.no_grad():
        u_grid, v_grid, p_grid = model(branch_input_grid, grid_points_tensor)
        u_grid = u_grid.reshape((grid_ny, grid_nx)).detach().numpy()
        v_grid = v_grid.reshape((grid_ny, grid_nx)).detach().numpy()
        p_grid = p_grid.reshape((grid_ny, grid_nx)).detach().numpy()
    
    plt.figure(figsize=(18,5))
    
    plt.subplot(1,3,1)
    plt.contourf(X, Y, u_grid, 20, cmap='jet')
    plt.colorbar()
    plt.title('Predicted u')
    
    plt.subplot(1,3,2)
    plt.contourf(X, Y, v_grid, 20, cmap='jet')
    plt.colorbar()
    plt.title('Predicted v')
    
    plt.subplot(1,3,3)
    plt.contourf(X, Y, p_grid, 20, cmap='jet')
    plt.colorbar()
    plt.title('Predicted p')
    
    plt.tight_layout()
    plt.show()
