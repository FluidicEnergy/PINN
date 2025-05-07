import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import Rbf
from scipy.spatial import cKDTree
from matplotlib import cm
import matplotlib.cbook as cbook
import matplotlib.colors as colors

dde.config.set_random_seed(48)
dde.config.set_default_float('float64')

# Properties
rho = 1
mu = 0.01
u_max = 0.3
L =4 #Boundary length
H=2 #boundary height
angle=10 #Attack angle of airfoil

# Airfoil Data input
data_str = """1.0000     0.00252
0.9500     0.01613
0.9000     0.02896
0.8000     0.05247
0.7000     0.07328
0.6000     0.09127
0.5000     0.10588
0.4000     0.11607
0.3000     0.12004
0.2500     0.11883
0.2000     0.11475
0.1500     0.10691
0.1000     0.09365
0.0750     0.08400
0.0500     0.07109
0.0250     0.05229
0.0125     0.03788
0.0000     0.00000
0.0125     -0.03788
0.0250     -0.05229
0.0500     -0.07109
0.0750     -0.08400
0.1000     -0.09365
0.1500     -0.10691
0.2000     -0.11475
0.2500     -0.11883
0.3000     -0.12004
0.4000     -0.11607
0.5000     -0.10588
0.6000     -0.09127
0.7000     -0.07328
0.8000     -0.05247
0.9000     -0.02896
0.9500     -0.01613
1.0000     -0.00252
"""

#Rotate Airfoil
def rotate_coordinates(coordinates, angle):
    # Define the rotation matrix for 30 degrees
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Rotate each coordinate
    rotated_coordinates = np.dot(coordinates, rotation_matrix)
    return rotated_coordinates


# Convert the string to a list of lists, splitting by whitespace
data_list = [list(map(float, line.split())) for line in data_str.strip().split('\n')]

# Convert the list of lists to a numpy array
array = np.array(data_list)
array[:,0]=array[:,0]-0.5

array= rotate_coordinates(array, angle)

# Boundarys

def boundary_wall(X, on_boundary):
    on_wall = np.logical_and(np.logical_or(np.isclose(X[1], -H/2), np.isclose(X[1], H/2)), on_boundary)
    return on_wall

def boundary_airfoil(X,on_boundary):

  on_airfoil= np.logical_and(np.isclose(array, X,atol=0.01).any(),on_boundary)
  return on_airfoil

def boundary_inlet(X, on_boundary):
  return on_boundary and np.isclose(X[0], -L/2)

def boundary_outlet(X, on_boundary):
    return on_boundary and np.isclose(X[0], L/2)

# Gaverning Equation
def pde(X, Y):
    du_x = dde.grad.jacobian(Y, X, i = 0, j = 0)
    du_y = dde.grad.jacobian(Y, X, i = 0, j = 1)
    dv_x = dde.grad.jacobian(Y, X, i = 1, j = 0)
    dv_y = dde.grad.jacobian(Y, X, i = 1, j = 1)
    dp_x = dde.grad.jacobian(Y, X, i = 2, j = 0)
    dp_y = dde.grad.jacobian(Y, X, i = 2, j = 1)
    du_xx = dde.grad.hessian(Y, X, i = 0, j = 0, component = 0)
    du_yy = dde.grad.hessian(Y, X, i = 1, j = 1, component = 0)
    dv_xx = dde.grad.hessian(Y, X, i = 0, j = 0, component = 1)
    dv_yy = dde.grad.hessian(Y, X, i = 1, j = 1, component = 1)

    pde_u = Y[:,0:1]*du_x + Y[:,1:2]*du_y + 1/rho * dp_x - (mu/rho)*(du_xx + du_yy)
    pde_v = Y[:,0:1]*dv_x + Y[:,1:2]*dv_y + 1/rho * dp_y - (mu/rho)*(dv_xx + dv_yy)
    pde_cont = du_x + dv_y

    return [pde_u, pde_v, pde_cont]


def filter_close_points(x, y, vx, vy, pressure, min_distance=0.05):
    tree = cKDTree(np.c_[x, y])
    mask = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if mask[i]:
            neighbors = tree.query_ball_point([x[i], y[i]], min_distance)
            neighbors.remove(i)
            mask[neighbors] = False
    return x[mask], y[mask], vx[mask], vy[mask], pressure[mask]


#Geometry & points

windtunnel= dde.geometry.Rectangle(xmin=[-L/2, -H/2], xmax=[L/2, H/2])
airfoil=dde.geometry.geometry_2d.Polygon(array)

geom = dde.geometry.csg.CSGDifference(windtunnel, airfoil)

print(round(np.cos(angle),2))
inner_rec  = dde.geometry.Rectangle([-0.75,-H/4],[0.75,H/4])
outer_dom  = dde.geometry.CSGDifference(windtunnel, inner_rec)
outer_dom  = dde.geometry.CSGDifference(outer_dom, airfoil)
inner_dom  = dde.geometry.CSGDifference(inner_rec, airfoil)

inner_points = inner_dom.random_points(6000)
outer_points = outer_dom.random_points(10000)

#geom_points=geom.random_points(8000)

windtunnel_points=windtunnel.random_boundary_points(1500)
airfoil_points=airfoil.random_boundary_points(800)

points = np.append(inner_points, outer_points, axis = 0)
points = np.append(points, windtunnel_points, axis = 0)
points = np.append(points, airfoil_points, axis = 0)

def fun_inlet_u(X):
  z=4*u_max*((H/2-X[:,1])/H)**2
  z=u_max*(1-(2*X[:,1]/H)**2)

  #z=0.3

  return z

#inlet boundary condition
n_in=100
rep=3

y_origin=np.linspace(-H/2,H/2,n_in).reshape(n_in,1)
y_in = y_origin

fun_u_origin=u_max*(1-(2*y_origin/H)**2)
fun_u=fun_u_origin

x_in=np.linspace(-L/2,-0.98*L/2,rep).reshape(rep,1)
x_in=np.repeat(x_in,n_in,0)
y_in = y_origin

for i in range(rep-1):
    y_in=np.vstack((y_in,y_origin))
    fun_u=np.vstack((fun_u,fun_u_origin))
    
inlet_points=np.hstack((x_in,y_in))


#Boundary condition

bc_wall_u = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component = 0)
bc_wall_v = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component = 1)
#bc_wall_p = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component = 2)

bc_airfoil_u = dde.DirichletBC(geom, lambda X: 0., boundary_airfoil, component = 0)
bc_airfoil_v = dde.DirichletBC(geom, lambda X: 0., boundary_airfoil, component = 1)
#bc_airfoil_p = dde.DirichletBC(geom, lambda X: 0., boundary_airfoil, component = 2)

bc_inlet_u = dde.PointSetBC(inlet_points,fun_u, component = 0)
#bc_inlet_u = dde.DirichletBC(geom, fun_inlet_u, boundary_inlet, component = 0)
#bc_inlet_u = dde.DirichletBC(geom, lambda X: u_in, boundary_inlet, component = 0)
bc_inlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_inlet, component = 1)

bc_outlet_p = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component = 2)
#bc_outlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component = 1)

bcs=[bc_wall_u, bc_wall_v, bc_airfoil_u ,bc_airfoil_v,  bc_inlet_u, bc_inlet_v, bc_outlet_p]


data = dde.data.PDE(geom,
                    pde,
                    bcs,
                    num_domain = 5000,
                    num_boundary = 1000,
                    num_test = 1000,
                    train_distribution = 'Hammersley' )


layer_size = [2] + [40] *8 + [3]
activation = "tanh"
initializer = "Glorot uniform"

net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("L-BFGS-B", loss_weights = [1, 1, 1, 1, 1, 1, 1, 9, 1,1])
model.train_step.optimizer_kwargs = {'options': {'maxcor': 50,
                                                   'ftol': 1.0 * np.finfo(float).eps,
                                             'maxfun':  50000,
                                                   'maxiter': 1,
                                                   'maxls': 50}}
#model.train(model_restore_path="0424-30000.pt")
model.restore("0516/0516-60000.pt")  # Replace ? with the exact filename


#load CFD Data
CFD_Data=pd.read_csv("CFD3.csv")
df=CFD_Data.drop([0,1,2,3],axis=0)
df.columns=['x','y','u','v','p']

x=df["x"].values.astype(np.float64)
y=df["y"].values.astype(np.float64)
u=df["u"].values.astype(np.float64)
v=df["v"].values.astype(np.float64)
p=df["p"].values.astype(np.float64)


samples=geom.random_points(300000)
CFD_samples=np.column_stack([x,y])

result_CFD = np.hstack([u.reshape(-1,1),v.reshape(-1,1),p.reshape(-1,1)])
result_PINN = model.predict(samples)

#interpolate
x_i, y_i, u, v, p = filter_close_points(x, y, u, v, p, min_distance=0.01)

rbf_u = Rbf(x_i, y_i, u, function='linear')
rbf_v = Rbf(x_i, y_i, v, function='linear')
rbf_p = Rbf(x_i, y_i, p, function='linear')


u=rbf_u(samples[:,0],samples[:,1]).reshape(-1,1)
v=rbf_v(samples[:,0],samples[:,1]).reshape(-1,1)
p=rbf_p(samples[:,0],samples[:,1]).reshape(-1,1)

result_CFD = np.hstack([u,v,p])

Err=np.abs(result_CFD-result_PINN)
relative_Err=np.abs((result_CFD-result_PINN)/(result_CFD+1e-4))
relative_Err=np.where(relative_Err>1,1,relative_Err)
Max_Z=[np.mean(np.abs(result_CFD[:,0])),np.mean(np.abs(result_CFD[:,1])),np.mean(np.abs(result_CFD[:,2]))]


color_legend = [[0, 0.35],[-0.15,0.15],[-0.15,0.15]]

ssi=0.05


for idx in range(3):
    plt.subplot(3,1,idx+1)
    plt.scatter(samples[:, 0],
                samples[:, 1],
                c = result_PINN[:,idx],
                cmap = 'jet',
                s = ssi)
    plt.colorbar()
    plt.clim(color_legend[idx])
    plt.xlim((0-L/2, L-L/2))
    plt.ylim((0-H/2, H-H/2))


plt.tight_layout()
plt.show()

for idx in range(3):
    plt.subplot(3,1,idx+1)
    plt.scatter(samples[:, 0],
                samples[:, 1],
                c = result_CFD[:,idx],
                cmap = 'jet',
                s = ssi)
    plt.colorbar()
    plt.clim(color_legend[idx])
    plt.xlim((0-L/2, L-L/2))
    plt.ylim((0-H/2, H-H/2))


plt.tight_layout()
plt.show()

color_legend = [[0, 1],[0,1],[0,1]]


for idx in range(3):
    Z=Err[:,idx]/Max_Z[idx]
    plt.subplot(3,1,idx+1)
    plt.scatter(samples[:, 0],
                samples[:, 1],
                c = Z,
                cmap = 'jet',
                s = ssi)
    plt.colorbar()
    #plt.clim(color_legend[idx])
    plt.xlim((0-L/2, L-L/2))
    plt.ylim((0-H/2, H-H/2))


plt.tight_layout()
plt.show()


fig,axes= plt.subplots(3,1)
for idx in range(3):
    Z=np.abs(result_CFD[:,idx]-result_PINN[:,idx])/np.mean(np.abs(result_CFD[:,idx]))
    axes[idx].scatter(samples[:, 0],
                samples[:, 1],
                c = Z,
                cmap = 'jet',
                s = ssi)
    norm=colors.LogNorm(vmin=np.min(Z),vmax=np.max(Z))
    sm=plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    fig.colorbar(sm,ax=axes[idx])
    axes[idx].set_xlim((-L/2,L- L/2))
    axes[idx].set_ylim((0-H/2, H-H/2))

plt.tight_layout()
plt.show()




for idx in range(3):
    plt.figure(figsize = (20, 4))
    plt.scatter(samples[:, 0],
                samples[:, 1],
                c = result_PINN[:, idx],
                s = 2)
    plt.colorbar()
    plt.clim(color_legend[idx])
    plt.xlim((0-L/2, L-L/2))
    plt.ylim((0-H/2, H-H/2))
    plt.tight_layout()
    plt.jet()
plt.show()

for idx in range(3):
    plt.figure(figsize = (20, 4))
    plt.scatter(samples[:, 0],
                samples[:, 1],
                c = result_CFD[:, idx],
                s = 2)
    plt.colorbar()
    plt.clim(color_legend[idx])
    plt.xlim((0-L/2, L-L/2))
    plt.ylim((0-H/2, H-H/2))
    plt.tight_layout()
    plt.jet()
plt.show()


for idx in range(3):
    
    plt.figure(figsize = (20, 4))
    plt.scatter(samples[:, 0],
                samples[:, 1],
                c = relative_Err[:, idx],
                s = 2,
                norm=colors.LogNorm(vmin=1e-4,vmax=1),
                cmap='jet')

    plt.colorbar()            
    plt.xlim((0-L/2, L-L/2))
    plt.ylim((0-H/2, H-H/2))
    plt.tight_layout()
    plt.jet()

plt.show()
