import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

dde.config.set_random_seed(48)
dde.config.set_default_float('float64')

# Properties
rho = 1
mu = 0.01
u_in = 0.3
L =15 #Domain Size
H=2 #inner domain Size
angle=0 #Attack angle of airfoil

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
#array[:,0]=array[:,0]-0.5

#array= rotate_coordinates(array, angle)

# Boundarys

def boundary_wall(X, on_boundary):
    on_wall = np.logical_and(np.logical_or(np.isclose(X[1], -L/2), np.isclose(X[1], L/2)), on_boundary)
    return on_wall

def boundary_airfoil(X,on_boundary):

  on_airfoil= np.logical_and(np.isclose(array, X,atol=0.01).any(),on_boundary)
  return on_airfoil

def boundary_inlet(X, on_boundary):
    on_cir=np.logical_and(np.isclose(X[0]**2+X[1]**2,(L/2)**2),X[0]<0)

    on_inlet= np.logical_and(on_cir,on_boundary)
    return on_inlet

def boundary_outlet(X, on_boundary):
    return on_boundary and np.isclose(X[0], L)

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

#Geometry & points

geo_Disk= dde.geometry.geometry_2d.Disk([0,0],L/2)
geo_Rec= dde.geometry.geometry_2d.Rectangle([0,-L/2],[L,L/2])

#geo_Disk=dde.geometry.csg.CSGDifference(geo_Disk,geo_Rec)
#geo_Rec=dde.geometry.csg.CSGDifference(geo_Rec,geo_Disk)

geo_boundary=dde.geometry.csg.CSGUnion(geo_Disk,geo_Rec)#-dde.geometry.csg.CSGIntersection(geo_Disk,geo_Rec)

airfoil=dde.geometry.geometry_2d.Polygon(array)

geom = dde.geometry.csg.CSGDifference(geo_boundary, airfoil)

#print(round(np.cos(angle),2))
inner_rec  = dde.geometry.Rectangle([-H+0.5,-H/2],[H+0.5,H/2])

outer_dom  = dde.geometry.CSGDifference(geom, inner_rec)
inner_dom  = dde.geometry.CSGDifference(inner_rec, airfoil)

inner_points = inner_dom.random_points(2000)
outer_points = outer_dom.random_points(10000)

#geom_points=geom.random_points(8000)

domain_points=geo_boundary.random_boundary_points(2000)
airfoil_points=airfoil.random_boundary_points(600)

points = np.append(inner_points, outer_points, axis = 0)
points = np.append(points, domain_points, axis = 0)
points = np.append(points, airfoil_points, axis = 0)



#Boundary condition

bc_wall_u = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component = 0)
bc_wall_v = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component = 1)
#bc_wall_p = dde.DirichletBC(geom, lambda X: 0., boundary_wall, component = 2)

bc_airfoil_u = dde.DirichletBC(geom, lambda X: 0., boundary_airfoil, component = 0)
bc_airfoil_v = dde.DirichletBC(geom, lambda X: 0., boundary_airfoil, component = 1)
#bc_airfoil_p = dde.DirichletBC(geom, lambda X: 0., boundary_airfoil, component = 2)


bc_inlet_u = dde.DirichletBC(geom, lambda X: u_in, boundary_inlet, component = 0)
#bc_inlet_u = dde.DirichletBC(geom, lambda X: u_in, boundary_inlet, component = 0)
bc_inlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_inlet, component = 1)

bc_outlet_p = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component = 2)
#bc_outlet_v = dde.DirichletBC(geom, lambda X: 0., boundary_outlet, component = 1)

bcs=[bc_wall_u, bc_wall_v, bc_airfoil_u ,bc_airfoil_v,  bc_inlet_u, bc_inlet_v, bc_outlet_p]


data = dde.data.PDE(geom,
                    pde,
                    bcs,
                    anchors = points,
                    num_domain = 0,
                    num_boundary = 0,
                    num_test = 6000,
                    train_distribution = 'Hammersley' )

plt.figure(1,figsize = (10,8))
plt.scatter(data.train_x_all[:,0], data.train_x_all[:,1], s = 0.5)
#plt.scatter(inlet_points[:, 0], inlet_points[:, 1], c = fun_u, s = 6.5, cmap = 'jet')
#plt.scatter(inlet_points[:, 0], inlet_points[:, 1], s = 0.5, color='k', alpha = 0.5)
plt.xlabel('x-direction length')
plt.ylabel('Distance from the middle of plates (m)')
plt.show()


layer_size = [2] + [40] *8 + [3]
activation = "tanh"
initializer = "Glorot uniform"

net = dde.maps.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("L-BFGS-B", loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1,1])
model.train_step.optimizer_kwargs = {'options': {'maxcor': 50,
                                                   'ftol': 1.0 * np.finfo(float).eps,
                                             'maxfun':  1,
                                                   'maxiter': 1,
                                                   'maxls': 50}}
#model.train(model_restore_path="0424-30000.pt")
model.restore("0524/0524-60163.pt")  # Replace ? with the exact filename


#dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=50000, maxfun=None, maxls=50)
#model.compile("L-BFGS-B", loss_weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

#losshistory, train_state = model.train(display_every = 100)

#dde.saveplot(losshistory, train_state, issave = False, isplot = True)


samples = geom.random_points(1000000)
result = model.predict(samples)

color_legend = [[0, 0.4], [-0.2, 0.25], [-0.15, 0.15]]

fig,axes= plt.subplots(3,1,figsize=(12,4))
for idx in range(3):
    Z=result[:,idx]
    axes[idx].scatter(samples[:, 0],
                samples[:, 1],
                c = result[:, idx],
                cmap = 'jet',
                s = 2)
    norm=colors.LogNorm(vmin=1e-5, vmax=10)
    sm=plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    fig.colorbar(sm,ax=axes[idx])
    axes[idx].set_xlim((-L/2, L))
    axes[idx].set_ylim((0-L/2, L/2))
    
plt.tight_layout()
plt.show()
