import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

#AXIS CONVENTION: +x forward, +y left, +z up

def rot_matrix_x(rad):
    return np.array([
        [1, 0, 0],
        [0, np.cos(rad), -np.sin(rad)],
        [0, np.sin(rad), np.cos(rad)],
    ])

def rot_matrix_y(rad):
    return np.array([
        [np.cos(rad), 0, np.sin(rad)],
        [0, 1, 0],
        [-np.sin(rad), 0, np.cos(rad)],
    ])

class Leg3D:
    def __init__(self,leg:str,l1:float=0.1,l2:float=0.4,l3:float=0.4,):
        if leg == "left":
            self.leg = 1
        elif leg == "right":
            self.leg = -1
        else:
            raise ValueError("Invalid leg. Expected 'left' or 'right'.")
        if l1 < 0 or l2<=0 or l3<=0:
            raise ValueError("Leg link lengths cannot be negative.")
        self.l1 = l1  #Hip abduction offset
        self.l2 = l2  #Thigh length
        self.l3 = l3  #Shank length
        self.r1_local = np.array([0,self.leg*self.l1,0])
        self.r2_local = np.array([self.l2,0,0])
        self.r3_local= np.array([self.l3,0,0])

    def forward_kinematics(self,thetas):
        theta1 = thetas[0]
        theta2 = thetas[1]
        theta3 = thetas[2]
        rot1 = rot_matrix_x(theta1)
        rot2 = rot_matrix_y(theta2)
        rot3 = rot_matrix_y(theta3)
        rot12 = rot1 @ rot2
        r1_local_leg = rot1 @ self.r1_local
        r2_local_leg = r1_local_leg + rot12 @ self.r2_local
        r3_local_leg = r2_local_leg + (rot12 @ rot3) @ self.r3_local
        return np.array([[0,0,0],r1_local_leg,r2_local_leg,r3_local_leg])

    def inverse_kinematics(self,target,knee_sign:int = 1): 
        if knee_sign not in (-1,1):
            raise ValueError("Invalid knee_sign (valid: 1,-1)")
        x = target[0]
        y = target[1]
        z = target[2]
        eps = 1e-9
        subroot = y**2 + z**2 - self.l1**2
        if subroot < -eps:
            raise ValueError("Out of reach foot cordinates")
        clamped_subroot = np.maximum(subroot, 0.0)
        theta1 = np.arctan2(z,y) + np.arctan2(np.sqrt(clamped_subroot),self.leg*self.l1)
        z_rotated = - y * np.sin(theta1) + z * np.cos(theta1)
        r_sq = x**2 + z_rotated**2
        if r_sq < eps:
            raise ValueError("r_sq is too small, infinite solutions because of joint alignment")
        C = (self.l3**2 - self.l2**2 - r_sq) / (2* self.l2)
    
        subroot = r_sq - C**2
        if subroot< -eps:
            raise ValueError("Out of reach foot cordinates, or too close")
        clamped_subroot = np.maximum(subroot, 0.0)
        theta2 = np.arctan2(x,z_rotated) + np.arctan2(C, knee_sign*np.sqrt(clamped_subroot)) 
        theta3 = np.arctan2(-z_rotated - self.l2 * np.sin(theta2), x - self.l2 * np.cos(theta2)) - theta2
        thetas = np.array([theta1,theta2,theta3])
        thetas_normalized = (thetas + np.pi) % (2 * np.pi) - np.pi
        return thetas_normalized


M = np.array([
    [  1,   0,   0,   0,   0,   0,  0],
    [ -6,   6,   0,   0,   0,   0,  0],
    [ 15, -30,  15,   0,   0,   0,  0],
    [-20,  60, -60,  20,   0,   0,  0],
    [ 15, -60,  90, -60,  15,   0,  0],
    [ -6,  30, -60,  60, -30,   6,  0],
    [  1,  -6,  15, -20,  15,  -6,  1]
], dtype=float)


#phi = t/T mod 1
def foot_trajectory(phi = 0,v = 0.75, T = 0.6, beta = 0.5, H = 0.1, z0 = 0.68, y0 = 0.1):
    if phi < 0 or phi >=1:
        raise ValueError(f"Phi is out of bounds phi:{phi}")
    if beta <= 0 or beta >=1:
        raise ValueError(f"Beta is out of bounds beta:{beta}")
    if T <= 0:
        raise ValueError(f"T is out of bounds T:{T}")
    s_stance = phi / beta
    s_swing = (phi-beta)/(1-beta)
    T_swing = (1-beta)*T
    L = v*beta*T #step length
    if phi>beta:
        #Control Points
        P0 = np.array([0,0])
        P1 = np.array([-v*T_swing/6,0])
        P2 = np.array([-v*T_swing/3,0])
        P3 = np.array([L/2,16*H/5])
        P4 = np.array([L+v*T_swing/3,0])
        P5 = np.array([L+v*T_swing/6,0])
        P6 = np.array([L,0])
        P_i = np.array([P0,P1,P2,P3,P4,P5,P6])
        S = np.array([s_swing**k for k in range(7)])
        local_cords = S @ M @ P_i
    else:
        local_cords = np.array([-s_stance*L + L,0])
    normalized_cords = np.array([local_cords[0],0,local_cords[1]]) + np.array([-L/2,y0,-z0])
    return normalized_cords


def update(frame, cords, artist):
    current = cords[frame]
    xs, ys, zs = current[:, 0], current[:, 1], current[:, 2]
    artist.set_data_3d(xs,ys,zs)

if __name__ == '__main__':
    v = 0.75 #m/s
    l1=0.1
    l2=0.4
    l3=0.4
    z0 = 0.85*(l2+l3) #Standing height
    leg1 = Leg3D(leg="right",l1=l1,l2=l2,l3=l3)

    error = 0
    N = 200
    phis = np.linspace(0,1,N,endpoint=False)
    targets = np.empty((N, 3))
    thetas = np.empty((N, 3))
    cordinates = np.empty((N, 4 ,3))
    for i in range(N):
        targets[i] = foot_trajectory(phi=phis[i],v=v,z0=z0,y0=leg1.leg*leg1.l1)
        thetas[i] = leg1.inverse_kinematics(targets[i])
        cordinates[i] = leg1.forward_kinematics(thetas[i])
        error = error + np.linalg.norm(targets[i] - cordinates[i][3])


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_box_aspect([1,1,1])
    (artist,) = ax.plot([], [], [], marker="o")
    ax.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-1, 0))
    ax.plot(targets[:,0], targets[:,1], targets[:,2], linestyle="--", alpha=0.5)


    X, Y = np.meshgrid([-0.5, 0.5], [-0.5, 0.5])
    Z = np.full_like(X, -z0, dtype=float)
    ax.plot_surface(X, Y, Z, alpha=0.2, color="black")


    anim = FuncAnimation(fig, update, frames=N, fargs=(cordinates, artist), interval=17, blit=False)
    plt.show()
