import matplotlib.pyplot as plt
import numpy as np

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
    def __init__(self,leg:str,l1:float=1.0,l2:float=4.0,l3:float=4.0,):
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
        

if __name__ == '__main__':
    #v_max = 0.75 #m/s
    #z0 = 0.85*(l2+l3) #Standing height
    leg1 = Leg3D(leg="right")
    target = np.array([0.5,-1,-6])
    thetas = leg1.inverse_kinematics(target)
    recovered = leg1.forward_kinematics(thetas)
    error = np.linalg.norm(target - recovered[3])
    print(np.rad2deg(thetas))
    print(recovered[3])
    print(error)

