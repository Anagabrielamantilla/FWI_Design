#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 18:39:52 2024

@author: ana.mantilla@correo.uis.edu.co 
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from acoustic_operator import *

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PART A: LOAD VELOCITY MODEL

vel_true = np.load('model1.npy')
vel_true = vel_true[:1,0,:,:]
vel_true = vel_true[np.newaxis, :, :]
vel_true = torch.from_numpy(vel_true)
print('vel_true shape is ', vel_true.shape)

# PART B: ADJUST VELOCITY MODEL TO 100X100

vel_true = F.interpolate(vel_true, size=(100, 100), mode='bicubic', align_corners=False)
vel_true = vel_true.squeeze(0)
#plt.imshow(vel_true[10,:,:])
#plt.colorbar( )
print('vel_true shape is ', vel_true.shape)

# PART C: SET ACQUISITION PARAMETERS

nbc =160; 
dx = 1; 
sx= 20; 
coord_gx = torch.arange(0, 100, dx); 
coord_sx = torch.arange(0, 100, sx);
nt = 1000;  
dt = 1e-4; 
freq = 12; 
s = ricker(freq, torch.linspace(0, 1, steps=nt)); 
isFS = False
movie = False #change to True if you want to watch the propagation  
nx = len(coord_gx)
nz = nx; 

#plt.plot(s)

courant = ((vel_true.mean()**2)*(dt**2))/dx**2
limit = 1/(np.sqrt(2))

if courant <=limit:
    print('Se cumple criterio de estabilidad de Courant')
else:
    print('Courant is', courant, 'Modificar delta t y delta x')
    
# PART D: PROPAGATE THE SEISMIC WAVEFIELD 

start = time.time()
seis, coord = acoustic_operator(vel_true, nbc, dx, nt, dt, s, isFS, movie, nx, nz, coord_sx, coord_gx)
end = time.time()
print('Total time of seismic modeling is ', end-start, 'seconds')

print('seis shape is ', seis.shape)

# PART E: PLOT THE RESULTING SHOTS

plt.imshow(seis[0,0,:,:], cmap='seismic', aspect='auto') 
plt.colorbar()
