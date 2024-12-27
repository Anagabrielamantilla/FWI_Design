import torch
import time
import matplotlib.pyplot as plt
import numpy as np

################################################################################################################
def ricker(f0, time_values, t0=None, a=None):
    """
    Función que genera una onda Ricker.

    http://subsurfwiki.org/wiki/Ricker_wavelet

    Parámetros
    ----------
    f0 : float
        Frecuencia pico para la onda Ricker en kHz.
    time_values : numpy.ndarray
        Valores discretizados de tiempo en ms.
    t0 : float, opcional
        Tiempo central. Si no se proporciona, se calculará como 1 / f0.
    a : float, opcional
        Amplitud de la onda. Si no se proporciona, se tomará como 1.

    Retorna
    ----------
    numpy.ndarray
        La onda Ricker.
    """
    if t0 is None:
        t0 = 1 / f0
    if a is None:
        a = 1
    
    r = np.pi * f0 * (time_values - t0)
    wavelet = a * (1 - 2 * r**2) * np.exp(-r**2)
    
    return wavelet

################################################################################################################
def padvel(v0, nbc, dx):
    """
    Rellena la matriz v0 con bordes replicados.

    Parameters:
        v0 (numpy.ndarray): Matriz original.
        nbc (int): Número de capas de borde a agregar.

    Returns:
        numpy.ndarray: Matriz con bordes rellenados.
    """
    
    # Replicar columnas en los bordes
    v = torch.cat([
        v0[:, 0:1].repeat(1, nbc),  # Replicar primera columna
        v0,
        v0[:, -1:].repeat(1, nbc)   # Replicar última columna
    ], dim=1)
    
    # Replicar filas en los bordes
    v = torch.cat([
        v[0:1, :].repeat(nbc, 1),  # Replicar primera fila
        v,
        v[-1:, :].repeat(nbc, 1)   # Replicar última fila
    ], dim=0)
    
    return v

################################################################################################################

def expand_source(s0, nt):
    """
    Expande el vector s0 para que tenga tamaño nt.

    Parameters:
        s0 (numpy.ndarray): Vector original.
        nt (int): Tamaño deseado.

    Returns:
        numpy.ndarray: Vector expandido.
    """
    nt0 = len(s0)
    if nt0 < nt:
        s = torch.zeros(nt, device=s0.device)
        s[:nt0] = s0
    else:
        s = s0
    
    return s

################################################################################################################


def adjust_sr(coord, dx, nbc):
    """
    Ajusta la posición de la superficie libre usando PyTorch.

    Parameters:
        coord (dict): Diccionario con las coordenadas 'sx', 'sz', 'gx', 'gz'.
        dx (float): Espaciamiento en la malla.
        nbc (int): Número de celdas en el borde.

    Returns:
        tuple: Coordenadas ajustadas (isx, isz, igx, igz).
    """
    isx = torch.round(coord['sx'] / dx).int() + 1 + nbc
    isz = torch.round(coord['sz'] / dx).int() + 1 + nbc
    igx = torch.round(coord['gx'] / dx).int() + 1 + nbc
    igz = torch.round(coord['gz'] / dx).int() + 1 + nbc

    if torch.abs(coord['sz']) < 0.5:
        isz += 1

    igz = igz + (coord['gz'] < 0.5).int()
    
    return isx, isz, igx, igz

################################################################################################################

def a2d_mod_abc22(vel,nbc,dx,nt,dt,s,coord,isFS,movie=False):
    '''
    This is the finite difference modelling for multiple velocity models
    using the acoustic wave equation. This is a python version
    modified by a MATLAB code taken from Center for Subsurface Imaging and 
    Fluid Modeling (CSIM), King Abdullah University of Science and Technology
    found in https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.FD/lab.FD2.8/lab.html
    
    Modified by: Ana Mantilla (anagmd2019@gmail.com)
    
    Parameters
    ----------
    vel : TENSOR
        Velocity model. Expected shape is (# velocity models, nx, nz).
    nbc : INT
        Grid number of boundary.
    dx : INT
        Grid interval.
    nt : INT
        Number of sample.
    dt : FLOAT
        Time interval (s).
    s : ARRAY
        Ricker wavelet.
    isFS : BOOLEAN
        Free surface condition.
    movie : BOOLEAN
        If True the movie propagation will appear.
    shots: TENSOR
        Empty array of shape (batch, minibatch, 1000, 70)
        where batch is number of velocity models and minibatch
        the number of shots per velocity model.

    Returns
    -------
    Output seismogram.
    '''
    ng = len(coord['gx'])
    
    v = padvel(vel, nbc, dx)
    alpha = (v * dt / dx) ** 2 
    beta_dt = (v * dt) ** 2
    s = expand_source(s, nt)
    isx, isz, igx, igz = adjust_sr(coord, dx, nbc)
    
    p1 = torch.zeros_like(v)
    p0 = torch.zeros_like(v)
    nzbc, nxbc = v.shape
    nzp = nzbc - nbc
    nxp = nxbc - nbc
  
    seis = torch.zeros((nt, ng))

    for it in range(nt):
        p = 2 * p1 - p0 + alpha * (
        torch.roll(p1, 1, dims=0) + torch.roll(p1, -1, dims=0) +
        torch.roll(p1, 1, dims=1) + torch.roll(p1, -1, dims=1) - 4 * p1)
        
        p[isz, isx] = p[isz, isx] + beta_dt[isz, isx] * s[it]

        if isFS:
            p[nbc, :] = 0.0
            p[nbc - 1:nbc - 4:-1, :] = -p[nbc + 1:nbc + 5]
            
        # Snapshot
        if movie == True:
            if it % 20 == 0:
                plt.clf()
                ax = plt.gca()
                im1 = ax.imshow(p[nbc:nzp, nbc:nxp].detach().cpu().numpy(), cmap='gray', extent=[0,vel.shape[0],vel.shape[1],0])
                plt.title(f'Time={it * dt:.2f}s')
                im2 = ax.imshow(v[nbc:nzp, nbc:nxp].detach().cpu().numpy(), cmap='jet', alpha=0.2, extent=[0,vel.shape[0],vel.shape[1],0])
                plt.pause(0.1)
            
        for ig in range(ng):
            seis[it, ig] = p[igz[ig], igx[ig]] 
            
            
        p0 = p1.clone()
        #p0 = p1
        p1 = p.clone()
        #p1 = p
    
    return seis

################################################################################################################

def z_score_normalize(data, nt):
    t = torch.arange(nt).view(-1, 1)
    data = (t**0.01)*data
    mean = torch.mean(data)
    std = torch.std(data)
    normalized_data = (data - mean) / std
    normalized_data = torch.tanh(normalized_data)
    return normalized_data

################################################################################################################

def acoustic_operator(vel, nbc, dx, nt, dt, s, isFS, movie, nx, nz, coord_sx, coord_gx):
    
    '''
    This is the finite difference modelling for multiple velocity models using the 
    acoustic wave equation with order of accuracy 2 in space and time. This is a 
    python version modified by a MATLAB code taken from Center for Subsurface Imaging 
    and Fluid Modeling (CSIM), King Abdullah University of Science and Technology found in 
    https://csim.kaust.edu.sa/files/SeismicInversion/Chapter.FD/lab.FD2.8/lab.html
    
    Modified by: Ana Mantilla (anagmd2019@gmail.com)
    
    Parameters
    ----------
    vel : TENSOR
        Velocity model. Expected shape is (# velocity models, nx, nz).
    nbc : INT
        Grid number of boundary.
    dx : INT
        Grid interval.
    nt : INT
        Number of sample.
    dt : FLOAT
        Time interval (s).
    s : ARRAY
        Ricker wavelet.
    isFS : BOOLEAN
        Free surface condition.
    movie : BOOLEAN
        If True the movie propagation will appear.si hice mi propio forward operator en 
    nshots : INT
        Number of shots.
    nx : INT
        Number of cells in x direction
    nz : INT
        Number of cells in z direction
    shots: ARRAY
        Empty array of shape (batch, minibatch, 1000, 70)
        where batch is number of velocity models and minibatch
        the number of shots per velocity model.

    Returns
    -------
    * Seismogram: normalize between -1 and 1. 
    * Coordinates: dictionary with sz,gx,gz and last sx
    * Shot position: list with x position of the sources during modeling
    '''
    
    shots = torch.zeros((vel.shape[0],  len(coord_sx), nt, nx))

    for v in range(vel.shape[0]):
        for idx, i in enumerate(coord_sx):
            coord = {}
            coord['sz'] = torch.tensor(5.0)
            coord['gx'] = coord_gx
            coord['gz'] = torch.full((len(coord['gx']),), 5)
            coord['sx'] = torch.tensor([i])

            shots[v, idx, :, :] = a2d_mod_abc22(torch.squeeze(vel[v, :, :]), nbc, dx, nt, dt, s, coord, isFS, movie)
            print('shot at x=', str(i.item()), ' meters', 'velocity model number:', str(v))
            
    # NORMALIZE DATA BETWEEN -1 AND 1
    shots = z_score_normalize(shots, nt)    

    return shots, coord
    #return shots

