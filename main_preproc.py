import sys
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import math
import datetime
import gc
import pandas as pd
from tools_preproc import *
from tabulate import tabulate



parameters = {'axes.labelsize': 18, 'axes.titlesize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18}
plt.rcParams.update(parameters)

map_path    = './maps/'
input_path  = './input/'
output_path = './output/'

input_file  = 'Ref_H-15F.fits'
output_file = 'Ref_H-15F_processed'


# Set parameters
Apf2eps     = 7.52
FRN1        = 4
NBFRMIN     = 7
NBLFG       = 4
CR_VARJ     = 120 # 120 for calibration, 300 for I don't know what.
CR_VMRJ     = 2   # 7 in the tool_preproc.py file ?
CR_THRJ     = 15

# Set end of previous acquisition at Nsec before begining of current acquisition
Nsec = 0

# Load maps
PIM_ADU_SAT     = fits.getdata(map_path+'sat98pct.fits')
PIM_ADU_MAXFIT  = fits.getdata(map_path+'sat70pct.fits')
PIM_ADU_DYN     = fits.getdata(map_path+'Dynamique_1V.fits')
PIM_REAL_NONLIN = map_path+'gamma_1V.fits'
P               = np.ravel(fits.getdata(map_path+'gamma_1V.fits'))

PIM_REAL_SIGFLU = np.nan


PIM_PR_SAT  = fits.open(map_path+'carte_persistance.fits')
PIM_PR_SAT  = np.ravel(PIM_PR_SAT[1].data)
PIM_PR_SATB = np.argwhere(PIM_PR_SAT > 0)

# Directly replace 0's with 1's after opening the map
conv = fits.getdata(map_path+'conv_2FW.fits') + 1

tau = fits.getdata(map_path+'tau_2FW.fits')
PIM_REAL_PPT1 = conv * tau[0]
PIM_REAL_PPT2 = conv * tau[1]

"""
PIM_REAL_PPT1 = 15
PIM_REAL_PPT2 = 150
"""

amp = fits.getdata(map_path+'amp_2FW.fits')
PIM_REAL_PPA1 = conv * amp[0]
PIM_REAL_PPA2 = conv * amp[1]

P_LEVEL       = 1
N_PREC        = 3

del conv, tau, amp

# Load pixel maps
indref = fits.getdata(map_path+'PixVerts.fits')
indv   = fits.getdata(map_path+'PixViolet.fits')

# import du fichier à étudier et transformation en tableau
RAMP = fits.open(input_path+input_file)
N =  np.shape(RAMP)[0]-1
print('Number of frames to process: ',N)
T0_RAMP       = datetime.datetime.fromisoformat(RAMP[0].header['HIERARCH ESO DET SEQ UTC']+'00')

TFIN_PREVRAMP = T0_RAMP - datetime.timedelta(Nsec/3600./24.)

Bk = table(input_path+input_file, N)

#Ek = rampeCDS(Bk, N)
gc.collect()

if N > NBFRMIN:
    # Etape 2 : pixels saturés et domaine de calcul du SIGNAL
    POM_FRN_SAT = PixSat(Bk, PIM_ADU_SAT, N, indref)
    FRN_MAXFIT = PlageFit(Bk, PIM_ADU_MAXFIT, N, indref)


    # Etape 3: construction de la rampe corrigee
    B3D = tableau3D(input_path+input_file, N)

    t_temps = np.shape(Bk)[0]
    del Bk
    colones_cor = correctionC(B3D)
    del B3D
    lignes_cor = correctionL(colones_cor, NBLFG)
    del colones_cor
    Ck = Tableau2DFlat(lignes_cor, t_temps)
    del lignes_cor
    # Etape 4: construction de la rampe differentielle
    Dk = rampeCDS(Ck, N)
    print(np.unique(FRN_MAXFIT))
    print(np.unique(POM_FRN_SAT))
    # Etape 5 : pixels touchés par un Rayon cosmique et Etape 6 : estimation du signal en adu/fr
    S_ADU, VAR_ADU, POM_FRN_CR, POM_NBF_FIT, Ac, Bc, Cc, Ncc = FitCosmic(Dk, FRN_MAXFIT, FRN1, PIM_REAL_NONLIN, PIM_ADU_DYN, POM_FRN_SAT,
                                                        CR_VARJ, CR_VMRJ, CR_THRJ, NBFRMIN, indv)
    del Dk

    # Etape 7: correction signal – Flux et conversion en e - / s
    POM_REAL_SIGNAL = S_ADU  # * Apf2eps
    POM_REAL_VAR = VAR_ADU  # * Apf2eps**2
    
    # Etape 8 : construction de la carte de persistance
    DT = (T0_RAMP - TFIN_PREVRAMP).seconds
    POM_REAL_PERSIG = CorrectifPersistance(PIM_PR_SATB, N, DT, PIM_REAL_PPA1, PIM_REAL_PPA2, PIM_REAL_PPT1,
                                           PIM_REAL_PPT2, P_LEVEL, N_PREC)

    # Etape 9 : mise a jour des variables utilisées par le preproc
    PIM_FRN_SAT = POM_FRN_SAT
    
    
    # Etape 10 : stockage fichier fits
    POM_REAL_SIGNAL[np.isnan(POM_REAL_SIGNAL)] = 0

    image = np.zeros([7, 2048,2048])
    image[0, :,:] = np.reshape(POM_REAL_SIGNAL, [2048, 2048])
    image[1, :,:] = np.reshape(np.sqrt(POM_REAL_VAR), [2048, 2048])
    image[2, :,:] = np.reshape(POM_FRN_CR, [2048, 2048])
    image[3, :,:] = np.reshape(POM_FRN_SAT, [2048, 2048])
    image[4, :,:] = np.reshape(POM_NBF_FIT, [2048, 2048])
    image[5, :,:] = np.reshape(FRN_MAXFIT, [2048, 2048])
    image[6, :,:] = np.reshape(POM_REAL_PERSIG, [2048, 2048])
    
    if "plot" in sys.argv:
        plt.figure('Processed image')
        plt.imshow(image[0], vmin=np.quantile(image[0], 0.1), vmax=np.quantile(image[0], 0.9))
        plt.show()

    SaveFit(image, 7, ['Signal', 'VarianceSignal', 'CarteComiques', 'PremiereFRSat', 'nbframeFit', 'maxfit', 'PERSIST'],
            output_path+output_file, 'ORIGIN', input_path+input_file)

if "check" in sys.argv:
    pix_check = 2048*(np.array([42,45,47,139,799])-1) + np.array([498,498,498,540,540])-1
    X = [498,498,498,540,540]
    Y = [42,45,47,139,799]
    numcas = ['N°'+str(i+1) for i in range(len(X))]

    df  = {'Row':[i+1 for i in range(len(X))],'X':X, 'Y':Y, 'numCas':numcas, 'A':Ac[pix_check], 'B':Bc[pix_check], 'C':Cc[pix_check], 'Delta':Bc[pix_check]**2-4.*Ac[pix_check]*Cc[pix_check], 'Nc':Ncc[pix_check],  'S_ADU':POM_REAL_SIGNAL[pix_check], 'VAR_ADU':POM_REAL_VAR[pix_check], 'POM_NBF_FIT':POM_NBF_FIT[pix_check]}
    tab = tabulate(df, headers='keys', tablefmt='psql')
    
    print('\n\n','\t'+tab.replace('\n','\n\t'),'\n\n')



