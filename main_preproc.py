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

map_path    = './maps/Julia/'
input_path  = './input/'
output_path = './output/'

input_file  = 'RAW_RAMP_NAME.fits'
output_file = 'PROCESSED_RAMP_NAME'


# Set parameters
Apf2eps     = 7.52
FRN1        = 4
NBFRMIN     = FRN1 + 3
NBLFG       = 4
CR_VARJ     = 120 # 120 for calibration, 300 for I don't know what.
CR_VMRJ     = 2   # 7 in the tool_preproc.py file ?
CR_THRJ     = 15

# Set end of previous acquisition at Nsec before begining of current acquisition
Nsec = 0

# Load maps
PIM_ADU_SAT      = np.ravel(fits.getdata(map_path+'PIM_ADU_SAT.fits'))
PIM_ADU_MAXFIT   = np.ravel(fits.getdata(map_path+'PIM_ADU_MAXFIT.fits'))
PIM_ADU_DYN      = fits.getdata('./maps/dynamique.fits')
PIM_REAL_NONLIN  = np.ravel(fits.getdata(map_path+'PIM_REAL_NONLIN.fits'))


PIM_REAL_SIGFLU = np.nan

PIM_PR_SAT  = fits.open('./maps/carte_persistance.fits')
PIM_PR_SAT  = np.ravel(PIM_PR_SAT[1].data)
PIM_PR_SATB = np.argwhere(PIM_PR_SAT > 0)


PIM_REAL_PPT1 = fits.getdata(map_path+'PIM_REAL_PT1.fits').T
PIM_REAL_PPT2 = fits.getdata(map_path+'PIM_REAL_PT2.fits').T

PIM_REAL_PPA1 = fits.getdata(map_path+'PIM_REAL_PA1.fits').T
PIM_REAL_PPA2 = fits.getdata(map_path+'PIM_REAL_PA2.fits').T

P_LEVEL       = 1
N_PREC        = 3

# del conv, tau, amp

# Load pixel maps
indref = fits.getdata('./maps/PixVerts.fits')
indv   = fits.getdata('./maps/PixViolet.fits')

# import du fichier à étudier et transformation en tableau
RAMP = fits.open(input_path+input_file)
N =  np.shape(RAMP)[0]-1
print('Number of frames to process: ',N)

try:
    T0_RAMP       = datetime.datetime.fromisoformat(RAMP[0].header['HIERARCH ESO DET SEQ UTC']+'00')
    TFIN_PREVRAMP = T0_RAMP - datetime.timedelta(Nsec/3600./24.)
except:
    TFIN_PREVRAMP = 100.
    
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

    # Etape 5 : pixels touchés par un Rayon cosmique et Etape 6 : estimation du signal en adu/fr
    S_ADU, VAR_ADU, POM_FRN_CR, POM_NBF_FIT, Ac, Bc, Cc, Ncc = FitCosmic(Dk, FRN_MAXFIT, FRN1, PIM_REAL_NONLIN, PIM_ADU_DYN, POM_FRN_SAT,
                                                        CR_VARJ, CR_VMRJ, CR_THRJ, NBFRMIN, indv)
    

    # Etape 7: correction signal
    POM_REAL_SIGNAL = S_ADU
    POM_REAL_VAR = VAR_ADU
    
    # Etape 8 : construction de la carte de persistance
    try:
        DT = (T0_RAMP - TFIN_PREVRAMP).seconds
    except:
        DT = 100.
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

    SaveFit(image, 7, ['Signal', 'VarianceSignal', 'CarteCosmiques', 'PremiereFRSat', 'nbframeFit', 'maxfit', 'PERSIST'],
            output_path+output_file, 'ORIGIN', input_path+input_file)
    del(image)
    gc.collect()
    
if "check" in sys.argv:
    X = [1021,1021,1021,1042,17,1025]
    Y = [1008,1013,1016,407,1016,1000]
    pix_check = 2048*(np.array(Y)-1) + np.array(X)-1
    numcas = ['N°'+str(i+1) for i in range(len(X))]
    df  = {'Row':[i+1 for i in range(len(X))],'X':X, 'Y':Y, 'numCas':numcas, 'A':Ac[pix_check], 'B':Bc[pix_check], 'C':Cc[pix_check], 'Delta':Bc[pix_check]**2-4.*Ac[pix_check]*Cc[pix_check], 'Nc':Ncc[pix_check],  'S_ADU':POM_REAL_SIGNAL[pix_check], 'VAR_ADU':POM_REAL_VAR[pix_check], 'POM_NBF_FIT':POM_NBF_FIT[pix_check]}
    tab = tabulate(df, headers='keys', tablefmt='psql')
    
    print('\n\n','\t'+tab.replace('\n','\n\t'),'\n\n')
    
    del(POM_REAL_SIGNAL, POM_NBF_FIT, S_ADU, VAR_ADU, POM_FRN_CR, Ac, Bc, Cc, Ncc)
    
    X,Y,num,A,B,C,Nc,D,S,V,P = [],[],[],[],[],[],[],[],[],[],[]
    with open('./output/2024-02-02-16-54-39-fichierStatsPixels.csv','r') as infile:
        for line in infile.readlines():
            if line[0] == "X":
                pass
            else:
                l = line.rstrip('\n').split(';')
                X.append(float(l[0]))
                Y.append(float(l[1]))
                num.append(l[2])
                A.append(float(l[3]))
                B.append(float(l[4]))
                C.append(float(l[5]))
                Nc.append(float(l[6]))
                D.append(float(l[7]))
                S.append(float(l[8]))
                V.append(float(l[9]))
                P.append(float(l[10]))

    plt.close('all')
    plt.figure()
    comp = abs(np.ravel(np.asarray(np.reshape(POM_REAL_VAR, [2048, 2048])).T[4:-4,4:-4])-V)
    comp[np.where(comp <=0.)[0]] = 1e-10
    plt.hist(np.log10(comp), bins=20)
    plt.yscale('log')
    plt.show()
    problem = np.where(comp >=1e3)[0]
    del(A,B,C,Nc,D,S,V,POM_REAL_SIGNAL)
    gc.collect()
    num = np.asarray(num)
    P = np.asarray(P)
    SAT = np.ravel(np.reshape(POM_FRN_SAT, [2048, 2048]).T[4:-4,4:-4])
