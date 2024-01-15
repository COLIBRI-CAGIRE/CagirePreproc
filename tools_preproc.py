import sys
from astropy.io import fits
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import math

pix_check = 2048*(np.array([42,45,47,139,799])-1) + np.array([498,498,498,540,540])-1

def table(fichier, fin):
    fit = fits.open(fichier)
    tab = np.zeros([fin, 4194304])
    for i in range(1, fin + 1):
        tab[i - 1] = np.ravel(fit[i].data)
    return (tab)


def table_simu(fichier, fin):
    fit = fits.open(fichier)
    tab = np.zeros([fin, 4194304])
    f = fit[0].data
    for i in range(0, fin):
        tab[i] = np.ravel(f[i, :, :])
    return (tab)


def PixSat(image, sat, fin, indref):
    # POM_FRN_SAT = carteP

    PixSat = np.argwhere(image[fin - 1, :] >= sat)

    FrameSat = np.sum(np.where(image < sat, 1, 0),axis=0)
    carteP = np.zeros(2048 * 2048)
    carteP[PixSat] = FrameSat[PixSat]+1
    carteP[indref] = 1

    return (carteP)


def PlageFit(image, sat70, N, indref):
    # Trouver les max de chaque pixel et leurs valeurs
    
    indMax = np.sum(np.where(image < sat70, 1, 0), axis=0)
    indMax[indref] = N
    return (indMax)


def tableau3D(fichier, fin):
    fit = fits.open(fichier)
    tab = np.zeros([fin, 2048, 2048])
    for i in range(1, fin + 1):
        a = fit[i].data
        tab[i - 1] = (a).astype(np.int32)
    return (tab)


def tableau3D_simu(fichier, fin):
    fit = fits.open(fichier)
    f = fit[0].data
    tab = np.zeros([fin, 2048, 2048])
    for i in range(0, fin):
        a = f[i, :, :]
        tab[i] = (a).astype(np.int32)
    return (tab)


def correctionC(image):
    # creation d'un masque  pour selectionner les colonnes des pixels de référence
    l = [1, 2, 2045, 2046]

    # print('mean',np.nanmean(image[0, l, 0:64]))

    # on découpe l'image en 32 voies et on calcul la médiane à appliquer sur chaque voie avec le masque
    split = np.split(image[:, l, :], 32, axis=2)  # *masque

    # image[:, l, :] = 0

    '''plt.figure('test image')
    plt.imshow(image[0])
    plt.show()'''

    # print(np.shape(image),np.shape(split))
    Corr = np.nanmean(split, axis=(2, 3))
    # print(np.shape(Corr))

    '''plt.figure('test')
    plt.plot(np.mean(Corr,axis=0))'''

    del split
    Corr = np.transpose(Corr)

    # création d'une cartographie des corrections
    C = np.repeat(Corr, 64, axis=1)
    axe = np.zeros(2048)
    Corr = (axe[:, np.newaxis] + C[:, np.newaxis, :])
    del C
    del axe

    # print(np.shape(Corr),Corr[0,0,0:64])
    '''plt.figure('correctif colonnes diff')
    plt.imshow(Corr[1]-Corr[0], vmin = np.quantile(Corr[1]-Corr[0],0.1),vmax = np.quantile(Corr[1]-Corr[0],0.9))
    plt.title('correctif colonnes')

    plt.figure('correctif colonnes 1')
    plt.imshow(Corr[1] , vmin=np.quantile(Corr[1] , 0.1), vmax=np.quantile(Corr[1] , 0.9))
    plt.title('correctif colonnes')
    plt.figure('correctif colonnes 0')
    plt.imshow(Corr[0], vmin=np.quantile(Corr[0], 0.1), vmax=np.quantile(Corr[0], 0.9))
    plt.title('correctif colonnes')'''

    '''plt.figure('image 40-3')
    plt.imshow(image[1] - image[3], vmin = np.quantile(image[40] - image[3],0.1), vmax = np.quantile(image[40] - image[3],0.9))
    plt.title('image')
    plt.figure('image1')
    plt.imshow(image[1],vmin = np.quantile(image[1],0.1),vmax = np.quantile(image[1],0.9))
    plt.colorbar(label='signal [ADU]')'''

    # image corrigée sur les colonnes
    Colonnes_Cor = image - Corr

    '''plt.figure('image3 corr')
    plt.imshow(Colonnes_Cor[1], vmin=np.quantile(Colonnes_Cor[1], 0.1), vmax=np.quantile(Colonnes_Cor[1], 0.9))
    plt.show()'''
    '''plt.figure('image40 - 3 corr')
    plt.imshow(Colonnes_Cor[40]-Colonnes_Cor[3], vmin=np.quantile(Colonnes_Cor[40]-Colonnes_Cor[3], 0.1), vmax=np.quantile(Colonnes_Cor[40]-Colonnes_Cor[3], 0.9))'''

    del Corr
    del image

    return (Colonnes_Cor)


def correctionL(image, NBLFG):
    # création du masque de selection des pixels de référence
    c = [0, 1, 2, 2045, 2046, 2047]

    # ATTENTION, CETTE FONCTION N'EST ICI PAS ADAPTÉE SI ON CHANGE NBLFG : A AJUSTER

    # image[:, 1:10, c] = 0
    image[:, (20 - NBLFG):(20 + NBLFG + 1), c] = 0
    # print(np.mean(image[0, 1:10, c]))
    # print(np.mean(image[0,(7 - NBLFG):(7 + NBLFG+1), c]))

    '''plt.figure('test image')
    plt.imshow(image[0])
    plt.show()'''

    # création d'un masque et calcul des moyennes des 3 lignes au dessus et au dessous de la ligne considérée
    CorrL = [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 1:10, c], axis=(1, 2)))]

    for k in range(NBLFG + 1, 2047 - NBLFG):
        # on calcul la médiane des pix de ref sur chauqe ligne
        CorrL += [(np.nanmean(image[:, (k - NBLFG):(k + NBLFG + 1), c], axis=(1, 2)))]

    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL += [(np.nanmean(image[:, 2038:2047, c], axis=(1, 2)))]
    CorrL = np.array(CorrL)

    '''plt.figure('test')
    plt.plot(np.mean(CorrL, axis=0))'''

    # création d'une cartographie des correction
    CorrL = CorrL[:, :, np.newaxis]
    Corr = np.repeat(CorrL, 2048, axis=2)
    del CorrL
    CorrL = np.moveaxis(Corr, 1, 0)
    CorrL[np.isnan(CorrL)] = 0

    # print(CorrL[0,2,50])
    # print(CorrL[0, 2, 150])
    # print(CorrL[0, 7, 50])
    # print(CorrL[0, 7, 150])

    '''plt.figure('correctif lignes')
    plt.imshow(CorrL[40] - CorrL[3], vmin=np.quantile(CorrL[40] -CorrL[3], 0.1), vmax=np.quantile(CorrL[40] -CorrL[3], 0.9))
    plt.title('correctif colonnes')

    plt.figure('image 40-3')
    plt.imshow(image[4] - image[3], vmin=np.quantile(image[40] - image[3], 0.1),
               vmax=np.quantile(image[40] - image[3], 0.9))
    plt.title('image')

    plt.figure('image3')
    plt.imshow(image[3], vmin=np.quantile(image[3], 0.1), vmax=np.quantile(image[3], 0.9))'''

    lignes_cor = image - CorrL

    '''plt.figure('image3 corrL')
    plt.imshow(lignes_cor[3], vmin=np.quantile(lignes_cor[3], 0.1), vmax=np.quantile(lignes_cor[3], 0.9))

    plt.figure('image40 - 3 corrL')
    plt.imshow(lignes_cor[40] - lignes_cor[3], vmin=np.quantile(lignes_cor[40] - lignes_cor[3], 0.1),
               vmax=np.quantile(lignes_cor[40] - lignes_cor[3], 0.9))'''
    # plt.show()

    del CorrL

    return (lignes_cor)


def Tableau2DFlat(image, fin):
    tab = np.zeros([fin, 4194304])
    for i in range(0, fin):
        a = image[i, :, :]
        tab[i] = np.ravel(a).astype(np.float32)
    return (tab)


def rampeCDS(image, fin):
    imCDS = np.zeros((fin, 2048 * 2048))
    # on met en première position l'image n°0 corrigée des pixels de ref
    imCDS[0] = image[0]  # on met en première position l'image n°0 corrigée du bias et des pixels de ref
    for i in range(0, fin - 1, 1):
        # on crée la rampe CDS corrigée des pixels de ref
        imCDS[i + 1] = ((image[i + 1]) - (image[i]))

    return (imCDS)


def FitCosmic(image, indSat, d, chemin_alpha, sat, framesat, CR_VARJ, CR_VMRJ, CR_THRJ, NBFRMIN, indv):
    PixaTraiter = indv
    cos = []
    im = image.copy()
    madLim = np.zeros(2048 * 2048)
    Med = np.zeros(2048 * 2048)
    vr = np.zeros(2048 * 2048)
    fr = np.zeros(2048 * 2048)
    F = np.zeros([1, 4194304])  # matrice des résultats du fit
    varO = np.zeros(4194304)  # variance sur l'offset
    carteCos = np.zeros(2048 * 2048)
    POM_NBF_FIT = np.zeros(2048 * 2048)

    # on prend ici un coef de non linéarité à partir d'un fichier de carac  / ou le fichier test pour ratir
    al = fits.getdata(chemin_alpha)
    alpha = np.ravel(al)
    
    Ac, Bc, Cc, Ncc = [np.zeros(2048 * 2048) for i in range(4)]
    
    for i in np.unique(indSat):
        if i - NBFRMIN > 0:
            pix_isat = np.intersect1d(np.argwhere(indSat == i), PixaTraiter)
            pix_isat = pix_isat[:, np.newaxis]
            longueur = (np.arange(d, i)).astype(np.int64)
            # print('i',i,np.shape(im))
            y1 = im[longueur, pix_isat]

            # selection des pixels non eratics et var/flux ok
            v = np.var(y1, axis=1)
            vr[pix_isat[:, 0]] = v

            f = np.mean(y1, axis=1)
            fr[pix_isat[:, 0]] = f

            CR = np.argwhere(v > CR_VARJ)  # 300 #calibration : 120
            g = np.argwhere(v / f > CR_VMRJ)  # 7 #calibration : 2
            indCR = np.intersect1d(g, CR)
            pix = pix_isat[indCR]
            y = im[longueur, pix]

            # calcul de la limite à partir de la mediane et de la MAD
            med = np.median(y, axis=1)
            med2 = med[:, np.newaxis]
            med2 = np.repeat(med2, np.shape(y)[1], axis=1)
            mad = np.median(np.absolute(y - med2), axis=1)
            # print(i, pix[:, 0],np.shape(madLim),np.shape(mad),np.shape(med))
            madLim[pix[:, 0]] = med + CR_THRJ * mad
            Med[pix[:, 0]] = med

            m = madLim[pix[:, 0]]
            m = m[:, np.newaxis]
            m = np.repeat(m, np.shape(y)[1], axis=1)
            
            cr = np.argwhere(y > m)
            p = cr[:, 0]  # pixels impactés
            t = cr[:, 1]  # temps de l'impact
            

            # selection des cosmics : un seul point au dessus de la limite
            uniq, u = np.unique(p, return_counts=True)
            val = uniq[np.argwhere(u == 1)]
            
            uni = np.argwhere(p == val)
            # start = np.argwhere(t != 0) #pas le premier impacté par cosmic

            if np.shape(uni)[0] != 0:

                un = uni[:, 1]  # np.intersect1d(uni[:, 1], start[:, 0])
                cr = cr[un]
                tps = cr[:, 1] + d
                p = cr[:, 0]
                pixel = pix[p]

                # on met le temps d'impact dans la carte des pixels impactés
                carteCos[pixel[:, 0]] = tps

                # on supprime le pixel impacté de la liste des pixels normaux
                pix_isat = np.setdiff1d(pix_isat, pixel)[:, np.newaxis]

                # on parcours les cosmics selectionés
                a0 = np.zeros(len(tps))
                er = np.zeros(len(tps))
                Nbfit = np.zeros(len(tps))
                for c in range(0, len(tps)):
                    med = np.nanmedian(im[:, pixel[c, 0]])
                    if med == 0: med = 1

                    l = round(im[tps[c], pixel[c, 0]] / med)

                    yf = im[longueur, pixel[c, 0]]

                    Al = alpha[pixel[c, 0]]
                    Nc = i - d + 1
                    FRNmaxfit = i + 1 
                    FRN1      = d
                    #A = (Al / 2) * ((N ** 2) + 2 * N * l - N - 2 * l * tps[c])
                    A = (Al / 2) * ( Nc**2 - Nc + 2*Nc*l -2.*tps[c]*l )
                    B = Nc - 1
                    C = - np.sum(yf) + im[tps[c], pixel[c, 0]]
                    delta = B * B - 4 * A * C
                    
                    a0[c] = (-B + np.sqrt(delta)) / (2 * A)

                    A1 = Al * (a0[c] ** 2)
                    A1 = np.repeat(A1, len(longueur))
                    Ek = yf - A1 * longueur
                    er[c] = np.var(Ek) / ((tps[c] - 1 - d + 1) + (i - (tps[c] + 1) + 1))
                    Nbfit[c] = Nc - 1
                    
                    Ac[pixel[c, 0]]  = A
                    Bc[pixel[c, 0]]  = B
                    Cc[pixel[c, 0]]  = C
                    Ncc[pixel[c, 0]] = B
                    
                F[:, pixel[:, 0]] = a0
                varO[pixel[:, 0]] = er
                POM_NBF_FIT[pixel[:, 0]] = Nbfit

            yfit = im[longueur, pix_isat]
            Alpha = alpha[pix_isat]

            A = Alpha[:, 0] * ((d + i) / 2)
            B = 1
            C = - np.sum((yfit), axis=1) / (i - d + 1)

            delta = B * B - 4 * A * C
            
            a0 = (-B + np.sqrt(delta)) / (2 * A)

            A1 = Alpha[:, 0] * (a0 ** 2)
            A1 = np.repeat(A1[:, np.newaxis], len(longueur), axis=1)
            Ek = yfit - A1 * longueur
            er = np.var(Ek, axis=1) / (i - d + 1)
            
            Ac[pix_isat[:, 0]]  = A
            Bc[pix_isat[:, 0]]  = B
            Cc[pix_isat[:, 0]]  = C
            Ncc[pix_isat[:, 0]] = i - d + 1
            
            F[:, pix_isat[:, 0]] = a0
            varO[pix_isat[:, 0]] = er
            POM_NBF_FIT[pix_isat[:, 0]] = i - d + 1

            # print(i,len(pix_isat[:, 0]),POM_NBF_FIT[pix_isat[:, 0]])


        elif i>0: # cas ou l'on a pas assez de points pour faire un calcul mais plus d'une frame à disposition
            pix_isat = np.intersect1d(np.argwhere(indSat == i), PixaTraiter)
            a = framesat[pix_isat]
            pix_isat = pix_isat[:, np.newaxis]
            S_adu = sat[pix_isat] / (framesat[pix_isat] - 0.5)
            F[:, pix_isat[:, 0]] = np.transpose(S_adu)
            Stdev = 0.5 * sat[pix_isat] / (framesat[pix_isat]**2 - framesat[pix_isat])
            varO[pix_isat[:, 0]] = np.transpose(Stdev)**2
            POM_NBF_FIT[pix_isat[:, 0]] = a
            
            Ac[pix_isat[:,0]] = np.nan
            Bc[pix_isat[:,0]] = np.nan
            Cc[pix_isat[:,0]] = np.nan
            Ncc[pix_isat[:,0]] = a
            
        else: # cas où il n'y a qu'une seule frame pour faire le calcul (saturation dès la première frame)
            pix_isat = np.intersect1d(np.argwhere(indSat == i), PixaTraiter)
            pix_isat = pix_isat[:, np.newaxis]
            S_adu = 2 * sat[pix_isat]
            F[:, pix_isat[:, 0]] = np.transpose(S_adu)
            varO[pix_isat[:, 0]] = np.transpose(sat[pix_isat])**2

            POM_NBF_FIT[pix_isat[:, 0]] = 1

            Ac[pix_isat[:,0]] = np.nan
            Bc[pix_isat[:,0]] = np.nan
            Cc[pix_isat[:,0]] = np.nan
            Ncc[pix_isat[:,0]] = 1
    
    B = F[0, :]
    varFlux = varO
    return (B, varFlux, carteCos, POM_NBF_FIT, Ac, Bc, Cc, Ncc)


def CorrectifPersistance(PIM_FRN_SAT, N, DT, amp_1, amp_2, tau_1, tau_2, P_LEVEL, N_PREC):
    TF_RAMP = DT + N * 1.33  # temps à la fin de la rampe

    # persistanceFIN = amp_1 * (1 - np.exp(-TF_RAMP / tau_1)) + amp_2 * (1 - np.exp(-TF_RAMP / tau_2))  # persistance si on avait commencé directement après le premier reset de la rampe prec
    # persistanceDEB = (amp_1 * (1 - np.exp((-DT + 1.33) / tau_1)) + amp_2 * ( 1 - np.exp((-DT + 1.33) / tau_2)))

    persistanceFIN = amp_1 * (1 - np.exp(-TF_RAMP / tau_1)) + amp_2 * (1 - np.exp(
        -TF_RAMP / tau_2))  # persistance si on avait commencé directement après le premier reset de la rampe prec
    persistanceDEB = (amp_1 * (1 - np.exp((-DT) / tau_1)) + amp_2 * (1 - np.exp((-DT) / tau_2)))

    # persistance - persistance accumulée du reset jusqu'au début de l'acquisition
    persistance = persistanceFIN - persistanceDEB

    persistance = np.ravel(persistance)
    # persistance = persistance / (N*1.33) # conversion en e/s pour donner un flux
    persistance = persistance / (10 * N)  # conversion en A/f pour donner un flux compatible avec la carte
    persistance[np.isnan(persistance)] = 0

    Pper = np.intersect1d(np.argwhere(PIM_FRN_SAT * P_LEVEL <= N_PREC), np.argwhere(PIM_FRN_SAT > 0))
    print('Number of pixels impacted by persistence: ', np.shape(Pper)[0])
    map_persistance = np.zeros(2048 * 2048)
    map_persistance[Pper] = persistance[Pper]
    map_persistance[np.isnan(map_persistance)] = 0

    return (map_persistance)


def SaveFit(image, nombreFrame, noms_entete, nom_fit, HEAD, Commentaire,overwrite=True):
    hdr = fits.Header()
    hdr[HEAD] = Commentaire
    primary = fits.PrimaryHDU(header=hdr)

    image_hdu = fits.ImageHDU(image[0], name=noms_entete[0])
    hdul = fits.HDUList([primary, image_hdu])
    hdr = hdul[1].header
    hdr.append('measure')
    hdul[1].header['measure'] = noms_entete[0]

    n = np.shape(image)[0]

    if nombreFrame != 1:
        for i in range(2, n + 1):
            hdul.append(fits.ImageHDU(image[i - 1], name=noms_entete[i-1]))
            hdr = hdul[i].header
            hdr.append('measure')
            hdul[i].header['measure'] = noms_entete[i - 1]

    nom = str(nom_fit) + '.fits'
    hdul.writeto(nom, overwrite=overwrite)


def Save(image, name, HEAD, Commentaire, type):
    hdu = fits.PrimaryHDU()
    hdu.data = image.astype(type)

    for i in range(len(HEAD)):
        hdu.header.set(HEAD[i], Commentaire[i])

    nom = str(name) + '.fits'
    hdu.writeto(nom)
