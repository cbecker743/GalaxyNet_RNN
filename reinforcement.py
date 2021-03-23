import tqdm
import numpy as np
import tensorflow as tp
import astropy.cosmology as ac
import scipy.stats as st
import halotools.mock_observables as htmo
import catalogues as cat
from scaling import *
from observations import *


#Load the weights in a numpy array
def get_weights(model, psodict):
    parameters = np.array([])
    modeltype_RNN = psodict['modeltype_RNN']
    if modeltype_RNN == True:
        for layer in model.layers:
            if len(layer.get_weights()) == 2:
                w,b = layer.get_weights()
                parameters = np.append(parameters,w.flatten())
                parameters = np.append(parameters,b.flatten())
            if len(layer.get_weights()) == 3:
                w1, w2, b = layer.get_weights()
                parameters = np.append(parameters,w1.flatten())
                parameters = np.append(parameters,w2.flatten())
                parameters = np.append(parameters,b.flatten())
                
    if modeltype_RNN == False:
        for layer in model.layers:
            if len(layer.get_weights()) == 2:
                w,b = layer.get_weights()
                parameters = np.append(parameters,w.flatten())
                parameters = np.append(parameters,b.flatten())
        
    return parameters

#Set the weights in a model from a numpy array
def set_weights(model, parameters, psodict):
    modeltype_RNN = psodict['modeltype_RNN']
    if modeltype_RNN == True:
        for layer in model.layers:
            if len(layer.get_weights()) == 2:
                n_input  = layer.input_shape[2]
                n_output = layer.output_shape[2] 
                w, parameters = np.split(parameters,[n_input*n_output])
                b, parameters = np.split(parameters,[n_output])
                w = w.reshape(n_input,n_output)
                layer.set_weights([w,b])
            if len(layer.get_weights()) == 3:
                n_input  = layer.input_shape[2] 
                n_output = layer.output_shape[2]
                w1, parameters = np.split(parameters,[n_input*n_output*3])
                w2, parameters = np.split(parameters,[n_output*n_output*3])
                b, parameters = np.split(parameters,[2*n_output*3])
                w1 = w1.reshape(n_input,n_output*3)
                w2 = w2.reshape(n_output,n_output*3)
                b = b.reshape(2,int(b.shape[0]/2))
                layer.set_weights([w1,w2,b])
                
    if modeltype_RNN == False:
        for layer in model.layers:
            if len(layer.get_weights()) == 2:
                n_input  = layer.input_shape[1] #93
                n_output = layer.output_shape[1] #93
                w, parameters = np.split(parameters,[n_input*n_output])
                b, parameters = np.split(parameters,[n_output])
                w = w.reshape(n_input,n_output)
                layer.set_weights([w,b])
        

def compute_statistics(psodict):

    halos              = psodict['halos']
    modeltype_RNN      = psodict['modeltype_RNN']
    galaxies           = psodict['galaxies']
    positions          = psodict['positions']
    mstar_bin_edges    = psodict['mstar_bin_edges']
    redshift_bin_edges = psodict['redshift_bin_edges']
    obssigma0          = psodict['obssigma0']
    obssigmaz          = psodict['obssigmaz']
    zmax_sig           = psodict['zmax_sig']
    ssfrthre           = psodict['ssfrthre']
    H0                 = psodict['H0']
    Om0                = psodict['Om0']
    Lbox               = psodict['Lbox']
    dmstar             = psodict['dmstar']
    ssfrmin            = psodict['ssfrmin']
    wp_redshift        = psodict['wp_redshift']
    wp_mass            = psodict['wp_mass']
    rmin               = psodict['rmin']
    rmax               = psodict['rmax']
    nrbin              = psodict['nrbin']
    
    mstar  = np.array(galaxies[:,0])
    sfr    = np.array(galaxies[:,1])
    if modeltype_RNN == True:
        ascale = np.array(halos[:,0])
    else:
        ascale = np.array(halos[:,7])
    z      = 1./ascale-1.0
    redshifts = np.unique(z)
    num2,_,_ = st.binned_statistic(redshifts, redshifts, 'count', bins=redshift_bin_edges)
    
    sigmaobs = obssigma0 + z * obssigmaz
    sigmaobs[z>=zmax_sig] = obssigma0 + zmax_sig * obssigmaz
    
    #Set minimum SSFR
    index  = np.where(sfr/10.**mstar<ssfrmin)
    sfr[index] = ssfrmin*10.**mstar[index]
    #Add scatter to the SFR
    np.random.seed(seed=42)
    sfrobs = sfr * 10.**np.random.normal(loc=0.0,scale=sigmaobs,size=sfr.size)
    #Add scatter to the stellar mass
    np.random.seed(seed=43)
    mstarobs = np.random.normal(loc=mstar,scale=sigmaobs,size=mstar.size)
    
    ssfr   = sfrobs/10.**mstarobs
    cosmo  = ac.FlatLambdaCDM(H0=H0,Om0=Om0)
    thre   = ssfrthre*1.e-9/cosmo.age([z]).value[0]
    index  = ssfr < thre
    ms_q   = mstarobs[index]
    z_q    = z[index]
    
    if mstarobs.size > 0:
        binned_count    = st.binned_statistic_2d(mstarobs, z,   z,    'count', bins=[mstar_bin_edges,redshift_bin_edges])
        binned_ssfr     = st.binned_statistic_2d(mstarobs, z,   ssfr, 'mean', bins=[mstar_bin_edges,redshift_bin_edges])
        binned_ssfr_sig = st.binned_statistic_2d(mstarobs, z,   ssfr, sigma_mean,  bins=[mstar_bin_edges,redshift_bin_edges])
        
    if ms_q.size > 0:
        binned_count_q  = st.binned_statistic_2d(ms_q,     z_q, z_q,  'count', bins=[mstar_bin_edges,redshift_bin_edges])

    if mstarobs.size > 0:
        if modeltype_RNN == True:  
            smf =[]
            for counter, redshift_bin in enumerate(binned_count.statistic.T):
                smf.append(np.log10(redshift_bin / dmstar / Lbox**3 / num2[counter])) 
            smf=np.array(smf)
            smf_sig = []
            for counter, redshift_bin in enumerate(binned_count.statistic.T):
                smf_sig.append(np.log10(redshift_bin * (1. + 1./np.sqrt(redshift_bin)) / Lbox**3 / dmstar / num2[counter]) - smf[counter])
            smf_sig = np.array(smf_sig) 
        else:
            smf     = np.log10(binned_count.statistic.T / dmstar / Lbox**3.0)
            smf_sig = np.log10(binned_count.statistic.T * (1. + 1./np.sqrt(binned_count.statistic.T)) / Lbox**3 / dmstar) - smf
        smf[binned_count.statistic.T == 0] = -np.inf
        smf_sig[binned_count.statistic.T == 0] = np.inf
              
    else:
        smf     = np.ones((redshift_bin_edges.size-1,mstar_bin_edges.size-1)) * 1.e10
        smf_sig = np.ones((redshift_bin_edges.size-1,mstar_bin_edges.size-1))

    if ms_q.size > 0:
        fq     = binned_count_q.statistic.T/binned_count.statistic.T
        fq_sig = 1./np.sqrt(binned_count_q.statistic.T)
        fq[binned_count_q.statistic.T == 0] = 0.0
    else:
        fq     = np.ones((redshift_bin_edges.size-1,mstar_bin_edges.size-1)) * 1.e10
        fq_sig = np.ones((redshift_bin_edges.size-1,mstar_bin_edges.size-1))

    if mstarobs.size > 0:            
        ssfr     = np.log10(binned_ssfr.statistic.T)
        ssfr_sig = np.log10(binned_ssfr.statistic.T + binned_ssfr_sig.statistic.T) - ssfr
        ssfr[np.isnan(binned_ssfr.statistic.T)]     = -np.inf
        ssfr_sig[np.isnan(binned_ssfr.statistic.T)] = np.inf
    else:
        ssfr     = np.ones((redshift_bin_edges.size-1,mstar_bin_edges.size-1)) * 1.e10
        ssfr_sig = np.ones((redshift_bin_edges.size-1,mstar_bin_edges.size-1))

    if z.size > 0:   
        ssum,_,_  = st.binned_statistic(z, sfrobs, 'sum', bins=redshift_bin_edges) 
        num,_,_   = st.binned_statistic(z, sfrobs, 'count', bins=redshift_bin_edges)
        
        if modeltype_RNN == True:
            csfrd = np.log10(ssum/Lbox**3.0/num2) 
            csfrd_sig = np.log10(ssum/Lbox**3.0/num2 * (1.0 + 1.0 / np.sqrt(num))) - csfrd
        else:
            csfrd     = np.log10(ssum/Lbox**3.0)
            csfrd_sig = np.log10(ssum/Lbox**3.0 * (1.0 + 1.0 / np.sqrt(num))) - csfrd
    else:
        csfrd     = np.ones(redshift_bin_edges.size-1) * 1.e10
        csfrd_sig = np.ones(redshift_bin_edges.size-1)
    
    wp_mod = np.empty([0,nrbin])
    for i,mass_bin in enumerate(wp_mass):
        index   = np.logical_and(np.logical_and(mstarobs >= mass_bin[0], mstarobs < mass_bin[1]), np.round(ascale,2) == 0.91)
        sample  = np.array(positions[index])
        log_rp  = np.linspace(np.log10(rmin),np.log10(rmax),nrbin)
        rp_bins = np.clip(10.**get_bin_edges(log_rp),0.0,Lbox/3.01)
        if sample.shape[0] > 10:
            wp_set  = htmo.wp(sample,rp_bins=rp_bins,pi_max=10./H0*100.0,period=Lbox)
            wp_set[-1:] = 0.0
            wp_set[-2:-1] = 0.0 # cutoff for high rbins
        else:
            wp_set  = np.zeros(rp_bins.size-1)
        wp_mod  = np.concatenate([wp_mod,wp_set.reshape(1,-1)],axis=0)
            
    return smf,smf_sig,fq,fq_sig,ssfr,ssfr_sig,csfrd,csfrd_sig,wp_mod    


def get_chi2(psodict, printchi=False):

    smf   = psodict['smf']
    fq    = psodict['fq']
    csfrd = psodict['csfrd']
    ssfr  = psodict['ssfr']
    wp    = psodict['wp']
    
    smf_mod,smf_sig_mod,fq_mod,fq_sig_mod,ssfr_mod,ssfr_sig_mod,csfrd_mod,csfrd_sig_mod,wp_mod = compute_statistics(psodict=psodict)

    smf_obs     = smf[0]
    smf_sig_obs = smf[1]
    chi2_smf    = (smf_obs-smf_mod)**2.0 / (smf_sig_obs**2.0 + smf_sig_mod**2.0)
    chi2_smf[np.isnan(chi2_smf)] = 0.0
    chi2_smf    = chi2_smf.sum()

    fq_obs     = fq[0]
    fq_sig_obs = fq[1]
    chi2_fq    = (fq_obs-fq_mod)**2.0 / (fq_sig_obs**2.0 + fq_sig_mod**2.0)
    chi2_fq[np.isnan(chi2_fq)] = 0.0
    chi2_fq    = chi2_fq.sum()

    ssfr_obs     = ssfr[0]
    ssfr_sig_obs = ssfr[1]
    chi2_ssfr    = (ssfr_obs-ssfr_mod)**2.0 / (ssfr_sig_obs**2.0 + ssfr_sig_mod**2.0)
    chi2_ssfr[np.isnan(chi2_ssfr)] = 0.0
    chi2_ssfr    = chi2_ssfr.sum()

    csfrd_obs     = csfrd[0]
    csfrd_sig_obs = csfrd[1]
    chi2_csfrd    = (csfrd_obs-csfrd_mod)**2.0 / (csfrd_sig_obs**2.0 + csfrd_sig_mod**2.0)
    chi2_csfrd[np.isnan(chi2_csfrd)] = 0.0
    chi2_csfrd    = chi2_csfrd.sum()

    wp_obs     = wp[1]
    wp_sig_obs = wp[2]
    diff       = wp_obs-wp_mod
    diff[wp_mod==0.0] = 0.0
    chi2_wp    = diff**2.0 / wp_sig_obs**2.0
    chi2_wp[np.isnan(chi2_wp)] = 0.0
    chi2_wp    = chi2_wp.sum()

    chi2 = chi2_smf + chi2_fq + chi2_csfrd + chi2_ssfr + chi2_wp

    if printchi:
        print(chi2,chi2_smf,chi2_fq,chi2_csfrd,chi2_ssfr,chi2_wp)
    
    return chi2


def pso_loss(parameters, psodict):
    modeltype_RNN = psodict['modeltype_RNN']
    if modeltype_RNN == True:
        model              = psodict['model']
        X                  = psodict['X_RNN_input']
        pos                = psodict['pos_RNN_input']
        galaxy_labels_used = psodict['galaxy_labels_used']
        halo_features_used   = psodict['halo_features_used']

        #First attach the parameters to the model weights
        set_weights(model,parameters, psodict)

        #Make a model prediction using the full data set
        y_pred  = model.predict(X, batch_size=8000)
        
        #Remove zerodpadding and Scale the predictions back to galaxy properties
        gal_pred, halo_pred, positions  = cat.data_without_zeropadding_RL2(X, y_pred, galaxy_labels_used, halo_features_used, pos) 

        #Feed the predictions to the dictionary
        psodict['galaxies'] = gal_pred
        psodict['halos'] = halo_pred
        psodict['positions']= positions
        
        #Get the corresponding Chi^2
        try:
            if np.isinf(gal_pred).sum() != 0:
                chi2 = 10**10
            elif np.isnan(gal_pred).sum() != 0:
                chi2 =10**10
            else:
                psodict['galaxies'] = gal_pred
                chi2 = get_chi2(psodict=psodict)
        except:
            chi2 =10**10
            print('ERROR')
        
    if modeltype_RNN == False:
        model              = psodict['model']
        X                  = psodict['X']
        galaxy_labels_used = psodict['galaxy_labels_used']

        #First attach the parameters to the model weights
        set_weights(model,parameters, psodict)

        #Make a model prediction using the full data set
        y_pred  = model.predict(X, batch_size=X.shape[0])

        #Scale the predictions back to galaxy properties
        gal_pred = np.array(get_targets_unscaled(y_pred,galaxy_labels_used))
        try:
            if np.isinf(gal_pred).sum() != 0:
                chi2 = 10**10
            elif np.isnan(gal_pred).sum() != 0:
                chi2 =10**10
            else:
                psodict['galaxies'] = gal_pred
                chi2 = get_chi2(psodict=psodict)
        except:
            chi2 =10**10
            print('ERROR')
    return chi2