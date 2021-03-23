import numpy as np
import h5py as hdf5
import tensorflow as tf
import scipy.stats as st
import scipy.optimize as so
import scipy.stats as st
import catalogues as cat
import astropy.cosmology as ac
import matplotlib.pyplot as plt
from matplotlib import colors
import re
from scaling import *
from reinforcement import get_bin_edges

#Plot DNN main sequence at some redshift
def plot_main_sequence(
    X,
    y,
    redshift,
    iscale,
    halo_features_used,
    galaxy_labels_used,
    H0=70.0,
    Om0=0.3,
    plot_obs=True,
    axis=[7.0,12.9,-6.20,1.8],
    fs=22,
    lw=3,
    frac=None,
    save=False,
    epoch=None
    ):
    
    fig = plt.figure(figsize=(12.0,8.0))
    
    plot_main_sequence_panel(fig,X,y,redshift,iscale,halo_features_used,galaxy_labels_used,H0=H0,Om0=Om0,plot_obs=plot_obs,axis=axis,fs=fs,lw=lw,frac=frac,nxpanel=1,nypanel=1,ipanel=1)
    
    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.99, hspace=0.0, wspace=0.0)

    if save:
        if epoch is None:
            plt.savefig('images/MS.jpg')
        else:
            plt.savefig('images/MS.{:05d}.jpg'.format(epoch))
        plt.close(fig)
    else:
        plt.show()

#Plot DNN main sequence at some redshift
def plot_main_sequence_panel(
    fig,
    X,
    y,
    redshift,
    iscale,
    halo_features_used,
    galaxy_labels_used,
    H0=70.0,
    Om0=0.3,
    plot_obs=True,
    axis=[7.0,12.9,-6.20,1.8],
    fs=22,
    lw=3,
    frac=None,
    nxpanel=1,
    nypanel=1,
    ipanel=1,
    barposition=[0.92,0.18,0.04,0.65],
    modelname=None,
    showredshift=True,
    seed=42
    ):
    
    #Unscale the prediction
    gal_pred       = np.array(get_targets_unscaled(y,galaxy_labels_used))
    halos_unscaled = np.array(get_features_unscaled(X,halo_features_used))

    np.random.seed(seed=seed)
    isample = np.random.permutation(gal_pred.shape[0]) # Array of randomly permuted indices
    gal_pred = gal_pred[isample]
    halos_unscaled = halos_unscaled[isample]
    
    if frac is not None:
        isample        = np.random.choice(gal_pred.shape[0],int(gal_pred.shape[0]*frac))
        gal_pred       = gal_pred[isample]
        halos_unscaled = halos_unscaled[isample]

    mstar = gal_pred[:,0]
    sfr   = gal_pred[:,1]
    apeak = halos_unscaled[:,4]
    z     = 1./halos_unscaled[:,iscale]-1.
    iselect = np.logical_and(halos_unscaled[:,iscale] >= 1./(redshift+1.)-0.01, halos_unscaled[:,iscale] < 1./(redshift+1.)+0.01)
    mstar = mstar[iselect] # iselect is an array contiang of boolean values which removes all False values
    sfr   = sfr[iselect]
    z     = z[iselect]
    apeak = apeak[iselect]
    if plot_obs:
        #Set minimum SSFR
        ssfrmin = 1.e-12
        sfr[np.where(sfr/10.**mstar<ssfrmin)] = ssfrmin*10.**mstar[np.where(sfr/10.**mstar<ssfrmin)]
    sfr   = np.log10(sfr)
    #Add scatter to the SFR to get observed SFR
    if plot_obs:
        cosmo = ac.FlatLambdaCDM(H0=H0,Om0=Om0)
        tcosmic = cosmo.age([z]).value[0]
        thre  = np.log10(0.3/tcosmic)-9.0
        ssfr  = sfr - mstar
        sig   = 0.2*(np.tanh(1.0*(thre-ssfr.flatten()))*0.5+0.5)+0.1
        np.random.seed(seed=42)
        rangau = np.random.normal(loc=0.0,scale=sig,size=sfr.size)
        sfrobs = sfr+rangau
        #Add scatter to the stellar mass to get observed mstar
        np.random.seed(seed=42)
        rangau = np.random.normal(loc=0.0,scale=0.1,size=mstar.size)
        mstarobs = mstar+rangau

    ax  = plt.subplot(nypanel,nxpanel,ipanel)
    ax.axis(axis)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    
    if showredshift:
        ax.annotate('$z={:.2f}$'.format(redshift), xy=(1.0-0.05, 0.05), xycoords='axes fraction', size=fs*1.1, ha='right', va='bottom')

    if modelname is not None:
        ax.annotate(modelname, xy=(0.05, 1.0-0.05), xycoords='axes fraction', size=fs*1.1, ha='left', va='top')

    if plot_obs:
        ln = ax.scatter(mstarobs,sfrobs,s=2,c=apeak*(redshift+1.0),cmap=plt.cm.jet_r, vmin=0.0, vmax=1.0)
    else:
        ln = ax.scatter(mstar,sfr,s=2,c=apeak*(redshift+1.0),cmap=plt.cm.jet_r, vmin=0.0, vmax=1.0)
    ax.set_xlabel('$\log (m_* / \mathrm{M}_\odot)$', size = fs)
    ax.set_ylabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})$', size = fs)
    
    if ipanel==1:
        barpos=fig.add_axes(barposition)
        cbar = plt.colorbar(ln,cax=barpos, ticks=[0.0,0.2,0.4,0.6,0.8,1.0])
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label('$a_\mathrm{p} \, / \, a$', fontsize=fs)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    
    ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
        
    if (ipanel-1) % nxpanel  > 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
    if int((ipanel-1)/nxpanel) < nypanel-1:
        ax.set_xticklabels([])
        ax.set_xlabel('')

#Define a callback class to save a MS plot after each epoch
class Callback_plot_MS(tf.keras.callbacks.Callback):
    
    def __init__(self, model, X):
        self.model = model
        self.X     = X
    
    def on_epoch_end(self, epoch, logs={}):
        save_main_sequence(self.model,self.X,epoch=epoch,z=0.0,plot_obs=False)

#This is a wrapper that first makes a prediction and then plots it
def save_main_sequence(model,X,epoch=None,z=0.0,H0=70.0,Om0=0.3,plot_obs=False,save=True):
    y_pred  = model.predict(X, batch_size=X.shape[0])
    plot_main_sequence(X,y_pred,z,X.shape[1]-1,halo_features_used,galaxy_labels_used,H0=H0,Om0=Om0,plot_obs=plot_obs,save=save,epoch=epoch)

#Plot DNN main sequence at some redshift
def plot_shmr(
    X,
    y,
    redshift,
    iscale,
    halo_features_used,
    galaxy_labels_used,
    axis=[10.0,15.0,7.0,12.0],
    vmin=-12.1,
    vmax=-8.5,
    fs=22,
    lw=3,
    frac=None,
    save=False,
    epoch=None):
    
    fig = plt.figure(figsize=(12.0,8.0))
    
    plot_shmr_panel(fig,X,y,redshift,iscale,halo_features_used,galaxy_labels_used,axis=axis,fs=fs,lw=lw,frac=frac,nxpanel=1,nypanel=1,ipanel=1)
    
    plt.subplots_adjust(left=0.12, right=0.99, bottom=0.13, top=0.99, hspace=0.0, wspace=0.0)
    
    if save:
        if epoch is None:
            plt.savefig('images/SHMR.jpg')
        else:
            plt.savefig('images/SHMR.{:05d}.jpg'.format(epoch))
        plt.close(fig)
    else:
        plt.show()

#Plot DNN main sequence at some redshift
def plot_shmr_panel(
    fig,
    X,
    y,
    redshift,
    iscale,
    halo_features_used,
    galaxy_labels_used,
    axis=[10.0,15.0,7.0,12.0],
    vmin=-12.1,
    vmax=-8.5,
    fs=22,
    lw=3,
    frac=None,
    nxpanel=1,
    nypanel=1,
    ipanel=1,
    barposition=[0.92,0.18,0.04,0.65],
    modelname=None,
    showredshift=True,
    seed=42
    ):
    
    #Unscale the prediction
    gal_pred       = np.array(get_targets_unscaled(y,galaxy_labels_used))
    halos_unscaled = np.array(get_features_unscaled(X,halo_features_used))

    np.random.seed(seed=seed)
    isample = np.random.permutation(gal_pred.shape[0])
    gal_pred = gal_pred[isample]
    halos_unscaled = halos_unscaled[isample]
    
    if frac is not None:
        isample        = np.random.choice(gal_pred.shape[0],int(gal_pred.shape[0]*frac))
        gal_pred       = gal_pred[isample]
        halos_unscaled = halos_unscaled[isample]
    
    mhalo = halos_unscaled[:,1]
    mstar = gal_pred[:,0]
    sfr   = gal_pred[:,1]
    z     = 1./halos_unscaled[:,iscale]-1.
    iselect = np.logical_and(halos_unscaled[:,iscale] >= 1./(redshift+1.)-0.01, halos_unscaled[:,iscale] < 1./(redshift+1.)+0.01)
    mhalo = mhalo[iselect]
    mstar = mstar[iselect]
    sfr   = sfr[iselect]
    z     = z[iselect]
    
    ax  = plt.subplot(nypanel,nxpanel,ipanel)
    ax.axis(axis)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    
    if showredshift:
        ax.annotate('$z={:.2f}$'.format(redshift), xy=(1.0-0.05, 0.05), xycoords='axes fraction', size=fs*1.1, ha='right', va='bottom')

    if modelname is not None:
        ax.annotate(modelname, xy=(0.05, 1.0-0.05), xycoords='axes fraction', size=fs*1.1, ha='left', va='top')

    ln = ax.scatter(mhalo,mstar,s=3,c=np.log10(sfr)-mstar,cmap=plt.cm.jet_r, vmin=vmin, vmax=vmax)
    ax.set_xlabel('$\log (M_\mathrm{p} / \mathrm{M}_\odot)$', size = fs)
    ax.set_ylabel('$\log (m_* / \mathrm{M}_\odot)$', size = fs)
    
    if ipanel==1:
        barpos=fig.add_axes(barposition)
        cbar = plt.colorbar(ln,cax=barpos, ticks=[-12.,-11.,-10.,-9.])
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label('sSFR', fontsize=fs)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

    ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
        
    if (ipanel-1) % nxpanel  > 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
    if int((ipanel-1)/nxpanel) < nypanel-1:
        ax.set_xticklabels([])
        ax.set_xlabel('')


#Define a callback class to save a SHMR plot after each epoch
class Callback_plot_SHMR(tf.keras.callbacks.Callback):
    
    def __init__(self, model, X):
        self.model = model
        self.X     = X
    
    def on_epoch_end(self, epoch, logs={}):
        save_shmr(self.model,self.X,epoch=epoch,z=0.0)

#This is a wrapper that first makes a prediction and then plots it
def save_shmr(model,X,epoch=None,z=0.0,save=True):
    y_pred  = model.predict(X, batch_size=X.shape[0])
    plot_shmr(X,y_pred,z,X.shape[1]-1,halo_features_used,galaxy_labels_used,save=save,epoch=epoch)

#Plot label vs prediction
def compare_input_prediction(y,y_pred,galaxy_labels_used,lw=3,fs=22,axis1=[7.0,12.3,7.0,12.3],axis2=[1.e-6,1.e2,1.e-6,1.e2],file=None):

    gal_input = np.array(get_targets_unscaled(y,galaxy_labels_used))
    gal_pred  = np.array(get_targets_unscaled(y_pred,galaxy_labels_used))
    
    mstar  = np.linspace(axis1[0],axis1[1],1000)
    sfr    = np.linspace(axis2[0],axis2[1],1000)
    
    ml,_,_ = st.binned_statistic(gal_input[:,0], gal_input[:,0], 'mean', bins=10)
    mp,_,_ = st.binned_statistic(gal_input[:,0], gal_pred[:,0], 'mean', bins=10)
    ma,_,_ = st.binned_statistic(gal_input[:,0], gal_pred[:,0], sigma, bins=10)
    sl,_,_ = st.binned_statistic(np.log10(gal_input[:,1]), np.log10(gal_input[:,1]), 'mean', bins=10)
    sp,_,_ = st.binned_statistic(np.log10(gal_input[:,1]), np.log10(gal_pred[:,1]), 'mean', bins=10)
    sa,_,_ = st.binned_statistic(np.log10(gal_input[:,1]), np.log10(gal_pred[:,0]), sigma, bins=10)

    fig = plt.figure(figsize=(12.0,6.0))

    ax = plt.subplot(1,2,1)
    ax.axis(axis1)
    ax.plot(gal_input[:,0],gal_pred[:,0],'k,')
    ax.plot(mstar,mstar,'b-',lw=2*lw) 
    ax.errorbar(ml,mp,ma,marker='x',color='red',ms=14,mec='red',linestyle='',mew=lw,elinewidth=lw-1, capsize=(12)/2, zorder=11)
    ax.set_xlabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{label}$', fontsize = fs)
    ax.set_ylabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{prediction}$', fontsize = fs)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)


    ax = plt.subplot(1,2,2)
    ax.axis(axis2)
    ax.plot(np.log10(gal_input[:,1]),np.log10(gal_pred[:,1]),'k,')
    ax.plot(sfr,sfr,'b-',lw=2*lw)
    ax.errorbar(sl,sp,sa,marker='x',color='red',ms=14,mec='red',mew=lw,elinewidth=lw-1, capsize=(12)/2, zorder=11)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.set_xlabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})_\mathrm{label}$', fontsize = fs)
    ax.set_ylabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})_\mathrm{prediction}$', fontsize = fs)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.99, wspace=0.0, hspace=0.0)
    
    if file is not None:
        plt.savefig(file)
    
    plt.show()
    

def compare_input_prediction_RNN(y_vali, y_pred_vali,
                                 y_test, y_pred_test,
                                 galaxy_labels_used,
                                 lw=3,fs=22,
                                 axis1=[7.0,12.3,7.0,12.3],
                                 axis2=[-6.5,3.5,-6.5,3.5],
                                 file=None):  
    
    gal_input_vali = np.array(get_features_unscaled(y_vali, galaxy_labels_used))
    gal_pred_vali = np.array(get_features_unscaled(y_pred_vali, galaxy_labels_used))
    gal_input_test = np.array(get_features_unscaled(y_test, galaxy_labels_used))
    gal_pred_test = np.array(get_features_unscaled(y_pred_test, galaxy_labels_used))
    
    mstar_vali  = np.linspace(axis1[0],axis1[1],1000)
    sfr_vali    = np.linspace(axis2[0],axis2[1],1000)
    ml_vali,_,_ = st.binned_statistic(gal_input_vali[:,0], gal_input_vali[:,0], 'mean', bins=10)
    mp_vali,_,_ = st.binned_statistic(gal_input_vali[:,0], gal_pred_vali[:,0], 'mean', bins=10)
    ma_vali,_,_ = st.binned_statistic(gal_input_vali[:,0], gal_pred_vali[:,0], sigma, bins=10)
    sl_vali,_,_ = st.binned_statistic(np.log10(gal_input_vali[:,1]), np.log10(gal_input_vali[:,1]), 'mean', bins=10)
    sp_vali,_,_ = st.binned_statistic(np.log10(gal_input_vali[:,1]), np.log10(gal_pred_vali[:,1]), 'mean', bins=10)
    sa_vali,_,_ = st.binned_statistic(np.log10(gal_input_vali[:,1]), np.log10(gal_pred_vali[:,1]), sigma, bins=10)
    
    mstar_test  = np.linspace(axis1[0],axis1[1],1000)
    sfr_test    = np.linspace(axis2[0],axis2[1],1000)
    ml_test,_,_ = st.binned_statistic(gal_input_test[:,0], gal_input_test[:,0], 'mean', bins=10)
    mp_test,_,_ = st.binned_statistic(gal_input_test[:,0], gal_pred_test[:,0], 'mean', bins=10)
    ma_test,_,_ = st.binned_statistic(gal_input_test[:,0], gal_pred_test[:,0], sigma, bins=10)
    sl_test,_,_ = st.binned_statistic(np.log10(gal_input_test[:,1]), np.log10(gal_input_test[:,1]), 'mean', bins=10)
    sp_test,_,_ = st.binned_statistic(np.log10(gal_input_test[:,1]), np.log10(gal_pred_test[:,1]), 'mean', bins=10)
    sa_test,_,_ = st.binned_statistic(np.log10(gal_input_test[:,1]), np.log10(gal_pred_test[:,1]), sigma, bins=10)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(20,20),
                        gridspec_kw={'hspace': 0.2, 'wspace': 0.2})
    
    
    ax1.axis(axis1)
    ax1.plot(gal_input_vali[:,0],gal_pred_vali[:,0],'k,')
    ax1.plot(mstar_vali,mstar_vali,'b-',lw=2*lw) 
    ax1.errorbar(ml_vali,mp_vali,ma_vali,marker='x',color='red',ms=14,mec='red',linestyle='',mew=lw,elinewidth=lw-1, capsize=(12)/2, zorder=11)
    ax1.set_xlabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{label}$', fontsize = fs)
    ax1.set_ylabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{prediction}$', fontsize = fs)
    ax1.text(7.5, 11.5,'Stellar_mass_validation', fontsize=25, bbox=dict(facecolor='dodgerblue', alpha=0.7))
    ax1.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax1.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax1.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(lw)
        
    ax2.axis(axis2)
    ax2.plot(np.log10(gal_input_vali[:,1]),np.log10(gal_pred_vali[:,1]),'k,')
    ax2.plot(sfr_vali,sfr_vali,'b-',lw=2*lw)
    ax2.errorbar(sl_vali,sp_vali,sa_vali,marker='x',color='red',ms=14,mec='red',mew=lw,elinewidth=lw-1, capsize=(12)/2, zorder=11)
    ax2.yaxis.set_label_position('right')
    ax2.text(-5.5, 2,'SFR_validation', fontsize=25, bbox=dict(facecolor='dodgerblue', alpha=0.7))
    ax2.yaxis.tick_right()
    ax2.set_xlabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})_\mathrm{label}$', fontsize = fs)
    ax2.set_ylabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})_\mathrm{prediction}$', fontsize = fs)
    ax2.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax2.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax2.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(lw)
        
    ax3.axis(axis1)   
    ax3.plot(gal_input_test[:,0],gal_pred_test[:,0],'k,')
    ax3.plot(mstar_test,mstar_test,'b-',lw=2*lw) 
    ax3.errorbar(ml_test,mp_test,ma_test,marker='x',color='red',ms=14,mec='red',linestyle='',mew=lw,elinewidth=lw-1, capsize=(12)/2, zorder=11)
    ax3.text(7.5, 11.5,'Stellar_mass_test', fontsize=25, bbox=dict(facecolor='dodgerblue', alpha=0.7))
    ax3.set_xlabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{label}$', fontsize = fs)
    ax3.set_ylabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{prediction}$', fontsize = fs)
    ax3.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax3.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax3.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(lw)
        
    ax4.axis(axis2)
    ax4.plot(np.log10(gal_input_test[:,1]),np.log10(gal_pred_test[:,1]),'k,')
    ax4.plot(sfr_test,sfr_test,'b-',lw=2*lw)
    ax4.errorbar(sl_test,sp_test,sa_test,marker='x',color='red',ms=14,mec='red',mew=lw,elinewidth=lw-1, capsize=(12)/2, zorder=11)
    ax4.yaxis.set_label_position('right')
    ax4.text(-5.5, 2,'SFR_test', fontsize=25, bbox=dict(facecolor='dodgerblue', alpha=0.7))
    ax4.yaxis.tick_right()
    ax4.set_xlabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})_\mathrm{label}$', fontsize = fs)
    ax4.set_ylabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})_\mathrm{prediction}$', fontsize = fs)
    ax4.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax4.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax4.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax4.spines[axis].set_linewidth(lw)
    
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.99, wspace=0.0, hspace=0.0)
    
    if file is not None:
        plt.savefig(file,dpi=100, bbox_inches = 'tight',
    pad_inches = 0.1)
    
    plt.show()

    
def plot_history(training,fs=22,lw=3,ymin=0.1,ymax=1.0):

    fig = plt.figure(figsize=(12.0,8.0))
    
    xmin = 0
    xmax = len(training['loss'])

    ax = plt.subplot(111)

    #Plot the training history
    ax.axis([xmin,xmax,ymin,ymax])
    ax.semilogy(training['val_loss'], 'r-',label='Validation Loss')
    ax.semilogy(training['loss'],'b-',label='Training Loss')

    ax.set_xlabel('Epoch',size=fs)
    ax.set_ylabel('Loss',size=fs)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([xmin,xmax])
    ax.set_xticks(np.arange(0,xmax,20))
    ax.set_xticks(np.arange(0,xmax,10),minor=True)
    ylow = int(10.**(-np.floor(np.log10(ymin)))*ymin+1.0)*10.**(np.floor(np.log10(ymin)))
    yupp = int(10.**(-np.floor(np.log10(ymax)))*ymax+1.0)*10.**(np.floor(np.log10(ymax)))
    dy   = 10.**(np.floor(np.log10(ymin))) 
    yticks = np.arange(ylow,yupp,dy)
    ax.set_yticks(yticks)
    formatstring = '{:.'+'{:d}'.format(int(np.log10(1./dy)))+'f}'
    ax.set_yticklabels([formatstring.format(s) for s in yticks])
    ax.set_yticks(np.arange(ymin,ymax,0.01),minor=True)
    ax.set_yticklabels([],minor=True)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    plt.legend(fontsize=fs*0.8,frameon=False,loc='upper right')
    plt.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.99, hspace=0.0, wspace=0.0)
    

def plot_history2(training,fs=22,lw=3,ymin=0.1,ymax=1.0):

    fig = plt.figure(figsize=(12.0,8.0))
    
    xmin = 0
    xmax = len(training['loss'])

    ax = plt.subplot(111)

    #Plot the training history
    ax.axis([xmin,xmax,ymin,ymax])
    ax.plot(training['val_loss'], 'darkorange',label='Validation Loss', linewidth=1.8)
    ax.semilogy(training['loss'],'dodgerblue',label='Training Loss', linewidth=1.8)

    ax.set_xlabel('Epochs',size=fs*1.3)
    ax.set_ylabel('Loss',size=fs*1.3)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([xmin,xmax])
    ax.set_xticks(np.arange(0,xmax,20))
    ax.set_xticks(np.arange(0,xmax,10),minor=True)
    ax.set_yticks([0.002,0.004,0.006,0.008,0.01,0.02, 0.03, 0.04])
    ax.set_yticks([],minor=True)
    ax.set_yticklabels([0.002,0.004,0.006,0.008,0.01,0.02, 0.03, 0.04])
    ax.set_xticklabels(np.arange(0,500,20),fontsize=14)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    plt.legend(fontsize=fs*1.2,frameon=True,loc='upper right')
    plt.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.99, hspace=0.0, wspace=0.0)
    plt.tight_layout()


def plot_smf(universe,redshifts,stellar_masses,average,model=None,compare=None,smf_huge=None,nx=None,ny=None,fs=22,lw=3,ms=12,plotfile=None):

    #Define plot components
    mark=[".","o","v","^","<",">","s","p","P","*","h","H","+","x","X","D","d"]
    colors=['blue','green','red','purple','olive','brown','gold','deepskyblue','lime','orange','navy']
    
    #Open SMF group
    smf = universe['SMF']
    smfkeys = [key for key in smf.keys()]
    #print(smfkeys)

    #Plot the SMF sets
    plt.figure(figsize=(16.0,6.8))

    xmin = 7.1
    xmax = 12.4
    ymin = -5.9
    ymax = -0.4

    zrange = get_bin_edges(redshifts)
    zrange = np.array(zrange)
    zmean  = 0.5*(zrange[:-1]+zrange[1:])
    nzbin  = zmean.size
    
    if ny is None:
        ny     = np.int(np.floor(np.sqrt(nzbin)))
    if nx is None:
        nx     = np.int(np.ceil(nzbin/ny))
        
    #Open Data group
    smfdata = smf['Data']
    smfdatakeys = [key for key in smfdata.keys()]

    #Open Set compound
    smfset = smf['Sets']
    smfsetzmin = smfset['Redshift_min']
    smfsetzmax = smfset['Redshift_max']
    smfsetzmean= 0.5*(smfsetzmin+smfsetzmax)
     
    #Go through each data set and make a subplot for each
    i = 0
    for iy in range(0,ny):
        for ix in range(0,nx):
            if i < zmean.size:
                #Make subplot for this redshift bin
                ax = plt.subplot(ny, nx, i+1)
                ax.axis([xmin, xmax, ymin, ymax])
                #Initialise the sets
                iset = 0
                #Go through all sets
                for setnum in smfdatakeys:
                    #If set is in right redshift range for this subplot
                    if (smfsetzmean[iset] >= zrange[i] and smfsetzmean[iset] < zrange[i+1]):
                        #Load set
                        smfset = smfdata[setnum]
                        x = smfset['Stellar_mass']
                        y = smfset['Phi_observed']
                        s = smfset['Sigma_observed']
                        s[s>1.0] = 0.0
                        ax.errorbar(x, y, yerr=s, marker=mark[iset % len(mark)], ls='none', color=colors[iset % len(colors)])
                    iset += 1 

                ax.errorbar(stellar_masses, average[0][i], yerr=average[1][i], marker='o', ls='none', color='black', ms=ms, elinewidth=lw, capsize=ms/2, mew=lw, label='Observed Mean')
                start = np.nonzero(average[0,i] > -np.inf)[0][0]
                if compare is not None:
                    ax.plot(stellar_masses[start:],compare[i,start:],'c-',lw=lw+2,zorder=9,label='Emerge')
                if model is not None:
                    ax.plot(stellar_masses[start:],model[i,start:],'r-',lw=lw+2,zorder=10,label='RNN + RL')
                if smf_huge is not None:
                    if smf_huge.shape[0] > i:
                        ax.plot(smf_huge[i,0,:],smf_huge[i,1,:],'g-',lw=lw+2,zorder=10,label='HugeMDPL')

                    
                if redshifts[i] >= 1.0:
                    ax.annotate('$z \sim {:4.0f}$'.format(redshifts[i]), xy=(1.0-0.1, 1.0-0.1), xycoords='axes fraction', size=fs, ha='right', va='top')
                else:
                    ax.annotate('$z \sim {:4.1f}$'.format(redshifts[i]), xy=(1.0-0.1, 1.0-0.1), xycoords='axes fraction', size=fs, ha='right', va='top')

                if i == 0:
                    ax.legend(fontsize=fs*0.7,frameon=False,loc='lower left',markerscale=0.5)
                
                ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
                ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
                ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(lw)

                if (ix > 0):
                    ax.tick_params(labelleft=False)
                else:
                    if (iy==1):
                        ax.set_ylabel(r'                         $\log_{10}(\Phi \, / \, \mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1})$', size=fs)
                ax.set_yticks(np.arange(int(ymin),ymax,2.0),minor=False)
                ax.set_yticks(np.arange(int(ymin),ymax,0.5),minor=True)
                if (iy == 0):
                    ax.tick_params(labelbottom=False)
                else:
                    if (ix==2):
                        ax.set_xlabel(r'$\log_{10}(m_* / \mathrm{M}_{\odot})$', size=fs)
                ax.set_xticks(np.arange(int(xmin)+1.0,xmax,2.0),minor=False)
                ax.set_xticks(np.arange(int(xmin)+0.5,xmax,0.5),minor=True)
                i += 1
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.13, top=0.99, hspace=0.0, wspace=0.0)

    if plotfile is not None:
        plt.savefig(plotfile, dpi=100)

    plt.show()


def plot_fq(universe,redshifts,stellar_masses,average,model=None,compare=None,nx=None,ny=None,fs=22,lw=3,ms=12,plotfile=None):

    #Define plot components
    mark=[".","o","v","^","<",">","s","p","P","*","h","H","+","x","X","D","d"]
    colors=['blue','green','red','purple','olive','brown','gold','deepskyblue','lime','orange','navy']
    
    #Open FQ group
    fq = universe['FQ']
    fqkeys = [key for key in fq.keys()]

    #Plot the FQ sets
    plt.figure(figsize=(16.0,4.0))

    xmin = 8.2
    xmax = 11.9
    ymin = 0.0
    ymax = 1.0

    zrange = get_bin_edges(redshifts)
    zrange = np.array(zrange)
    zmean  = 0.5*(zrange[:-1]+zrange[1:])
    nzbin  = zmean.size
    
    if ny is None:
        ny     = np.int(np.floor(np.sqrt(nzbin)))
    if nx is None:
        nx     = np.int(np.ceil(nzbin/ny))
        
    #Open Data group
    fqdata = fq['Data']
    fqdatakeys = [key for key in fqdata.keys()]

    #Open Set compound
    fqset = fq['Sets']
    fqsetzmin = fqset['Redshift_min']
    fqsetzmax = fqset['Redshift_max']
    fqsetzmean= 0.5*(fqsetzmin+fqsetzmax)
     
    #Go through each data set and make a subplot for each
    i = 0
    for iy in range(0,ny):
        for ix in range(0,nx):
            if i < zmean.size:
                #Make subplot for this redshift bin
                ax = plt.subplot(ny, nx, i+1)
                ax.axis([xmin, xmax, ymin, ymax])
                #Initialise the sets
                iset = 0
                #Go through all sets
                for setnum in fqdatakeys:
                    #If set is in right redshift range for this subplot
                    if (fqsetzmean[iset] >= zrange[i] and fqsetzmean[iset] < zrange[i+1]):
                        #Load set
                        fqset = fqdata[setnum]
                        x = fqset['Stellar_mass']
                        y = fqset['Fq_observed']
                        s = fqset['Sigma_observed']
                        #s[s>1.0] = 0.0
                        ax.errorbar(x, y, yerr=s, marker=mark[iset % len(mark)], ls='none', color=colors[iset % len(colors)])
                    iset += 1 

                ax.errorbar(stellar_masses, average[0][i], yerr=average[1][i], marker='o', ls='none', color='black', ms=ms, elinewidth=lw, capsize=ms/2, mew=lw, label='Observed Mean')
                start = np.nonzero(average[0,i] > -np.inf)[0][0]
                if compare is not None:
                    ax.plot(stellar_masses[start:],compare[i,start:],'c-',lw=lw+2,zorder=9,label='Emerge')
                if model is not None:
                    ax.plot(stellar_masses[start:],model[i,start:],'r-',lw=lw+2,zorder=10,label='RNN + RL')

                if redshifts[i] >= 1.0:
                    if i<4:
                        ax.annotate('$z \sim {:4.0f}$'.format(redshifts[i]), xy=(0.1, 1.0-0.1), xycoords='axes fraction', size=fs, ha='left', va='top')
                    else:
                        ax.annotate('$z \sim {:4.0f}$'.format(redshifts[i]), xy=(0.1, 1.0-0.5), xycoords='axes fraction', size=fs, ha='left', va='top')
                else:
                    ax.annotate('$z \sim {:4.1f}$'.format(redshifts[i]), xy=(0.1, 1.0-0.1), xycoords='axes fraction', size=fs, ha='left', va='top')

                if i == 4:
                    ax.legend(fontsize=fs*0.7,frameon=False,loc='upper left',markerscale=0.5)
                
                ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
                ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
                ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(lw)

                if (ix > 0):
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(r'$f_\mathrm{q}$', size=fs)
                ax.set_yticks(np.arange(int(ymin),ymax,0.2),minor=False)
                ax.set_yticks(np.arange(int(ymin),ymax,0.1),minor=True)
                ax.set_xlabel(r'$\log_{10}(m_* / \mathrm{M}_{\odot})$', size=fs)
                ax.set_xticks(np.arange(int(xmin)+1.0,xmax,1.0),minor=False)
                ax.set_xticks(np.arange(int(xmin)+0.2,xmax,0.2),minor=True)
                i += 1
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.22, top=0.99, hspace=0.0, wspace=0.0)

    if plotfile is not None:
        plt.savefig(plotfile, dpi=100)

    plt.show()



def plot_ssfr(universe,redshifts,stellar_masses,average,model=None,compare=None,nx=None,ny=None,fs=22,lw=3,ms=12,plotfile=None):

    #Define plot components
    mark=[".","o","v","^","<",">","s","p","P","*","h","H","+","x","X","D","d"]
    colors=['blue','green','red','purple','olive','brown','gold','deepskyblue','lime','orange','navy']
    
    #Open SSFR group
    ssfr = universe['SSFR']
    ssfrkeys = [key for key in ssfr.keys()]

    #Plot the SSFR sets
    plt.figure(figsize=(16,4))

    xmin = 1.0
    xmax = 10.0
    ymin = -11.8
    ymax = -6.6

    mrange = get_bin_edges(stellar_masses)
    mrange = np.array(mrange)
    mmean  = 0.5*(mrange[:-1]+mrange[1:])
        
    if ny is None:
        ny     = np.int(np.floor(np.sqrt(nzbin)))
    if nx is None:
        nx     = np.int(np.ceil(nzbin/ny))

    #Open Data group
    ssfrdata = ssfr['Data']
    ssfrdatakeys = [key for key in ssfrdata.keys()]
     
    #Go through each data set and make a subplot for each
    i = 0
    for iy in range(0,ny):
        for ix in range(0,nx):
            if i < stellar_masses.size:
                #Make subplot for this redshift bin
                ax = plt.subplot(ny, nx, i+1, xscale='log')
                ax.axis([xmin, xmax, ymin, ymax])
                #Initialise the sets
                iset = 0
                #Go through all sets
                for setnum in ssfrdatakeys:
                    #Load set
                    ssfrset = ssfrdata[setnum]
                    z = ssfrset['Redshift']
                    x = ssfrset['Stellar_mass']
                    y = ssfrset['Ssfr_observed']
                    s = ssfrset['Sigma_observed']

                    index = np.logical_and(x >= mrange[i], x < mrange[i+1])
                    z = z[index]
                    y = y[index]
                    s = s[index]
                    
                    ax.errorbar(z+1, y, yerr=s, marker=mark[iset % len(mark)], ls='none', color=colors[iset % len(colors)])
                    iset += 1 

                ax.errorbar(redshifts+1, average[0,:,i], yerr=average[1,:,i], marker='o', ls='none', color='black', ms=ms, elinewidth=lw, capsize=ms/2, mew=lw, label='Observed Mean')
                if compare is not None:
                    ax.plot(redshifts+1,compare[:,i],'c-',lw=lw+2,zorder=9,label='Emerge')
                if model is not None:
                    ax.plot(redshifts+1,model[:,i],'r-',lw=lw+2,zorder=10,label='RNN + RL')
                
                ax.annotate('$\log_{10}( m_* / \mathrm{M}_\odot) \sim '+'{:4.1f}$'.format(mmean[i]), xy=(0.05, 1.0-0.05), xycoords='axes fraction', size=fs, ha='left', va='top')
                if i == 0:
                    ax.legend(fontsize=fs*0.7,frameon=False,loc='lower right',markerscale=0.5)
                
                if (ix > 0):
                    ax.tick_params(labelleft=False)
                else:
                    ax.set_ylabel(r'$\log_{10}(\mathrm{sSFR} \; / \; yr^{-1})$', size=fs)
                if (iy > 0):
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel(r'$z$', size=fs)
                ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
                ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
                ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)
                for axis in ['top','bottom','left','right']:
                    ax.spines[axis].set_linewidth(lw)
                    
                ax.set_yticks(np.arange(int(ymin),ymax,1.0),minor=False)
                ax.set_yticks(np.arange((ymin),ymax,0.2),minor=True)
                ax.set_xticks(np.arange(1.0,xmax+1.0,1.0),minor=False)
                ax.set_xticklabels(['0','1','2','','4','','6','','','',])                    
                i += 1
    plt.subplots_adjust(left=0.08, right=0.99, bottom=0.22, top=0.99, hspace=0.0, wspace=0.0)

    if plotfile is not None:
        plt.savefig(plotfile, dpi=100)

    plt.show()


def plot_csfrd(universe,redshifts,average,model=None,compare=None,fs=22,lw=3,ms=12,plotfile=None):
    
    #Define plot components
    mark=[".","o","v","^","<",">","s","p","P","*","h","H","+","x","X","D","d"]
    colors=['blue','green','red','purple','olive','brown','gold','deepskyblue','lime','orange','navy']

    #Open CSFRD group
    csfrd = universe['CSFRD']
    #Open Data group
    csfrddata = csfrd['Data']
    csfrddatakeys = [key for key in csfrddata.keys()]

    #Plot the CSFRD sets
    plt.figure(figsize=(8,6))

    xmin = 1.0
    xmax = 15.0
    ymin = -3.9
    ymax = -0.4

    #Open plot
    ax = plt.subplot(111, xscale="log")
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_xlabel(r'$z$', size=fs)
    ax.set_ylabel(r'$\log_{10}( {\rho}_* \, / \, \mathrm{M}_{\odot} \, \mathrm{yr}^{-1} \, \mathrm{Mpc}^{-3})$', size=fs)
    ax.set_xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
    ax.set_xticklabels(['0','1','2','3','4','5','6',' ','8',' ','',' ','12',' '])
    ax.set_yticks(np.arange(int(ymin),ymax,1.0),minor=False)
    ax.set_yticks(np.arange((ymin),ymax,0.2),minor=True)
    
    #Go through each data set and plot each point
    i = 0
    for setnum in csfrddatakeys:
        csfrdset = csfrddata[setnum]
        x = csfrdset['Redshift']+1
        y = csfrdset['Csfrd_observed']
        s = csfrdset['Sigma_observed']
        ax.errorbar(x, y, yerr=s, marker=mark[i % len(mark)], ls='none', color=colors[i % len(colors)])
        i += 1

    ax.errorbar(redshifts+1,average[0],yerr=average[1], marker='o', ls='none', color='black', ms=ms, elinewidth=lw, capsize=ms/2, mew=lw, label='Observed Mean')
    if compare is not None:
        ax.plot(redshifts+1,compare,'c-',lw=lw+2,zorder=9,label='Emerge')
    if model is not None:
        ax.plot(redshifts+1,model,'r-',lw=lw+2,zorder=10,label='RNN + RL')
    ax.legend(fontsize=fs,frameon=False,loc='lower left',markerscale=0.5)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)

    plt.subplots_adjust(left=0.13, right=0.99, bottom=0.13, top=0.99, hspace=0.0, wspace=0.0)

    if plotfile is not None:
        plt.savefig(plotfile, dpi=100)

    plt.show()
    

def plot_wp(universe,rad=None,model=None,compare=None,nx=None,ny=None,fs=22,lw=3,ms=12,plotfile=None):

    #Plot the WP sets
    plt.figure(figsize=(16,4))

    xmin = 0.002
    xmax = 180.0
    ymin = 0.2
    ymax = 40000.0

    #Open Data group
    wp = universe['Clustering']
    sets = wp['Sets']
    wpdata = wp['Data']
    wpdatakeys = [key for key in wpdata.keys()]
    del wpdatakeys[0]

    #Define the number of subplots per axis
    nsets = len(wpdatakeys)
    if ny is None:
        ny     = np.int(np.floor(np.sqrt(nsets)))
    if nx is None:
        nx     = np.int(np.ceil(nsets/ny))
    
    #Go through each data set and make a subplot for each
    i = 0
    for iy in range(0,ny):
        for ix in range(0,nx):
            ax = plt.subplot(ny, nx, i+1, xscale="log", yscale="log")
            ax.axis([xmin, xmax, ymin, ymax])
            if (i < nsets):
                wpset = wpdata[wpdatakeys[i]]
                xo = wpset['Radius'][:-2]
                yo = wpset['Wp_observed'][:-2]
                so = wpset['Sigma_observed'][:-2]
#                 xo = wpset['Radius']
#                 yo = wpset['Wp_observed']
#                 so = wpset['Sigma_observed']
                ym = wpset['Wp_model']
                sm = wpset['Sigma_model']
                ax.errorbar(xo, yo, yerr=so, marker='o', ls='none', color='black', ms=ms, elinewidth=lw, capsize=ms/2, mew=lw, label='Observed Mean')
#                 if compare is not None:
#                     if i==0:
#                         ax.plot(rad,compare[i],'c-',lw=lw+2,zorder=9,label='Emerge')
#                     if i==1:
#                         ax.plot(rad,compare[i],'c-',lw=lw+2,zorder=9,label='Emerge')
#                     if i==2:
#                         ax.plot(rad[2:],compare[i,2:],'c-',lw=lw+2,zorder=9,label='Emerge')
#                     if i==3:
#                         ax.plot(rad[4:],compare[i,4:],'c-',lw=lw+2,zorder=9,label='Emerge')
#                 if model is not None and rad is not None:
#                     #ax.plot(rad[model[i]>0],model[i,model[i]>0],'r-',lw=lw,zorder=10,label='GalaxyNet + RL')
#                     if i==0:
#                         ax.plot(rad,model[i],'r-',lw=lw+2,zorder=10,label='GalaxyNet + RL')
#                     if i==1:
#                         ax.plot(rad,model[i],'r-',lw=lw+2,zorder=10,label='GalaxyNet + RL')
#                     if i==2:
#                         ax.plot(rad[2:],model[i,2:],'r-',lw=lw+2,zorder=10,label='GalaxyNet + RL')
#                     if i==3:
#                         ax.plot(rad[4:],model[i,4:],'r-',lw=lw+2,zorder=10,label='GalaxyNet + RL')
                if compare is not None:
                    if i==0:
                        ax.plot(rad[:-2],compare[i,:-2],'c-',lw=lw+2,zorder=9,label='Emerge')
                    if i==1:
                        ax.plot(rad[:-2],compare[i,:-2],'c-',lw=lw+2,zorder=9,label='Emerge')
                    if i==2:
                        ax.plot(rad[2:][:-2],compare[i,2:][:-2],'c-',lw=lw+2,zorder=9,label='Emerge')
                    if i==3:
                        ax.plot(rad[4:][:-2],compare[i,4:][:-2],'c-',lw=lw+2,zorder=9,label='Emerge')
                if model is not None and rad is not None:
                    #ax.plot(rad[model[i]>0],model[i,model[i]>0],'r-',lw=lw,zorder=10,label='GalaxyNet + RL')
                    if i==0:
                        ax.plot(rad[:-2],model[i,:-2],'r-',lw=lw+2,zorder=10,label='RNN + RL')
                    if i==1:
                        ax.plot(rad[:-2],model[i,:-2],'r-',lw=lw+2,zorder=10,label='RNN + RL')
                    if i==2:
                        ax.plot(rad[2:][:-2],model[i,2:][:-2],'r-',lw=lw+2,zorder=10,label='RNN + RL')
                    if i==3:
                        ax.plot(rad[4:][:-2],model[i,4:][:-2],'r-',lw=lw+2,zorder=10,label='RNN+ RL')


                ax.annotate('${:4.2f} < \log m_* < {:4.2f}$'.format(sets['Minimum_Mass'][i+1],sets['Maximum_Mass'][i+1]), xy=(0.05, 1.0-0.05), xycoords='axes fraction', size=fs*0.8, ha='left', va='top')
            if (ix > 0):
                ax.tick_params(labelleft=False)
            else:
                ax.set_ylabel(r'$w_p \, / \, \mathrm{Mpc}$', size=fs)
            ax.set_yticks([1.,10.,100.,1000.,10000.0])
            ax.set_yticklabels(['$10^0$','$10^1$','$10^2$','$10^3$','$10^4$'])
            if (iy > 0):
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r'$r_p \, / \, \mathrm{Mpc}$', size=fs)
                ax.set_xticks([0.01,0.1,1.0,10.,100.])
                ax.set_xticklabels(['0.01','0.1','1','10',''])
            if i==3:
                ax.legend(fontsize=fs*0.8,frameon=False,loc='lower left',markerscale=0.5)
            ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
            ax.tick_params(width=lw*1.0, length=3*lw, which='major', direction='in', pad=fs/2)
            ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(lw)
            i += 1
    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.21, top=0.99, hspace=0.0, wspace=0.0)

    if plotfile is not None:
        plt.savefig(plotfile, dpi=100)

    plt.show()





#Plot DNN main sequence at some redshift
def plot_shmratio_panel(
    fig,
    X,
    y,
    redshift,
    iscale,
    halo_features_used,
    galaxy_labels_used,
    axis=[10.0,15.0,-5.0,0.0],
    vmin=-12.1,
    vmax=-8.5,
    fs=22,
    ms=6,
    lw=3,
    frac=None,
    nxpanel=1,
    nypanel=1,
    ipanel=1,
    barposition=[0.92,0.18,0.04,0.65],
    modelname=None,
    showredshift=True,
    usepeakmass=True,
    showaverage=False,
    showfit=False,
    seed=42,
    bins=None
    ):
    
    #Unscale the prediction
    gal_pred       = np.array(get_targets_unscaled(y,galaxy_labels_used))
    halos_unscaled = np.array(get_features_unscaled(X,halo_features_used))

    np.random.seed(seed=seed)
    isample = np.random.permutation(gal_pred.shape[0])
    gal_pred = gal_pred[isample]
    halos_unscaled = halos_unscaled[isample]
    
    if frac is not None:
        isample        = np.random.choice(gal_pred.shape[0],int(gal_pred.shape[0]*frac))
        gal_pred       = gal_pred[isample]
        halos_unscaled = halos_unscaled[isample]

    if usepeakmass:
        mhalo = halos_unscaled[:,1]
    else:
        mhalo = halos_unscaled[:,0]
    mstar = gal_pred[:,0]
    sfr   = gal_pred[:,1]
    z     = 1./halos_unscaled[:,iscale]-1.
    iselect = np.logical_and(halos_unscaled[:,iscale] >= 1./(redshift+1.)-0.01, halos_unscaled[:,iscale] < 1./(redshift+1.)+0.01)
    mhalo = mhalo[iselect]
    mstar = mstar[iselect]
    sfr   = sfr[iselect]
    z     = z[iselect]
    
    ax  = plt.subplot(nypanel,nxpanel,ipanel)
    ax.axis(axis)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    
    if showredshift:
        ax.annotate('$z={:.2f}$'.format(redshift), xy=(1.0-0.05, 0.05), xycoords='axes fraction', size=fs*1.1, ha='right', va='bottom')

    if modelname is not None:
        ax.annotate(modelname, xy=(0.05, 1.0-0.05), xycoords='axes fraction', size=fs*1.1, ha='left', va='top')

    if showaverage:
        if bins is None:
            bins = np.linspace(axis[0],axis[1],10)        
        mh,_,_ = st.binned_statistic(mhalo, mhalo,       'mean', bins=bins)
        mg,_,_ = st.binned_statistic(mhalo, mstar-mhalo, 'mean', bins=bins)
        sg,_,_ = st.binned_statistic(mhalo, mstar-mhalo, sigma,  bins=bins)
        ax.errorbar(mh,mg,sg,marker='o',color='white',ms=ms+2,mec='white',linestyle='',lw=lw+2, mew=lw+2,elinewidth=lw+4, capsize=(12)/2,zorder=10)
        ax.errorbar(mh,mg,sg,marker='o',color='black',ms=ms,  mec='black',linestyle='',lw=lw,   mew=lw,  elinewidth=lw+1, capsize=(8)/2, zorder=11)

        if showfit:
            xdata = mh[np.isfinite(sg)]
            ydata = mg[np.isfinite(sg)]
            sdata = sg[np.isfinite(sg)]
            if xdata.size > 5:
                fit   = so.curve_fit(msmh, xdata, ydata, np.array([12.0, 0.13, 1.3, 0.6]), sdata)
                mhfit = np.arange(np.max([mhalo.min(),axis[0]]), np.min([mhalo.max(),axis[1]]), 0.01)
                mgfit = msmh(mhfit,fit[0][0],fit[0][1],fit[0][2],fit[0][3])
                ax.plot(mhfit,mgfit,'k-',lw=lw+2)
    
    ln = ax.scatter(mhalo,mstar-mhalo,s=3,c=np.log10(sfr)-mstar,cmap=plt.cm.jet_r, vmin=vmin, vmax=vmax)
    ax.set_xlabel('$\log (M_\mathrm{p} / \mathrm{M}_\odot)$', size = fs)
    ax.set_ylabel('$\log (m_* / \mathrm{M}_\mathrm{p})$', size = fs)
    
    if ipanel==1:
        barpos=fig.add_axes(barposition)
        cbar = plt.colorbar(ln,cax=barpos, ticks=[-12.,-11.,-10.,-9.])
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label('sSFR', fontsize=fs)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

    ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
        
    if (ipanel-1) % nxpanel  > 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
    if int((ipanel-1)/nxpanel) < nypanel-1:
        ax.set_xticklabels([])
        ax.set_xlabel('')


def msmh(x, m1, n, b, g):
    return np.log10(2.0*n)-np.log10((10.**(x-m1))**(-b)+(10.**(x-m1))**(g))

def plot_data_availability(files):
    len_redshifts = {}
    xaxis=[]
    for file in files:
        len_redshifts.update({file : len(cat.load_single_file_to_panda_df(file))})
        xaxis.append(re.findall(r"\bZ\d\d", file))
    flat_xaxis = [item for sublist in xaxis for item in sublist]
    plt.bar(flat_xaxis, len_redshifts.values())
    plt.title('Amount of Data per Redshift')
    plt.show()
    print('Total Amount = {}'.format(sum(list(len_redshifts.values()))))
    
def compare_input_prediction_RNN(y,y_pred,galaxy_labels_used,lw=3,fs=22,axis1=[7.0,12.3,7.0,12.3],axis2=[-6.5,3.5,-6.5,3.5],file=None):  
    
    gal_input = np.array(get_features_unscaled(y, galaxy_labels_used))
    gal_pred = np.array(get_features_unscaled(y_pred, galaxy_labels_used))
    
    mstar  = np.linspace(axis1[0],axis1[1],1000)
    sfr    = np.linspace(axis2[0],axis2[1],1000)
    ml,_,_ = st.binned_statistic(gal_input[:,0], gal_input[:,0], 'mean', bins=10)
    mp,_,_ = st.binned_statistic(gal_input[:,0], gal_pred[:,0], 'mean', bins=10)
    ma,_,_ = st.binned_statistic(gal_input[:,0], gal_pred[:,0], sigma, bins=10)
    sl,_,_ = st.binned_statistic(np.log10(gal_input[:,1]), np.log10(gal_input[:,1]), 'mean', bins=10)
    sp,_,_ = st.binned_statistic(np.log10(gal_input[:,1]), np.log10(gal_pred[:,1]), 'mean', bins=10)
    sa,_,_ = st.binned_statistic(np.log10(gal_input[:,1]), np.log10(gal_pred[:,0]), sigma, bins=10)

    fig = plt.figure(figsize=(12.0,6.0))

    ax = plt.subplot(1,2,1)
    ax.axis(axis1)
    ax.plot(gal_input[:,0],gal_pred[:,0],'k,')
    ax.plot(mstar,mstar,'b-',lw=2*lw) 
    ax.errorbar(ml,mp,ma,marker='x',color='red',ms=14,mec='red',linestyle='',mew=lw,elinewidth=lw-1, capsize=(12)/2, zorder=11)
    ax.set_xlabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{label}$', fontsize = fs)
    ax.set_ylabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{prediction}$', fontsize = fs)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)


    ax = plt.subplot(1,2,2)
    ax.axis(axis2)
    ax.plot(np.log10(gal_input[:,1]),np.log10(gal_pred[:,1]),'k,')
    ax.plot(sfr,sfr,'b-',lw=2*lw)
    ax.errorbar(sl,sp,sa,marker='x',color='red',ms=14,mec='red',mew=lw,elinewidth=lw-1, capsize=(12)/2, zorder=11)
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.set_xlabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})_\mathrm{label}$', fontsize = fs)
    ax.set_ylabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})_\mathrm{prediction}$', fontsize = fs)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.9, top=0.99, wspace=0.0, hspace=0.0)
    
    if file is not None:
        plt.savefig(file)
    
    plt.show()

def compare_scaled_input_prediction_RNN(y_vali,y_vali_pred,lw=3,fs=15,file=None):
    mstar_y_vali_pred = y_vali_pred[:,0]
    mstar_y_vali = y_vali[:,0]
    SFR_y_vali_pred = y_vali_pred[:,1]
    SFR_y_vali = y_vali[:,1]

    sfr    = np.linspace(0.0,1.0,1000)
    mstar  = np.linspace(0.0,1.0,1000)

    fig = plt.figure(figsize=(12.0,6.0))

    ax = plt.subplot(1,2,1)
    ax.plot(mstar_y_vali, mstar_y_vali_pred,'k,')
    ax.plot(mstar,mstar,'b-')
    ax.set_xlabel('$\ (m_*)_\mathrm{label}$', fontsize = fs)
    ax.set_ylabel('$\ (m_*)_\mathrm{prediction}$', fontsize = fs)
    ax.set_title('$\ (m_*)_\mathrm{prediction}$ vs. $\ (m_*)_\mathrm{label}$')
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)

    ax = plt.subplot(1,2,2)
    ax.plot(SFR_y_vali, SFR_y_vali_pred,'k,')
    ax.plot(sfr,sfr,'b-')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.set_xlabel('$\ (SFR)_\mathrm{label}$', fontsize = fs)
    ax.set_ylabel('$\ (SFR)_\mathrm{prediction}$', fontsize = fs)
    ax.set_title('$\ (SFR)_\mathrm{prediction}$ vs. $\ (SFR)_\mathrm{label}$')
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    if file is not None:
            plt.savefig(file)
    plt.show()

def plot_main_sequence_panel_RNN(
    fig,
    X,
    y,
    redshift_min,
    redshift_max,
    iscale,
    halo_features_used,
    galaxy_labels_used,
    H0=70.0,
    Om0=0.3,
    plot_obs=True,
    axis=[7.0,12.9,-6.20,1.8],
    fs=22,
    lw=3,
    frac=None,
    nxpanel=1,
    nypanel=1,
    ipanel=1,
    barposition=[0.92,0.18,0.04,0.65],
    modelname=None,
    showredshift=True,
    Unscale=True,
    seed=42
    ):
    
    if Unscale == False:
        gal_pred = y
        halos_unscaled = X
    else:
        gal_pred       = np.array(get_targets_unscaled(y,galaxy_labels_used))
        halos_unscaled = np.array(get_features_unscaled(X,halo_features_used))

    np.random.seed(seed=seed)
    isample = np.random.permutation(gal_pred.shape[0]) 
    gal_pred = gal_pred[isample]
    halos_unscaled = halos_unscaled[isample]
    
    if frac is not None:
        isample        = np.random.choice(gal_pred.shape[0],int(gal_pred.shape[0]*frac))
        gal_pred       = gal_pred[isample]
        halos_unscaled = halos_unscaled[isample]

    mstar = gal_pred[:,0]
    sfr   = gal_pred[:,1]
    apeak = halos_unscaled[:,5]
    z     = 1./halos_unscaled[:,iscale]-1.
    iselect = np.logical_and(halos_unscaled[:,iscale] >= 1./(redshift_max+1.)-0.01, halos_unscaled[:,iscale] < 1./(redshift_min+1.)+0.01)
    mstar = mstar[iselect] # iselect is an array contiang of boolean values which removes all False values
    sfr   = sfr[iselect]
    z     = z[iselect]
    apeak = apeak[iselect]
    if plot_obs:
        #Set minimum SSFR
        ssfrmin = 1.e-12
        sfr[np.where(sfr/10.**mstar<ssfrmin)] = ssfrmin*10.**mstar[np.where(sfr/10.**mstar<ssfrmin)]
    sfr   = np.log10(sfr)
    #Add scatter to the SFR to get observed SFR
    if plot_obs:
        cosmo = ac.FlatLambdaCDM(H0=H0,Om0=Om0)
        tcosmic = cosmo.age([z]).value[0]
        thre  = np.log10(0.3/tcosmic)-9.0
        ssfr  = sfr - mstar
        sig   = 0.2*(np.tanh(1.0*(thre-ssfr.flatten()))*0.5+0.5)+0.1
        np.random.seed(seed=42)
        rangau = np.random.normal(loc=0.0,scale=sig,size=sfr.size)
        sfrobs = sfr+rangau
        #Add scatter to the stellar mass to get observed mstar
        np.random.seed(seed=42)
        rangau = np.random.normal(loc=0.0,scale=0.1,size=mstar.size)
        mstarobs = mstar+rangau

    ax  = plt.subplot(nypanel,nxpanel,ipanel)
    ax.axis(axis)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    
    if showredshift:
        ax.annotate('$z={:.2f}-{:.2f}$'.format(redshift_min,redshift_max), xy=(1.0-0.05, 0.05), xycoords='axes fraction', size=fs*0.9, ha='right', va='bottom')

    if modelname is 'Emerge':
        ax.annotate(modelname, xy=(0.05, 1.0-0.05), xycoords='axes fraction', size=fs*1.1, ha='left', va='top')
    else:
        ax.annotate(modelname, xy=(1.0-0.05, 0.05), xycoords='axes fraction', size=fs*1.1, ha='right', va='bottom')       

    if plot_obs:
        ln = ax.scatter(mstarobs,sfrobs,s=2,c=apeak*(0.5*(redshift_max+redshift_min)+1.0),cmap=plt.cm.jet_r, vmin=0.0, vmax=1.0)
    else:
        ln = ax.scatter(mstar,sfr,s=2,c=apeak*(0.5*(redshift_max+redshift_min)+1.0),cmap=plt.cm.jet_r, vmin=0.0, vmax=1.0)
    ax.set_xlabel('$\log (m_* / \mathrm{M}_\odot)$', size = fs)
    ax.set_ylabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})$', size = fs)
    
    if ipanel==1:
        barpos=fig.add_axes(barposition)
        cbar = plt.colorbar(ln,cax=barpos, ticks=[0.0,0.2,0.4,0.6,0.8,1.0])
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label('$a_\mathrm{p} \, / \, a$', fontsize=fs)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    
    ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
        
    if (ipanel-1) % nxpanel  > 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
    if int((ipanel-1)/nxpanel) < nypanel-1:
        ax.set_xticklabels([])
        ax.set_xlabel('')
        
def plot_shmr_panel_RNN(
    fig,
    X,
    y,
    redshift_min,
    redshift_max,
    iscale,
    halo_features_used,
    galaxy_labels_used,
    axis=[10.0,15.0,7.0,12.0],
    vmin=-12.1,
    vmax=-8.5,
    fs=22,
    lw=3,
    frac=None,
    nxpanel=1,
    nypanel=1,
    ipanel=1,
    barposition=[0.92,0.18,0.04,0.65],
    modelname=None,
    showredshift=True,
    Unscale=True,
    seed=42
    ):
    
    #Unscale the prediction
    if Unscale == False:
        gal_pred = y
        halos_unscaled = X
    else:
        gal_pred       = np.array(get_targets_unscaled(y,galaxy_labels_used))
        halos_unscaled = np.array(get_features_unscaled(X,halo_features_used))

    np.random.seed(seed=seed)
    isample = np.random.permutation(gal_pred.shape[0])
    gal_pred = gal_pred[isample]
    halos_unscaled = halos_unscaled[isample]
    
    if frac is not None:
        isample        = np.random.choice(gal_pred.shape[0],int(gal_pred.shape[0]*frac))
        gal_pred       = gal_pred[isample]
        halos_unscaled = halos_unscaled[isample]
    
    mhalo = halos_unscaled[:,2]
    mstar = gal_pred[:,0]
    sfr   = gal_pred[:,1]
    z     = 1./halos_unscaled[:,iscale]-1.
    iselect = np.logical_and(halos_unscaled[:,iscale] >= 1./(redshift_max+1.)-0.01, halos_unscaled[:,iscale] < 1./(redshift_min+1.)+0.01)
    mhalo = mhalo[iselect]
    mstar = mstar[iselect]
    sfr   = sfr[iselect]
    z     = z[iselect]
    
    ax  = plt.subplot(nypanel,nxpanel,ipanel)
    ax.axis(axis)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    
    if showredshift:
        ax.annotate('$z={:.2f}-{:.2f}$'.format(redshift_min,redshift_max), xy=(1.0-0.05, 0.05), xycoords='axes fraction', size=fs*0.9, ha='right', va='bottom')

    if modelname is not None:
        ax.annotate(modelname, xy=(0.05, 1.0-0.05), xycoords='axes fraction', size=fs*1.1, ha='left', va='top')

    ln = ax.scatter(mhalo,mstar,s=3,c=np.log10(sfr)-mstar,cmap=plt.cm.jet_r, vmin=vmin, vmax=vmax)
    ax.set_xlabel('$\log (M_\mathrm{p} / \mathrm{M}_\odot)$', size = fs)
    ax.set_ylabel('$\log (m_* / \mathrm{M}_\odot)$', size = fs)
    
    if ipanel==1:
        barpos=fig.add_axes(barposition)
        cbar = plt.colorbar(ln,cax=barpos, ticks=[-12.,-11.,-10.,-9.])
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label('sSFR', fontsize=fs)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

    ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
        
    if (ipanel-1) % nxpanel  > 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
    if int((ipanel-1)/nxpanel) < nypanel-1:
        ax.set_xticklabels([])
        ax.set_xlabel('')

def plot_compare_models(models_hist, models, ymin=0.003, ymax=0.02,fs=22,lw=3, savename=None):
    
    from keras.utils.layer_utils import count_params
    
    data = []
    data_mins = []
    params = []

    for i in models_hist:
        file = hdf5.File('RNNs/Models/{}'.format(i),'r')
        histfile = hdf5.File('RNNs/Models/{}'.format(i),'r')
        training = {'loss': np.array(histfile['loss']).tolist(), 'val_loss': np.array(histfile['val_loss']).tolist()}
        data.append(training['val_loss'])
        data_mins.append(np.min(training['val_loss']))
    for i in models:
        model = tf.keras.models.load_model('RNNs/Models/{}'.format(i))
        trainable_count = count_params(model.trainable_weights)
        params.append(trainable_count)

    fully_nested = [list(zip(*[(ix+1,y) for ix,y in enumerate(x)])) for x in data]
    names1 = [i[:-3] for i in models]
    names2 = ["{} (Number of trainable Parameters: {:02})".format(names1_, params_) for names1_, params_ in zip(names1, params)]
    xmin = 0
    xmax = 200
    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    ax.axis([xmin,xmax,ymin,ymax])
    ax.set_xlabel('Epoch',size=fs)
    ax.set_ylabel('Val_loss',size=fs)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([xmin,xmax])
    ax.set_xticks(np.arange(0,xmax,20))
    ax.set_xticks(np.arange(0,xmax,10),minor=True)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    #ax.set_yscale('logit')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    for l in fully_nested:
        ax.semilogy(*l)
    ax.legend(names2, fontsize=15, loc = 'lower left')
    plt.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.99, hspace=0.0, wspace=0.0)
    plt.tight_layout()
    plt.savefig(savename)
    
def plot_compare_models2(models_hist, models, ymin=0.003, ymax=0.02, xmin=0, xmax = 100,fs=22,lw=3, savename='asd.png'):
    import matplotlib.cm as cm
    from keras.utils.layer_utils import count_params
    x = np.arange(len(models))
    ys = [i+x+(i*x)**2 for i in range(len(models))]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    #colors2 = ['black', 'gray', 'darkred','red','orangered','darkblue','magenta','blue']
    data = []
    data_mins = []
    params = []
    asd=[1,2,3,4,5,6,7,8]
    for i in models_hist:
        file = hdf5.File('Master/compare_F/{}'.format(i),'r')
        histfile = hdf5.File('Master/compare_F/{}'.format(i),'r')
        training = {'loss': np.array(histfile['loss']).tolist(), 'val_loss': np.array(histfile['val_loss']).tolist()}
        data.append(training['val_loss'])
        data_mins.append(np.min(training['val_loss']))
    for i in models:
        model = tf.keras.models.load_model('Master/compare_F/{}'.format(i))
        trainable_count = count_params(model.trainable_weights)
        params.append(trainable_count)
    
    data_mins2 = [np.round(i,5) for i in data_mins]
    
    fully_nested = [list(zip(*[(ix+1,y) for ix,y in enumerate(x)])) for x in data]
    names1 = [i[:-3] for i in models]
#     names1 = ['$F_2$','$F_3$','$F_4$','$F_5$','$F_6$','$F_7$','$F_8$','$F_9$','$F_{10}$','$F_{11}$']
    for counter, i in enumerate(names1):
        if counter in [5,6,7]:
            names1[counter] = i + '   ' 
#         if counter in [0]:
#             names1[counter] = i + '              ' 
#         elif counter in [1]:
#             names1[counter] = i + '          '
#         elif counter in [2]:
#             names1[counter] = i + '  '
#         elif counter in [5]:
#             names1[counter] = i + '           '
#         elif counter in [6]:
#             names1[counter] = i + '               '
#         elif counter in [8]:
#             names1[counter] = i + '           '

    names2 = ["{} $(Params: {}, Val\_loss_{{min}}: {:02})$".format(names1_, params_,data_mins2) for names1_, params_,data_mins2 in zip(names1, params,data_mins2)]
    names3 = ['{}    $Val\_loss_{{min}}:$ {}'.format(i,j) for i,j in zip(names1, data_mins2)]

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)
    ax.axis([xmin,xmax,ymin,ymax])
    ax.set_xlabel('Epochs',size=fs)
    ax.set_ylabel('Validation_loss',size=fs)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.set_xticks([0,20,40,60,80,100])
    ax.set_xticks(np.arange(0,xmax,10),minor=True)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    #ax.set_yscale('logit')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    for counter, l in enumerate(fully_nested):  
        ax.semilogy(*l, color=colors[counter], linewidth=1.2)
    ax.set_ylim([ymin,ymax])
    ax.set_xlim([xmin,xmax])
    ax.set_yticks([0.004,0.006,0.008,0.01,0.02, 0.03,0.04, 0.05])
    ax.set_yticklabels([0.004,0.006,0.008,0.01,0.02, 0.03,0.04, 0.05])
    ax.legend(names3, fontsize=11, loc = 'upper right', ncol=2, handleheight=2.4, labelspacing=0.05)
    # try this to align legend
    #fontsize='xx-large', ncol=2,handleheight=2.4, labelspacing=0.05
    
    plt.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.99, hspace=0.0, wspace=0.0)
    plt.tight_layout()
    plt.savefig(savename,dpi=100, bbox_inches = 'tight', pad_inches = 0.1)
    return data_mins

def get_feature_importance(forest_reg, halo_features_used, lw=3,fs=8,file='feature_importance.png'):
    features = []
    labels   = []
    feature_importances = forest_reg.feature_importances_
    for feature in halo_features_used:
        if feature[1] == 'categorical':
            features.append(feature[0])
            labels.append(feature[2])
        else:
            features.append(feature[0])
            labels.append(feature[4]) 

    i = np.argsort(feature_importances)[::-1]   
    feature_importances = np.array(feature_importances)[i]
    features            = np.array(features)[i]
    labels              = np.array(labels)[i]

    fig = plt.figure(figsize=(12.0,8.0))
    ax  = plt.subplot(111,yscale='log',ylim=(0.002,0.7))
    x_co = np.arange(len(features))
    ax.bar(x_co, feature_importances, align='center', color='blue',alpha=1.0)
    ax.set_ylabel('Feature Importance', size = fs)
    ax.set_xlabel('Feature', size = fs)
    plt.xticks(x_co, labels, fontsize = fs)
    plt.yticks([0.01,0.1],['0.01',' 0.1'],fontsize = fs)
    ax.set_xticklabels(labels,verticalalignment='bottom')
    ax.tick_params(width=0,pad=fs*1.7, axis='x')
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2, axis='y')
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2, axis='y')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    plt.subplots_adjust(left=0.10, right=0.99, bottom=0.12, top=0.99, hspace=0.0, wspace=0.0)

    if file is not None:
        plt.savefig(file)

    plt.show()
    
    return feature_importances

#used for compare_calculated_mstar function
def normalize(x):
    y = (x-np.min(x))/(np.max(x)-np.min(x))
    return y

def compare_calculated_mstar(mstar_calc, mstar_calc_insitu, mstar_pred, mstar_label,compare='hist', lw=3,fs=15,file=None):

    mstar  = np.linspace(0.0,12.0,1000)
    binwidth = 0.5
    threshold = 0.0
    bins =  np.arange(2, 12 + binwidth, binwidth)
    bin_counts1, bin_edges, binnumber1 = st.binned_statistic(mstar_calc_insitu, mstar_pred,statistic='count', bins=bins)
    bin_counts2, bin_edges, binnumber2 = st.binned_statistic(mstar_calc_insitu, mstar_label,statistic='count', bins=bins)
    bin_counts3, bin_edges, binnumber3 = st.binned_statistic(mstar_calc, mstar_pred,statistic='count', bins=bins)
    bin_counts4, bin_edges, binnumber4 = st.binned_statistic(mstar_calc, mstar_label,statistic='count', bins=bins)
        
    bin_means1, bin_edges, binnumber1 = st.binned_statistic(mstar_calc_insitu, mstar_pred,statistic='mean', bins=bins)
    bin_means2, bin_edges, binnumber2 = st.binned_statistic(mstar_calc_insitu, mstar_label,statistic='mean', bins=bins)
    bin_means3, bin_edges, binnumber3 = st.binned_statistic(mstar_calc, mstar_pred,statistic='mean', bins=bins)
    bin_means4, bin_edges, binnumber4 = st.binned_statistic(mstar_calc, mstar_label,statistic='mean', bins=bins)
#     bin_means1 = [0 if np.isnan(x) else x for x in bin_means1]                                                
#     bin_means2 = [0 if np.isnan(x) else x for x in bin_means2] 

    bin_sum1, bin_edges, binnumber1 = st.binned_statistic(mstar_calc_insitu, mstar_calc_insitu,statistic='sum', bins=bins) #pred
    bin_sum2, bin_edges, binnumber2 = st.binned_statistic(mstar_calc_insitu, mstar_calc_insitu,statistic='sum', bins=bins) #label
    bin_sum3, bin_edges, binnumber3 = st.binned_statistic(mstar_calc, mstar_calc,statistic='sum', bins=bins) #pred
    bin_sum4, bin_edges, binnumber4 = st.binned_statistic(mstar_calc, mstar_calc,statistic='sum', bins=bins) #label
    bin_sum_norm1 = normalize([*bin_sum1, *bin_sum3])
    bin_sum_norm2 = normalize([*bin_sum2, *bin_sum4])
    bin_sum1  = bin_sum_norm1[0:len(bins)-1]
    bin_sum3  = bin_sum_norm1[len(bins)-1:len(bin_sum_norm1)]
    bin_sum2  = bin_sum_norm1[0:len(bins)-1]
    bin_sum4  = bin_sum_norm1[len(bins)-1:len(bin_sum_norm2)]
    
    bin_centers = bin_edges[1:] - binwidth/2

    diff_counts1 =  bin_counts3 - bin_counts1
    diff_counts2 =  bin_counts4 - bin_counts2
    
    diff_mean1 = bin_means3 - bin_means1
    diff_mean2 = bin_means4 - bin_means2

    diff_sum1 =  bin_sum3 - bin_sum1
    diff_sum2 =  bin_sum4 - bin_sum2

    
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(20,20), sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.2, 'wspace': 0.2})

    ax1.plot(mstar_calc_insitu, mstar_pred,'k,', alpha=0.01)
    ax1.plot(mstar,mstar,'dodgerblue')
    ax1.text(3, 10,'Insitu', fontsize=20, bbox=dict(facecolor='dodgerblue', alpha=0.7))
    ax1.set_xlabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{integrated}$', fontsize = fs)
    ax1.set_ylabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{prediction}$', fontsize = fs)
    ax1.set_xlim(2,12)
    ax1.set_ylim(2,12)
    ax1.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax1.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax1.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(lw)
        
    ax2.plot(mstar_calc_insitu, mstar_label,'k,', alpha=0.01)
    ax2.plot(mstar,mstar,'dodgerblue')
    ax2.text(3, 10,'Insitu', fontsize=20, bbox=dict(facecolor='dodgerblue', alpha=0.7))
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.tick_right()
    ax2.set_xlabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{integrated}$', fontsize = fs)
    ax2.set_ylabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{label}$', fontsize = fs)
    ax2.set_xlim(2,12)
    ax2.set_ylim(2,12)
    ax2.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax2.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax2.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax2.spines[axis].set_linewidth(lw)
        
    
    ax3.plot(mstar_calc, mstar_pred,'k,', alpha=0.01)
    ax3.plot(mstar,mstar,'dodgerblue')
    ax3.text(3, 10,'Insitu+Exsitu', fontsize=20, bbox=dict(facecolor='dodgerblue', alpha=0.7))
    ax3.set_xlabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{integrated}$', fontsize = fs)
    ax3.set_ylabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{prediction}$', fontsize = fs)
    ax3.set_xlim(2,12)
    ax3.set_ylim(2,12)
    ax3.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax3.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax3.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    if compare == 'count':
        ax5 = ax3.twinx()
        above_threshold1 = np.maximum(diff_counts1, 0)
        below_threshold1 = np.minimum(diff_counts1, threshold)
        ax5.hlines(0, 2, 12)
        ax5.bar(bin_centers, below_threshold1, 0.5, color="r", alpha=0.3)
        ax5.bar(bin_centers, above_threshold1, 0.5, color="g",bottom=below_threshold1, alpha=0.3)
        ax5.set_ylabel('Difference in Counts')
    elif compare == 'mean':
        ax5 = ax3.twinx()
        ax5.scatter(bin_centers, diff_mean1, c='b', s=50)
        ax5.hlines(0, 2, 12)
        ax5.set_ylabel('Difference in Means')
    elif compare == 'sum':
        ax5 = ax3.twinx()
        above_threshold1 = np.maximum(diff_sum1, 0)
        below_threshold1 = np.minimum(diff_sum1, threshold)
        ax5.hlines(0, 2, 12)
        ax5.tick_params(axis='both', direction='out', which = 'both', bottom=True, top=True, right=True, labelsize=fs*0.5)
        ax5.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
        ax5.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
        ax5.bar(bin_centers, below_threshold1, 0.5, color="r", alpha=0.3)
        ax5.bar(bin_centers, above_threshold1, 0.5, color="g",bottom=below_threshold1, alpha=0.3)
        #ax5.set_ylabel('Difference in Normalized Sum')
    for axis in ['top','bottom','left','right']:
        ax3.spines[axis].set_linewidth(lw)

    ax4.plot(mstar_calc, mstar_label,'k,', alpha=0.01)
    ax4.plot(mstar,mstar,'dodgerblue')
    ax4.text(3, 10,'Insitu+Exsitu', fontsize=20, bbox=dict(facecolor='dodgerblue', alpha=0.7))
    ax4.yaxis.set_label_position('left')
    ax4.yaxis.tick_right()
    ax4.set_xlabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{integrated}$', fontsize = fs)
    ax4.set_ylabel('$\log (m_* / \mathrm{M}_\odot)_\mathrm{label}$', fontsize = fs)
    ax4.set_xlim(2,12)
    ax4.set_ylim(2,12)
    ax4.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax4.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax4.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    if compare == 'count':
        ax6 = ax4.twinx()
        above_threshold2 = np.maximum(diff_counts2, 0)
        below_threshold2 = np.minimum(diff_counts2, threshold)
        ax6.hlines(0, 2, 12)
        ax6.bar(bin_centers, below_threshold2, 0.5, color="r", alpha=0.3)
        ax6.set_ylabel('Difference in Counts')
        ax6.bar(bin_centers, above_threshold2, 0.5, color="g",bottom=below_threshold2, alpha=0.3)
    elif compare == 'mean':
        ax6 = ax4.twinx()
        ax6.scatter(bin_centers, diff_mean2, c='b', s=50)
        ax6.hlines(0, 2, 12)
        ax6.set_ylabel('Difference in Means')
    elif compare == 'sum':
        ax6 = ax4.twinx()
        above_threshold2 = np.maximum(diff_sum2, 0)
        below_threshold2 = np.minimum(diff_sum2, threshold)
        ax6.hlines(0, 2, 12)
        ax6.bar(bin_centers, below_threshold2, 0.5, color="r", alpha=0.3)
        ax6.bar(bin_centers, above_threshold2, 0.5, color="g",bottom=below_threshold1, alpha=0.3)
        ax6.tick_params(axis='both', direction='out', which = 'both', bottom=True, top=True, right=True, labelsize=fs*0.5)
        ax6.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
        ax6.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
        ax6.set_ylabel('Difference in normalized sum', fontsize = fs)
    for axis in ['top','bottom','left','right']:
        ax4.spines[axis].set_linewidth(lw)
    
    #fig.suptitle('Integrated Stellar mass calculated from RNN sfr_predictions', fontsize=20, y = 0.93)
        
    if file is not None:
            plt.savefig(file,dpi=100, bbox_inches = 'tight',
    pad_inches = 0.1)
            
    #plt.tight_layout()
    plt.show()
    
def plot_mso_history(file, fs=20, lw=3, savefile=None):
    gbestfile = hdf5.File(file,'r')
    history = np.array(gbestfile['Swarm_history'])
    iterations = len(history[0])
    n_swarms = len(history)
    x = np.arange(1,iterations+1,1)
    xmax = iterations
    for counter, i in enumerate(history[0]):
        if i == 0:
            xmax = counter
            break    
    
    fig, ax = plt.subplots(1, figsize = (12.0,8.0))
    for i in range(n_swarms):
        plt.plot(x, history[i], label='Swarm{}'.format(i+1), linewidth=lw)
    ax.set_xlim(0,xmax-1)
    ax.set_xlabel('Iterations',size=fs)
    ax.set_ylabel('Loss',size=fs)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.set_xticks(np.arange(0,xmax,5))
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    plt.legend(fontsize=fs*0.8,frameon=False,loc='upper right')
    plt.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.99, hspace=0.0, wspace=0.0)
    plt.title('Training History MSO', fontsize=1.25*fs)
    plt.savefig(savefile)
    gbestfile.close()
    
def plot_mso_history2(file1,file2,file3, fs=20, lw=3, savefile=None):
    import matplotlib.cm as cm
    from keras.utils.layer_utils import count_params
    x = np.arange(6)
    ys = [i+x+(i*x)**2 for i in range(6)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    gbestfile1 = hdf5.File(file1,'r')
    gbestfile2 = hdf5.File(file2,'r')
    gbestfile3 = hdf5.File(file3,'r')
    history1 = np.array(gbestfile1['Swarm_history']).tolist()
    history2 = np.array(gbestfile2['Swarm_history']).tolist()
    history3 = np.array(gbestfile3['Swarm_history']).tolist()
    for counter, i in enumerate(history2):
        history2[counter] = i[:-4] # delete zero terms of history files
    for counter, i in enumerate(history3):
        history3[counter] = i[:-2]
    history = np.concatenate((history1, history2, history3), axis = 1)
    iterations = len(history[0])
    n_swarms = len(history)
    x = np.arange(1,iterations+1,1)
    xmax = iterations 
    fig, ax = plt.subplots(1, figsize = (12.0,8.0))
    for i in range(n_swarms):  
        ax.plot(x,history[i], label='Swarm{}'.format(i+1), color=colors[i], linewidth=1.5)
#     for i in range(n_swarms):
#         plt.plot(x, history[i], label='Swarm{}'.format(i+1), linewidth=lw)
    ax.set_xlim(0,xmax-1)
    ax.set_xlabel('Iterations',size=fs*1.2)
    ax.set_ylabel('Loss',size=fs*1.2)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    ax.tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=2*lw, which='minor', direction='in', pad=fs/2)
    ax.set_xticks([0,20,40,60,80,100,120,140])
    ax.set_xticks(np.arange(0,xmax,10),minor=True)
    ax.set_yticks([640, 660, 680, 700, 720, 740, 760, 780, 800])
    ax.set_yticks(np.arange(650,820,10),minor=True)
    
    

    ax.set_ylim([640,820])
    ax.set_xlim([0,xmax])
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
    #ax.legend(names3, fontsize=15, loc = 'upper right', ncol=2, handleheight=2.4, labelspacing=0.05)
    plt.legend(fontsize=fs,loc='upper right', ncol=2, handleheight=2.4, labelspacing=0.05)
    plt.subplots_adjust(left=0.09, right=0.99, bottom=0.11, top=0.99, hspace=0.0, wspace=0.0)
#    plt.title('Training History MSO', fontsize=1.25*fs)
    plt.savefig(savefile, dpi=100)
    gbestfile.close()
  
def compare_chi2(chi2_em, chi2_mod, lw=4, fs=15):
    chi2_list = [r'$L(\theta)_{Total}$', r'$L(\theta)_{SMF}$', r'$L(\theta)_{FQ}$', 
                 r'$L(\theta)_{CSFRD}$', r'$L(\theta)_{sSFR}$', r'$L(\theta)_{WP}$']
    chi2_list = ['Total', 'SMF', r'FQ', 
                 'CSFRD', 'sSFR', 'WP']
    x = np.arange(len(chi2_list))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots(figsize=(12,8))
    rects1 = ax.bar(x - width/2, chi2_em, width, label='Emerge', color='deepskyblue')
    rects2 = ax.bar(x + width/2, chi2_mod, width, label='RNN + RL', color='limegreen')
    
    ax.set_xticks(x)
    ax.set_xticklabels(chi2_list)
    plt.legend(fontsize=fs*2,loc='upper right')
    ax.tick_params(axis='both', direction='out', which = 'both', bottom=True, top=False, left=True, right=False, labelsize=fs*1.5)
    ax.tick_params(width=lw, length=2*lw, which='major', direction='out', pad=fs/2)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
#    ax.set_title('Compare Chi2 Results', size=1.5*fs)
    ax.set_ylabel(r'$L(\theta)$',size=1.5*fs)


    
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig('Compare_Chi2.png',dpi=100, bbox_inches = 'tight',
    pad_inches = 0.1)
    plt.show()
    
def plot_main_branches(df_RNN, df_RNN_RL, df_NN, df_NN_RL, fs = 25, lw=3,file=None):
    fig, axs = plt.subplots(5, 2, figsize=(40,60),gridspec_kw={'hspace': 0, 'wspace': 0.2}, sharex='all')
    eins = [0,0,1,0,2,0,3,0,4,0]
    zwei = [0,0,0,1,0,2,0,3,0,4]
    xticks = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    yticks = [6,7,8,9,10,11]
    

    for counter, i in enumerate(range(len(df_RNN))):
        if i & 1 == 0:
            axs[eins[counter], 0].set_xscale('log')
            axs[eins[counter], 0].plot(df_RNN[i]['Scale'],df_RNN[i]['Stellar_mass'],'-o', color='black',  label='$m_{\star \,Emerge_{label}}$', alpha = 0.9)
            axs[eins[counter], 0].plot(df_RNN[i]['Scale'],df_RNN[i]['mstar_pred'],linestyle='dashed', color='orange', label='$m_{\star \,RNN_{prediction}}$')
            axs[eins[counter], 0].plot(df_RNN[i]['Scale'],df_RNN[i]['mstar_integrated'], color='orange', label='$m_{\star \,RNN_{integrated}}$')
            axs[eins[counter], 0].plot(df_RNN_RL[i]['Scale'],df_RNN_RL[i]['mstar_pred'],linestyle='dashed', color='magenta', label='$m_{\star \,RNN+RL_{prediction}}$')
            axs[eins[counter], 0].plot(df_RNN_RL[i]['Scale'],df_RNN_RL[i]['mstar_integrated'], color='magenta', label='$m_{\star \,RNN+RL_{integrated}}$')
            axs[eins[counter], 0].plot(df_NN[i]['Scale'],df_NN[i]['mstar_pred'],linestyle='dashed', color='blue', label='$m_{\star \,NN_{prediction}}$')
            axs[eins[counter], 0].plot(df_NN[i]['Scale'],df_NN[i]['mstar_integrated'], color='blue', label='$m_{\star \,NN_{integrated}}$')
            axs[eins[counter], 0].plot(df_NN_RL[i]['Scale'],df_NN_RL[i]['mstar_pred'],linestyle='dashed', color='green', label='$m_{\star \,NN+RL_{prediction}}$')
            axs[eins[counter], 0].plot(df_NN_RL[i]['Scale'],df_NN_RL[i]['mstar_integrated'], color='green', label='$m_{\star \,NN+RL_{integrated}}$')
            axs[eins[counter], 0].set_xlabel('a', fontsize = fs)
            axs[eins[counter], 0].set_ylabel('$\log (m_* / \mathrm{M}_\odot)$', fontsize = fs)
            axs[eins[counter], 0].tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
            axs[eins[counter], 0].tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
            axs[eins[counter], 0].set_xticks(xticks)
            axs[eins[counter], 0].set_xticklabels([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=fs*0.7)
            #axs[eins[counter], 0].set_xlim(0.01,1.0)
            axs[eins[counter], 0].set_yticks(yticks)
            axs[eins[counter], 0].set_ylim(5,12)
            axs[eins[counter], 0].legend(loc='best', fontsize='xx-large')
            for axis in ['top','bottom','left','right']:
                axs[eins[counter], 0].spines[axis].set_linewidth(lw)
        else:
            axs[zwei[counter], 1].set_xscale('log')
            axs[zwei[counter], 1].plot(df_RNN[i]['Scale'],df_RNN[i]['Stellar_mass'],'-o', color='black',  label='$m_{\star \,Emerge_{label}}$', alpha =0.9)
            axs[zwei[counter], 1].plot(df_RNN[i]['Scale'],df_RNN[i]['mstar_pred'],linestyle='dashed' , color='orange', label='$m_{\star \,RNN_{prediction}}$')
            axs[zwei[counter], 1].plot(df_RNN[i]['Scale'],df_RNN[i]['mstar_integrated'], color='orange', label='$m_{\star \,RNN_{integrated}}$')
            axs[zwei[counter], 1].plot(df_RNN_RL[i]['Scale'],df_RNN_RL[i]['mstar_pred'],linestyle='dashed', color='magenta', label='$m_{\star \,RNN+RL_{prediction}}$')
            axs[zwei[counter], 1].plot(df_RNN_RL[i]['Scale'],df_RNN_RL[i]['mstar_integrated'], color='magenta', label='$m_{\star \,RNN+RL_{integrated}}$')
            axs[zwei[counter], 1].plot(df_NN[i]['Scale'],df_NN[i]['mstar_pred'],linestyle='dashed', color='blue', label='$m_{\star \,NN_{prediction}}$')
            axs[zwei[counter], 1].plot(df_NN[i]['Scale'],df_NN[i]['mstar_integrated'], color='blue', label='$m_{\star \,NN_{integrated}}$')
            axs[zwei[counter], 1].plot(df_NN_RL[i]['Scale'],df_NN_RL[i]['mstar_pred'],linestyle='dashed', color='green', label='$m_{\star \,NN+RL_{prediction}}$')
            axs[zwei[counter], 1].plot(df_NN_RL[i]['Scale'],df_NN_RL[i]['mstar_integrated'], color='green', label='$m_{\star \,NN+RL_{integrated}}$')
            axs[zwei[counter], 1].set_xlabel('a', fontsize = fs)
            axs[zwei[counter], 1].set_ylabel('$\log (m_* / \mathrm{M}_\odot)$', fontsize = fs)
            axs[zwei[counter], 1].tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
            axs[zwei[counter], 1].tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
            axs[zwei[counter], 1].set_xticks(xticks)
            axs[zwei[counter], 1].set_xticklabels([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],fontsize=fs*0.7)
            #axs[zwei[counter], 1].set_xlim(0.01,1.0)
            axs[zwei[counter], 1].set_yticks(yticks)
            axs[zwei[counter], 1].set_ylim(5,12)
            axs[zwei[counter], 1].legend(loc='best', fontsize='xx-large')
            for axis in ['top','bottom','left','right']:
                axs[zwei[counter], 1].spines[axis].set_linewidth(lw)
            
    #fig.suptitle('Comparison of RNN and NN results for several main branches', fontsize=40, y =0.92)
    if file is not None:
            plt.savefig(file, dpi=100, bbox_inches = 'tight',
    pad_inches = 0.1)
    plt.show()

def plot_baryon_efficiency(df, fs = 25, lw=3, file=None):
    redshift = [0,.1,.5,1,2]
    zmin = [0, 0.09, 0.45, 0.99, 1.95]
    zmax = [0.0000001, 0.11, 0.55, 1.01, 2.05]
    xticks = [11,12,13,14]
    yticks = [-1,-2,-3,-4]
    fig, axs = plt.subplots(1, 5, figsize=(24,8),gridspec_kw={'hspace': 0, 'wspace': 0}, sharey='all')
    for counter, i in enumerate(redshift):

        df_select = df[(df['Redshift'] >= zmin[counter]) & (df['Redshift'] < zmax[counter])]
        t = df_select['Stellar_mass']
        halo_mass_peak = np.array(df_select['Halo_peak_mass'])
        ratio = np.log10(np.array(df_select['SFR']) / np.array(df_select['Halo_Growth_rate']))
        im = axs[counter].scatter(halo_mass_peak, ratio, linewidths=.01, label ='z = {}'.format(redshift[counter]), c=t, cmap = 'jet')

        if counter == 0:
            axs[counter].set_ylabel('$log(\.m_* / \.M_h)$', fontsize = fs)
        axs[counter].set_aspect('equal')
        axs[counter].set_xlabel('$\log (M_h / \mathrm{M}_\odot)$', fontsize = fs)
        axs[counter].tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
        axs[counter].tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
        axs[counter].set_xticks(xticks)
        axs[counter].set_xlim(10,14.5)
        axs[counter].set_yticks(yticks)
        axs[counter].set_ylim(-5,0)
        axs[counter].legend(loc='best', fontsize='xx-large',frameon=True)
        for axis in ['top','bottom','left','right']:
            axs[counter].spines[axis].set_linewidth(lw)
            
    cbar_ax = fig.add_axes([0.82, 0.32, 0.01, 0.36])
    cbar_ax.tick_params(labelsize=fs)
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('$log(m_* / \mathrm{M}_\odot)$', fontsize=fs)
    fig.subplots_adjust(right=0.8)
            
    if file is not None:
            plt.savefig(file)

    plt.show()

def plot_baryon_efficiency_2(df, compare_list, fs = 25, lw=3, file=None):
    redshift = [0, 0.1, 0.5, 1, 2]
    zmin = [0, 0.09, 0.45, 0.99, 1.95]
    zmax = [0.0000001, 0.11, 0.55, 1.01, 2.05]
    xticks = [11, 12, 13, 14]
    yticks = [-1, -2, -3, -4]
    fig, axs = plt.subplots(len(compare_list), 5, figsize=(24,32),gridspec_kw={'hspace': 0, 'wspace': 0}, sharey='all',sharex='all')
    for counter2, compare in enumerate(compare_list):
        images = []
        for counter, i in enumerate(redshift):
            
            df_select = df[(df['Redshift'] >= zmin[counter]) & (df['Redshift'] < zmax[counter])]
            t = df_select[compare]
            halo_mass_peak = np.array(df_select['Halo_peak_mass'])
            ratio = np.log10(np.array(df_select['SFR']) / np.array(df_select['Halo_Growth_rate']))
            images.append(axs[counter2,counter].scatter(halo_mass_peak, ratio, linewidths=.01, label ='z = {}'.format(redshift[counter]),c=t, cmap = 'jet'))
            
            if counter == 0:
                axs[counter2,counter].set_ylabel('$\.m_* / \.M_h$', fontsize = fs)
            
            #axs[counter2,counter].set_aspect('equal')
            axs[counter2,counter].set_xlabel('$\log (M_h / \mathrm{M}_\odot)$', fontsize = fs)
            axs[counter2,counter].tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
            axs[counter2,counter].tick_params(width=lw*1.0, length=4*lw, which='major', direction='in', pad=fs/2)
            axs[counter2,counter].set_xticks(xticks)
            axs[counter2,counter].set_xlim(10,14.5)
            axs[counter2,counter].set_yticks(yticks)
            axs[counter2,counter].set_ylim(-5,0)
            #axs[counter2,counter].legend(loc='upper right', fontsize='xx-large')
            axs[counter2,counter].annotate('$z={:.1f}$'.format(redshift[counter]), 
            xy=(1.0-0.05, 0.95), xycoords='axes fraction', size=fs*0.9, ha='right', va='top')
            #bbox=dict(boxstyle='round', fc='w'))
            for axis in ['top','bottom','left','right']:
                axs[counter2,counter].spines[axis].set_linewidth(lw)
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        if compare_list[compare] == 'log':
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

        for im in images:
            im.set_norm(norm)
        cbar_ax = fig.add_axes([0.82, 1-(0.108*(counter2)+0.21), 0.01, 0.5/len(compare_list)])
        if counter2 == 6:
            cbar_ax.tick_params(labelsize=fs/1.3)
        else:
            cbar_ax.tick_params(labelsize=fs/1.3)
        if counter2 == 6:
            cbar = fig.colorbar(images[0], cax=cbar_ax)
        else:
            cbar = fig.colorbar(images[0], cax=cbar_ax)
        #cbar = fig.colorbar(images[0], cax=cbar_ax)
#         names1 = ['log($m_\star/ M_\odot$)','log($\dot m_\star/ M_\odot yr^{-1}$)',
#                   'log($M_h/ M_\odot$)','log($M_p/ M_\odot$)',
#                   'log($\dot{M}_h/ M_\odot yr^{-1}$)','log($\dot{M}_p/ M_\odot yr^{-1}$)',
#                   'log($c$)']
        names1 = ['log($m_\star/ M_\odot$)','$\dot m_\star [M_\odot yr^{-1}]$',
                  'log($M_h/ M_\odot$)','log($M_p/ M_\odot$)',
                  '$\dot{M}_h [M_\odot yr^{-1}]$','$\dot{M}_p [M_\odot yr^{-1}]$',
                  '$c$']
        cbar.set_label(names1[counter2], fontsize=fs/1.3)
#         if compare_list[compare] == 'log':
#             cbar.set_label('log(' + compare + ')', fontsize=fs/2)
#         else:
#             cbar.set_label(compare, fontsize=fs/2)
 
    fig.subplots_adjust(right=0.8)
            
    if file is not None:
            plt.savefig(file,dpi=100, bbox_inches = 'tight',
    pad_inches = 0.1)

    plt.show()

def plot_main_branches(df_RNN, df_NN, fs = 15, lw=3,file=None):
    fig, axs = plt.subplots(len(df_RNN), 2, figsize=(15,20),gridspec_kw={'hspace': 0, 'wspace': 0.2})
    mstar  = np.linspace(5.0,12.0,1000)
    j=-1
    for i in range(len(df_RNN)):
        axs[i, 0].scatter(df_RNN[i]['mstar_integrated'],df_RNN[i]['Stellar_mass'], color='b',  label='RNN')
        # c=df_RNN[i]['Scale'],cmap='winter',
        axs[i, 0].scatter(df_NN[i]['mstar_integrated'],df_NN[i]['Stellar_mass'], color='r', label='NN')
        # c=df_NN[i]['Scale'],cmap='autumn'
        axs[i, 0].set_xlabel('$\ (m_*)_\mathrm{integrated}$', fontsize = fs)
        axs[i, 0].set_ylabel('$\ (m_*)_\mathrm{label}$', fontsize = fs)
        axs[i, 0].plot(mstar,mstar,'black')
        if i == 0:
            axs[i, 0].legend()
        axs[j+1, 1].scatter(df_RNN[i]['mstar_integrated'],df_RNN[i]['mstar_pred'], color='b', label='RNN')
        axs[j+1, 1].scatter(df_NN[i]['mstar_integrated'],df_NN[i]['mstar_pred'], color='r', label='NN')
        axs[j+1, 1].set_xlabel('$\ (m_*)_\mathrm{integrated}$', fontsize = fs)
        axs[j+1, 1].set_ylabel('$\ (m_*)_\mathrm{prediction}$', fontsize = fs)
        axs[j+1, 1].plot(mstar,mstar,'black')
        if i == 0:
            axs[j+1, 1].legend() 
        j += 1
        
    fig.suptitle('Comparison of RNN and NN results several for main branches', fontsize=20, y =0.92)

    if file is not None:
            plt.savefig(file)
    plt.show()
    
def plot_main_sequence_panel_RNN_RL(
    fig,
    X,
    y,
    redshift_min,
    redshift_max,
    iscale,
    halo_features_used,
    galaxy_labels_used,
    H0=70.0,
    Om0=0.3,
    plot_obs=True,
    axis=[7.0,12.9,-6.20,1.8],
    fs=22,
    lw=3,
    frac=None,
    nxpanel=1,
    nypanel=1,
    ipanel=1,
    barposition=[0.92,0.18,0.04,0.65],
    modelname=None,
    showredshift=True,
    Unscale=True,
    seed=42
    ):
    
    if Unscale == False:
        gal_pred = y
        halos_unscaled = X
    else:
        gal_pred       = np.array(get_targets_unscaled(y,galaxy_labels_used))
        halos_unscaled = np.array(get_features_unscaled(X,halo_features_used))

    np.random.seed(seed=seed)
    isample = np.random.permutation(gal_pred.shape[0]) 
    gal_pred = gal_pred[isample]
    halos_unscaled = halos_unscaled[isample]
    
    if frac is not None:
        isample        = np.random.choice(gal_pred.shape[0],int(gal_pred.shape[0]*frac))
        gal_pred       = gal_pred[isample]
        halos_unscaled = halos_unscaled[isample]

    mstar = gal_pred[:,0]
    sfr   = gal_pred[:,1]
    apeak = halos_unscaled[:,5]
    z     = 1./halos_unscaled[:,iscale]-1.
    iselect = np.logical_and(halos_unscaled[:,iscale] >= 1./(redshift_max+1.)-0.01, halos_unscaled[:,iscale] < 1./(redshift_min+1.)+0.01)
    mstar = mstar[iselect] # iselect is an array contiang of boolean values which removes all False values
    sfr   = sfr[iselect]
    z     = z[iselect]
    apeak = apeak[iselect]
    if plot_obs:
        #Set minimum SSFR
        ssfrmin = 1.e-12
        sfr[np.where(sfr/10.**mstar<ssfrmin)] = ssfrmin*10.**mstar[np.where(sfr/10.**mstar<ssfrmin)]
    sfr   = np.log10(sfr)
    #Add scatter to the SFR to get observed SFR
    if plot_obs:
        cosmo = ac.FlatLambdaCDM(H0=H0,Om0=Om0)
        tcosmic = cosmo.age([z]).value[0]
        thre  = np.log10(0.3/tcosmic)-9.0
        ssfr  = sfr - mstar
        sig   = 0.2*(np.tanh(1.0*(thre-ssfr.flatten()))*0.5+0.5)+0.1
        np.random.seed(seed=42)
        rangau = np.random.normal(loc=0.0,scale=sig,size=sfr.size)
        sfrobs = sfr+rangau
        #Add scatter to the stellar mass to get observed mstar
        np.random.seed(seed=42)
        rangau = np.random.normal(loc=0.0,scale=0.1,size=mstar.size)
        mstarobs = mstar+rangau

    ax  = plt.subplot(nypanel,nxpanel,ipanel)
    ax.axis(axis)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    
    if showredshift:
        ax.annotate('$z={:.2f}-{:.2f}$'.format(redshift_min,redshift_max), xy=(1.0-0.05, 0.05), xycoords='axes fraction', size=fs*0.9, ha='right', va='bottom')

    if modelname is 'Emerge':
        ax.annotate(modelname, xy=(0.05, 1.0-0.05), xycoords='axes fraction', size=fs*1.1, ha='left', va='top')
    else:
        ax.annotate(modelname, xy=(1.0-0.05, 0.05), xycoords='axes fraction', size=fs*1.1, ha='right', va='bottom')       

    if plot_obs:
        ln = ax.scatter(mstarobs,sfrobs,s=2,c=apeak*(0.5*(redshift_max+redshift_min)+1.0),cmap=plt.cm.jet_r, vmin=0.0, vmax=1.0)
    else:
        ln = ax.scatter(mstar,sfr,s=2,c=apeak*(0.5*(redshift_max+redshift_min)+1.0),cmap=plt.cm.jet_r, vmin=0.0, vmax=1.0)
    ax.set_xlabel('$\log (m_* / \mathrm{M}_\odot)$', size = fs)
    ax.set_ylabel('$\log (\Psi / \mathrm{M}_\odot\mathrm{yr}^{-1})$', size = fs)
    
    if ipanel==1:
        barpos=fig.add_axes(barposition)
        cbar = plt.colorbar(ln,cax=barpos, ticks=[0.0,0.2,0.4,0.6,0.8,1.0])
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label('$a_\mathrm{p} \, / \, a$', fontsize=fs)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')
    
    ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
        
    if (ipanel-1) % nxpanel  > 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
    if int((ipanel-1)/nxpanel) < nypanel-1:
        ax.set_xticklabels([])
        ax.set_xlabel('')
        
        
def plot_shmr_panel_RNN_RL(
    fig,
    X,
    y,
    redshift_min,
    redshift_max,
    iscale,
    halo_features_used,
    galaxy_labels_used,
    axis=[10.0,15.0,7.0,12.0],
    vmin=-12.1,
    vmax=-8.5,
    fs=22,
    lw=3,
    frac=None,
    nxpanel=1,
    nypanel=1,
    ipanel=1,
    barposition=[0.92,0.18,0.04,0.65],
    modelname=None,
    showredshift=True,
    Unscale=True,
    seed=42
    ):
    
    #Unscale the prediction
    if Unscale == False:
        gal_pred = y
        halos_unscaled = X
    else:
        gal_pred       = np.array(get_targets_unscaled(y,galaxy_labels_used))
        halos_unscaled = np.array(get_features_unscaled(X,halo_features_used))

    np.random.seed(seed=seed)
    isample = np.random.permutation(gal_pred.shape[0])
    gal_pred = gal_pred[isample]
    halos_unscaled = halos_unscaled[isample]
    
    if frac is not None:
        isample        = np.random.choice(gal_pred.shape[0],int(gal_pred.shape[0]*frac))
        gal_pred       = gal_pred[isample]
        halos_unscaled = halos_unscaled[isample]
    
    mhalo = halos_unscaled[:,2]
    mstar = gal_pred[:,0]
    sfr   = gal_pred[:,1]
    z     = 1./halos_unscaled[:,iscale]-1.
    iselect = np.logical_and(halos_unscaled[:,iscale] >= 1./(redshift_max+1.)-0.01, halos_unscaled[:,iscale] < 1./(redshift_min+1.)+0.01)
    mhalo = mhalo[iselect]
    mstar = mstar[iselect]
    sfr   = sfr[iselect]
    z     = z[iselect]
    
    ax  = plt.subplot(nypanel,nxpanel,ipanel)
    ax.axis(axis)
    ax.tick_params(axis='both', direction='in', which = 'both', bottom=True, top=True, left=True, right=True, labelsize=fs)
    
    if showredshift:
        ax.annotate('$z={:.2f}-{:.2f}$'.format(redshift_min,redshift_max), xy=(1.0-0.05, 0.05), xycoords='axes fraction', size=fs*0.9, ha='right', va='bottom')

    if modelname is not None:
        ax.annotate(modelname, xy=(0.05, 1.0-0.05), xycoords='axes fraction', size=fs*1.1, ha='left', va='top')

    ln = ax.scatter(mhalo,mstar,s=3,c=np.log10(sfr)-mstar,cmap=plt.cm.jet_r, vmin=vmin, vmax=vmax)
    ax.set_xlabel('$\log (M_\mathrm{p} / \mathrm{M}_\odot)$', size = fs)
    ax.set_ylabel('$\log (m_* / \mathrm{M}_\odot)$', size = fs)
    
    if ipanel==1:
        barpos=fig.add_axes(barposition)
        cbar = plt.colorbar(ln,cax=barpos, ticks=[-12.,-11.,-10.,-9.])
        cbar.ax.tick_params(labelsize=fs)
        cbar.set_label('sSFR', fontsize=fs)
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

    ax.tick_params(width=lw*1.0, length=2*lw, which='major', direction='in', pad=fs/2)
    ax.tick_params(width=lw*0.5, length=1*lw, which='minor', direction='in', pad=fs/2)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(lw)
        
    if (ipanel-1) % nxpanel  > 0:
        ax.set_yticklabels([])
        ax.set_ylabel('')
        
    if int((ipanel-1)/nxpanel) < nypanel-1:
        ax.set_xticklabels([])
        ax.set_xlabel('')