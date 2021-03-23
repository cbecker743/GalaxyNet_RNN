"""
"""
import numpy as np
import pandas as pd
import itertools
import pickle
import h5py as hdf5
import tensorflow as tf
from scaling import *
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

#from .source_halo_selection import get_source_bin_from_target_bin

__all__ = ('load_file', 'load_data')

def load_file(path,filename,halo_features_used, galaxy_labels_used, load_galaxies=True, dtypetf=tf.float32):
    """ Load features and labels from an existing halo-galaxy catalogue.

    Parameters
    ----------
    path: string
        String that defines the path to the catalogue file

    filename: string
        String that defines the file name of the catalogue

    halo_features_used: list


    haloprop_and_bins_dict : dict
        Python dictionary storing the collection of halo properties and bins.
        Each key should be the name of the property to be binned;
        each value should be a two-element tuple storing two ndarrays,
        the first with shape (num_halos, ), the second with shape (nbins, ),
        where ``nbins`` is allowed to vary from property to property.
    Returns
    -------
    cell_ids : ndarray
        Numpy integer array of shape (num_halos, ) storing the integer of the
        (possibly multi-dimensional) bin of each halo.
    Examples
    --------
    In this example, we bin our halos simultaneously by mass and concentration:
    >>> num_halos = 50
    >>> mass = 10**np.random.uniform(10, 15, num_halos)
    >>> conc = np.random.uniform(1, 25, num_halos)
    >>> num_bins_mass, num_bins_conc = 12, 11
    >>> mass_bins = np.logspace(10, 15, num_bins_mass)
    >>> conc_bins = np.logspace(1.5, 20, num_bins_conc)
    >>> cell_ids = halo_bin_indices(mass=(mass, mass_bins), conc=(conc, conc_bins))
    In this case, all values in the ``cell_ids`` array
    will be in the interval [0, num_bins_mass*num_bins_conc).
    """
    print('Loading '+filename)
    file = hdf5.File(path+'/'+filename,'r')
    filekeys = [key for key in file.keys()]
    #print(filekeys)
    dataset = file[filekeys[0]]
    #print(dataset.dtype)
    #Get all the meta-data in the HDF5 files
    attkeys   = dataset.attrs.keys()
    attvalues = dataset.attrs.values()
    attrs = dict(zip(attkeys,attvalues))
    #print(attkeys)
    z     = 1./attrs['Scale Factor']-1.0
    Lbox  = attrs['Box Size']
    H0    = attrs['Hubble Parameter'] * 100.0
    Om0   = attrs['Omega_0']
    meta  = tf.constant([[z,Lbox,H0,Om0]],dtype=dtypetf)
    halos = tf.constant([[]],dtype=dtypetf,shape=(dataset.shape[0],0))
    for i in range(len(halo_features_used)):
        if halo_features_used[i][0] == 'Scale_Factor':
            halos = tf.concat([halos,tf.ones((dataset.shape[0],1),dtype=dtypetf)*attrs['Scale Factor']],1)
        else:
            halos = tf.concat([halos,tf.constant(dataset[halo_features_used[i][0]],dtype=dtypetf,shape=(dataset.shape[0],1))],1)
    if load_galaxies:    
        galaxies = tf.constant([[]],dtype=dtypetf,shape=(dataset.shape[0],0))
        for i in range(len(galaxy_labels_used)):
            galaxies = tf.concat([galaxies,tf.constant(dataset[galaxy_labels_used[i][0]],dtype=dtypetf,shape=(dataset.shape[0],1))],1)
        return halos, galaxies, meta
    else:
        return halos, meta



def load_data(path,files,halo_features_used,galaxy_labels_used,load_galaxies=True,dtypetf=tf.float32):
    halos    = tf.constant([[]],dtype=dtypetf,shape=(0,len(halo_features_used)))
    galaxies = tf.constant([[]],dtype=dtypetf,shape=(0,len(galaxy_labels_used)))
    meta     = tf.constant([],  dtype=dtypetf,shape=(0,4))
    for file in files:
        if load_galaxies:
            h, g, m  = load_file(path, file, halo_features_used, galaxy_labels_used,load_galaxies=load_galaxies)
            galaxies = tf.concat([galaxies,g],0)
        else:
            h, m = load_file(path, file, halo_features_used, galaxy_labels_used,load_galaxies=load_galaxies)
        halos    = tf.concat([halos,h],0)
        meta     = tf.concat([meta,m],0)
    if load_galaxies:
        return halos,galaxies,meta
    else:
        return halos,meta

#Load emerge galaxy catalogue
def load_positions_file(path,filename,dtypetf=tf.float32):
    print('Loading positions from '+filename)
    file = hdf5.File(path+'/'+filename,'r')
    filekeys = [key for key in file.keys()]
    dataset = file[filekeys[0]]
    x = tf.constant(dataset['X_pos'],dtype=dtypetf,shape=(dataset.shape[0],1))
    y = tf.constant(dataset['Y_pos'],dtype=dtypetf,shape=(dataset.shape[0],1))
    z = tf.constant(dataset['Z_pos'],dtype=dtypetf,shape=(dataset.shape[0],1))
    pos = tf.concat([x,y,z],axis=1)
    return pos


def load_positions(path,files,dtypetf=tf.float32):
    positions = tf.constant([[]],dtype=dtypetf,shape=(0,3))
    for file in files:
        p  = load_positions_file(path, file)
        positions = tf.concat([positions,p],axis=0)
    return positions


#Load emerge galaxy catalogue
def load_posvel_file(path,filename,dtypetf=tf.float32):
    print('Loading positions and velocities from '+filename)
    file = hdf5.File(path+'/'+filename,'r')
    filekeys = [key for key in file.keys()]
    dataset = file[filekeys[0]]
    x = tf.constant(dataset['X_pos'],dtype=dtypetf,shape=(dataset.shape[0],1))
    y = tf.constant(dataset['Y_pos'],dtype=dtypetf,shape=(dataset.shape[0],1))
    z = tf.constant(dataset['Z_pos'],dtype=dtypetf,shape=(dataset.shape[0],1))
    u = tf.constant(dataset['X_vel'],dtype=dtypetf,shape=(dataset.shape[0],1))
    v = tf.constant(dataset['Y_vel'],dtype=dtypetf,shape=(dataset.shape[0],1))
    w = tf.constant(dataset['Z_vel'],dtype=dtypetf,shape=(dataset.shape[0],1))    
    pos = tf.concat([x,y,z],axis=1)
    vel = tf.concat([u,v,w],axis=1)
    return pos,vel


def load_posvel(path,files,dtypetf=tf.float32):
    positions  = tf.constant([[]],dtype=dtypetf,shape=(0,3))
    velocities = tf.constant([[]],dtype=dtypetf,shape=(0,3))
    for file in files:
        p,v = load_posvel_file(path, file)
        positions  = tf.concat([positions,p],axis=0)
        velocities = tf.concat([velocities,v],axis=0)
    return positions, velocities


#Get weights from stellar mass function to scale loss function
def get_loss_weights(galaxies,galaxy_labels_used,norm=100.0,dm=0.2):
    dm        = 0.2
    bins      = np.arange(int(galaxy_labels_used[0][1]/dm)*dm-dm/2,int(galaxy_labels_used[0][2]/dm+1.5)*dm,dm)
    smf,mstar = np.histogram(galaxies[:,0],bins=bins)
    mstar     = 0.5*(mstar[0:-1]+mstar[1:])
    weights   = 1./np.interp(galaxies[:,0], mstar, smf)
    weights   = norm*weights/weights.mean()
    return weights.astype(np.float32)


def load_single_file_to_panda_df(filename, path='P200'):
    file = hdf5.File(path+'/'+filename,'r')
    filekeys = [key for key in file.keys()]
    dataset = file[filekeys[0]]
    dataset2 = np.array(dataset)
    df = pd.DataFrame(dataset2)
    return df


def load_MergerTree_to_panda_df(filename, path):
    file = hdf5.File(path+'/'+filename,'r')
    filekeys = [key for key in file.keys()]
    MergerTree = file[filekeys[0]]
    Snapshots = file[filekeys[1]]
    subfilekeysMerg = [key for key in MergerTree.keys()]
    Galaxy_df = pd.DataFrame(np.array(MergerTree[subfilekeysMerg[0]]))
    Offset_df = pd.DataFrame(np.array(MergerTree[subfilekeysMerg[1]]))
    subfilekeysSnap = [key for key in Snapshots.keys()]
    Snapshots_df = pd.DataFrame(np.array(Snapshots[subfilekeysSnap[0]]))
    return Galaxy_df, Offset_df, Snapshots_df


def get_whole_dataset(gal_list, columns_active, columns_used, columns_scaled, HGRP=False,HPMS=True, Main_merger=False, Merger=False, Main_galaxies=False, Type=False):
    # get whole_dataset with scaled columns added at the end
    whole_dataset = pd.concat(gal_list, ignore_index=True)
    if HGRP == True:
        with open('pkl_Data/Features/HGRP.pkl', 'rb') as f:
            HGRP_list = pickle.load(f)
        whole_dataset['Halo_growth_peak'] = HGRP_list
    if HPMS == True:
        with open('pkl_Data/Features/HPMS.pkl', 'rb') as f:
            HPMS_list = pickle.load(f)
        whole_dataset['Scale_half_peak_mass'] = HPMS_list
    if Merger == True:
        with open('pkl_Data/Features/merger.pkl', 'rb') as f:
            merger_list = pickle.load(f)
        whole_dataset['Merger'] = merger_list
    if Main_galaxies == True:
        with open('pkl_Data/Features/main_galaxies.pkl', 'rb') as f:
            main_galaxies_list = pickle.load(f)
        whole_dataset['Main_galaxies'] = main_galaxies_list
    if Type == True:
        with open('pkl_Data/Features/central.pkl', 'rb') as f:
            central_list = pickle.load(f)
        whole_dataset['Central'] = central_list
        with open('pkl_Data/Features/satellite.pkl', 'rb') as f:
            satellite_list = pickle.load(f)
        whole_dataset['Satellite'] = satellite_list
        with open('pkl_Data/Features/orphan.pkl', 'rb') as f:
            orphan_list = pickle.load(f)
        whole_dataset['Orphan'] = orphan_list
    actives = tf.convert_to_tensor(np.array(whole_dataset[columns_active]))
    active_columns_scaled = pd.DataFrame(np.array(get_features_scaled(actives, columns_used)),columns=columns_scaled)
    whole_dataset_scaled = pd.concat([whole_dataset, active_columns_scaled], axis=1)
    return whole_dataset_scaled


def split_dataset_into_MergerTrees(galaxies, offset):
    length_list = list(offset['Offset'])
    l_mod = length_list + [max(length_list)+offset['Ngal'][-1:].tolist()[0]]
    list_of_trees = [galaxies.iloc[l_mod[n]:l_mod[n+1]] for n in range(len(l_mod)-1)]
    return list_of_trees


def Reduce_Trees_to_Main_Branch(Trees):
    new_Trees = []
    for tree in Trees:
        start_id = tree[0:1]['ID'].tolist()[0]
        full_list = tree.index.tolist()
        in_list = []
        for index, row in tree.iterrows():
            if start_id + 1 == row[5] or row[5] == 0 and start_id - 1 == row[3]:
                start_id = start_id +1
                in_list.append(index)
        out_list = [i for i in full_list if i not in in_list]
        final_df = tree.drop(out_list)
        new_Trees.append(final_df)
    print('##########end_of_function############')
    return new_Trees


def prepare_features(Trees, halo_columns_active, minvalue=0.01):
    # Scales,  Zero Pads and puts Data into list containing all  
    halos = []
    positions = []
    for tree in Trees:
        if len(tree) != 0:
            df = tree[halo_columns_active]
            index_list = df.index.values.tolist() 
            top = np.round(max(tree['Scale']),2)
            bottom = np.round(min(tree['Scale']),2)
            diff_to_top = np.round(100 - top*100,2)
            diff_to_bottom = np.round(93 - diff_to_top - (top - bottom)*100,2)
            array_top = np.arange(min(index_list)-diff_to_top,min(index_list),1)
            array_bottom = np.arange(max(index_list)+1,max(index_list)+diff_to_bottom,1)
            new_index_list = list(itertools.chain(array_top, index_list, array_bottom))
            df_2 = df.reindex(new_index_list,  fill_value=minvalue)
            df_2['Scale'] = np.arange(0.08,1.01,0.01)[::-1]
            full_scale_scaled = get_full_a_scaled(Trees)
            df_2['Scale'] = full_scale_scaled
            df_3 = df_2.drop(['Scale_scaled'], axis=1)
            df_4 = df_3.iloc[::-1]
            df_positions = df_4[['X_pos','Y_pos','Z_pos']]
            df_5 =  df_4.drop(['X_pos','Y_pos','Z_pos'], axis=1)
            array_1 = np.array(df_5)
            halos.append(array_1)
            array_2 = np.array(df_positions)
            positions.append(array_2)
    halos_tf = tf.convert_to_tensor(halos)
    positions_tf = tf.convert_to_tensor(positions)
    return halos_tf, positions_tf


def prepare_labels(Trees, galaxy_columns_active, minvalue=0.01):
    # Scales,  Zero Pads and puts Data into list containing all  
    galaxies = []
    weights = []
    for tree in Trees:
        if len(tree) != 0:
            df = tree[galaxy_columns_active]
            index_list = df.index.values.tolist() 
            top = np.round(max(tree['Scale']),2)
            bottom = np.round(min(tree['Scale']),2)
            diff_to_top = np.round(100 - top*100,2)
            diff_to_bottom = np.round(93 - diff_to_top - (top - bottom)*100,2)
            array_top = np.arange(min(index_list)-diff_to_top,min(index_list),1)
            array_bottom = np.arange(max(index_list)+1,max(index_list)+diff_to_bottom,1)
            new_index_list = list(itertools.chain(array_top, index_list, array_bottom))
            df_2 = df.reindex(new_index_list,  fill_value=minvalue)
            df_2['Scale'] = np.arange(0.08,1.01,0.01)[::-1]
            df_3 = df_2.drop(['Scale'], axis=1)
            df_4 = df_3.iloc[::-1]
            df_weights = df_4['weights']
            df_5 =  df_4.drop(['weights'], axis=1)
            array_1 = np.array(df_5)
            galaxies.append(array_1)
            array_2 = np.array(df_weights)
            weights.append(array_2)
    galaxies_tf = tf.convert_to_tensor(galaxies)
    weights_tf = tf.convert_to_tensor(weights)
    return galaxies_tf, weights_tf

def find_tree_indices(tree):
    dfs = []
    tree = tree[::-1]
    for index, row in tree.iterrows():
        in_list = []
        if row[5] == 0:
            last_desc_id = row[3]
            in_list.append(index)
            for index2, row2 in tree.iterrows():
                if row2[1] == last_desc_id and tree.loc[[in_list[-1]]]['ID'].values.tolist()[0] == row2[1] + 1:
                    last_desc_id = row2[3] 
                    in_list.append(index2)
            next_df = tree.drop(in_list)
            dfs.append(in_list)
            tree = next_df
    # this part is just a fix to catch galaxies which merged in first snapshot
    for i in dfs:
        if len(i) == 0:
            for index3, row3 in tree.iterrows():
                i.append(index3)
    return dfs

def split_trees(trees, HGRP=False, HPMS=False):
    dfs =[]
    for tree in trees:
        in_lists = find_tree_indices(tree)
        for liste in in_lists:
            full_list = tree.index.tolist()
            out_list = [i for i in full_list if i not in liste]
            final_tree = tree.drop(out_list)
            dfs.append(final_tree)
    if HGRP == True:
        for tree in dfs:
            HGRP_list = []
            for index, row  in tree[::-1].iterrows():
                if len(HGRP_list) == 0:
                    HGRP = row[12]
                    HGRP_list.append(HGRP)
                else:
                    HGRP = np.max(HGRP_list)
                    HGR = row[12]
                    if HGR > HGRP:
                        HGRP_list.append(HGR)
                    else:
                        HGRP_list.append(HGRP)
            tree['Halo_growth_peak'] = HGRP_list[::-1]
    if HPMS == True:
        for counter, tree in enumerate(dfs):
            HPMS_list = []
            for index, row  in tree[::-1].iterrows():
                HPM = np.log10((10**row[13])/2)
                for index, row  in tree[::-1].iterrows():
                    halo_mass = row[11]
                    if halo_mass >= HPM:
                        HPMS = np.round(row[0],2)
                        HPMS_list.append(HPMS)
                        break
            tree['Scale_half_peak_mass'] = HPMS_list[::-1]
    print('##########end_of_function############')
    return dfs


def get_full_a_scaled(trees):   
    for i in trees:
        if len(i) == 93:
            a_scaled_full = i['Scale_scaled'].tolist()
            break
    return a_scaled_full

def get_cosmo_age(a_scale):
    # returns ages in units of Gyr
    # returns delta between two adjacent array members in Gyr
    cosmo = FlatLambdaCDM(H0=67.81 * u.km / u.s / u.Mpc, Tcmb0=2.726 * u.K, Om0=0.308)
    z  = 1./np.array(a_scale) - 1.0
    cosmoages = cosmo.age(z).value.tolist()
    cosmodeltas =[]
    for counter, i in enumerate(cosmoages):
        if counter < len(cosmoages) - 1:
            diff = cosmoages[counter+1] - i 
            cosmodeltas.append(diff) 
    return np.array(cosmoages), np.array(cosmodeltas)

def cosmo_age(a_scale):
    cosmo = FlatLambdaCDM(H0=67.81 * u.km / u.s / u.Mpc, Tcmb0=2.726 * u.K, Om0=0.308)
    z  = 1./np.array(a_scale) - 1.0
    cosmoages = cosmo.age(z).value.tolist()
    z2  = 1./np.array(a_scale+0.01) - 1.0
    cosmoages2 = cosmo.age(z2).value.tolist()
    cosmodelta = cosmoages2 - cosmoages
    return cosmoages, cosmodelta

def get_data_without_zeropadding(X_vali, y_vali, y_vali_pred, halo_features_used):
    new_X_vali = pd.DataFrame(np.array(X_vali).reshape(len(X_vali)*93,len(halo_features_used)))
    new_y_vali = np.array(y_vali).reshape(len(y_vali)*93,2)
    new_y_vali_pred = np.array(y_vali_pred).reshape(len(y_vali_pred)*93,2)
    
    new_mstar_vali = new_y_vali[:,0]
    new_SFR_vali = new_y_vali[:,1]
    new_mstar_vali_pred = new_y_vali_pred[:,0]
    new_SFR_vali_pred = new_y_vali_pred[:,1]
    
    new_X_vali['new_mstar_vali'] = new_mstar_vali
    new_X_vali['new_SFR_vali'] = new_SFR_vali
    new_X_vali['new_mstar_vali_pred'] = new_mstar_vali_pred
    new_X_vali['new_SFR_vali_pred'] = new_SFR_vali_pred
    
    new_X_vali_2 = new_X_vali[new_X_vali[1] != 0.0 ]
    
    df_y_vali = new_X_vali_2.loc[:, 'new_mstar_vali' : 'new_SFR_vali']
    df_y_vali_pred = new_X_vali_2.loc[:, 'new_mstar_vali_pred' : 'new_SFR_vali_pred']
    df_x_vali = new_X_vali_2.loc[:, :'new_mstar_vali']
    df_x_vali = df_x_vali.drop(labels='new_mstar_vali', axis=1)
    
    X_vali_original = tf.convert_to_tensor(np.array(df_x_vali))
    y_vali_original = tf.convert_to_tensor(np.array(df_y_vali))
    y_vali_pred_original = tf.convert_to_tensor(np.array(df_y_vali_pred))
    
    return X_vali_original, y_vali_original, y_vali_pred_original


def load_positions_RNN(dataset,dtypetf=tf.float32):
    x = tf.constant(dataset['X_pos'],dtype=dtypetf,shape=(dataset.shape[0],1))
    y = tf.constant(dataset['Y_pos'],dtype=dtypetf,shape=(dataset.shape[0],1))
    z = tf.constant(dataset['Z_pos'],dtype=dtypetf,shape=(dataset.shape[0],1))
    pos = tf.concat([x,y,z],axis=1)
    return pos

def data_without_zeropadding_RL(X, y, y_pred, galaxy_labels_used, halo_features_used, pos):
#   this function reads in data of shape(x,y,z), reshapes it to (x*y,z), removes zero padded values, unscales the 
#   halos and galaxies and returns them as a np.array    
    halo_columns_active = [column_name[0] for column_name in halo_features_used] 
    new_X = pd.DataFrame(np.array(X).reshape(len(X)*93,len(halo_features_used)), columns=halo_columns_active)
    new_pos = np.array(pos).reshape(len(pos)*93,3)
    new_y = np.array(y).reshape(len(y)*93,2)
    new_y_pred = np.array(y_pred).reshape(len(y_pred)*93,2)

    new_mstar_pred = new_y_pred[:,0]
    new_SFR_pred = new_y_pred[:,1]
    new_mstar_pred_original = new_y[:,0]
    new_SFR_pred_original = new_y[:,1]
    new_x_pos = new_pos[:,0]
    new_y_pos = new_pos[:,1]
    new_z_pos = new_pos[:,2]

    new_X['new_mstar_pred'] = new_mstar_pred
    new_X['new_SFR_pred'] = new_SFR_pred
    new_X['new_mstar_orig'] = new_mstar_pred_original
    new_X['new_SFR_orig'] = new_SFR_pred_original
    new_X['new_x_pos'] = new_x_pos
    new_X['new_y_pos'] = new_y_pos
    new_X['new_z_pos'] = new_z_pos

    new_X_zeropad = new_X[new_X['Halo_mass'] != 0.0]

    df_partial_y = tf.convert_to_tensor(np.array(new_X_zeropad[['new_mstar_pred','new_SFR_pred']]))
    df_partial_y_original = tf.convert_to_tensor(np.array(new_X_zeropad[['new_mstar_orig','new_SFR_orig']]))
    df_partial_X = tf.convert_to_tensor(np.array(new_X_zeropad[halo_columns_active]))
    df_partial_pos = tf.convert_to_tensor(np.array(new_X_zeropad[['new_x_pos','new_y_pos','new_z_pos']]))

    galaxies = np.array(get_targets_unscaled(df_partial_y, galaxy_labels_used))
    galaxies_original = np.array(get_targets_unscaled(df_partial_y_original, galaxy_labels_used))
    halos = np.array(get_features_unscaled(df_partial_X, halo_features_used))
    pos = np.array(df_partial_pos)
    
    return galaxies, halos, pos, galaxies_original


def data_without_zeropadding_RL2(X, y_pred, galaxy_labels_used, halo_features_used, pos):
#   this function reads in data of shape(x,y,z), reshapes it to (x*y,z), removes zero padded values, unscales the 
#   halos and galaxies and returns them as a np.array    
    halo_columns_active = [column_name[0] for column_name in halo_features_used] 
    new_X = pd.DataFrame(np.array(X).reshape(len(X)*93,len(halo_features_used)), columns=halo_columns_active)
    new_pos = np.array(pos).reshape(len(pos)*93,3)
    new_y_pred = np.array(y_pred).reshape(len(y_pred)*93,2)

    new_mstar_pred = new_y_pred[:,0]
    new_SFR_pred = new_y_pred[:,1]
    new_x_pos = new_pos[:,0]
    new_y_pos = new_pos[:,1]
    new_z_pos = new_pos[:,2]

    new_X['new_mstar_pred'] = new_mstar_pred
    new_X['new_SFR_pred'] = new_SFR_pred
    new_X['new_x_pos'] = new_x_pos
    new_X['new_y_pos'] = new_y_pos
    new_X['new_z_pos'] = new_z_pos

    new_X_zeropad = new_X[new_X['Halo_mass'] != 0.0]

    df_partial_y = tf.convert_to_tensor(np.array(new_X_zeropad[['new_mstar_pred','new_SFR_pred']]))
    df_partial_X = tf.convert_to_tensor(np.array(new_X_zeropad[halo_columns_active]))
    df_partial_pos = tf.convert_to_tensor(np.array(new_X_zeropad[['new_x_pos','new_y_pos','new_z_pos']]))

    galaxies = np.array(get_targets_unscaled(df_partial_y, galaxy_labels_used))
    halos = np.array(get_features_unscaled(df_partial_X, halo_features_used))
    pos = np.array(df_partial_pos)
    
    return galaxies, halos, pos


def get_HGRP(Trees, whole_dataset):
    HGRP = np.zeros(shape=len(whole_dataset))
    for tree in Trees:
        for index, row in tree.iterrows():
            HGRP[index] = row[34] #for HPMS row[50]
    return HGRP

def get_main_galaxies_and_mergers(Trees, whole_dataset):
    main_merger_list = np.zeros(shape=len(whole_dataset))
    merger_list = np.zeros(shape=len(whole_dataset))
    main_galaxies_list = np.zeros(shape=len(whole_dataset))
    main_branches = Reduce_Trees_to_Main_Branch(Trees)
    for tree in main_branches:
        for index, row in tree.iterrows():
            if row['Num_prog'] > 1:
                main_merger_list[index] = 1
            main_galaxies_list[index] = 1
    for tree2 in Trees:
        for index, row in tree2.iterrows():
            if row['Num_prog'] > 1:
                merger_list[index] = 1
    return merger_list, main_merger_list, main_galaxies_list

def get_galaxy_type(whole_dataset):
    central_list = np.zeros(shape=len(whole_dataset))
    satellite_list = np.zeros(shape=len(whole_dataset))
    orphan_list = np.zeros(shape=len(whole_dataset))
    for index, row in whole_dataset.iterrows():
        if row[32] == 0:
            central_list[index] = 1
        elif row[32] == 1:
            satellite_list[index] = 1
        else:
            orphan_list[index] = 1   
    return central_list, satellite_list, orphan_list

def unscale_timeserieses(y,y_pred, galaxy_labels_used):
    df = pd.DataFrame()
    labels = np.array(y).reshape(len(y)*93,2)
    predictions = np.array(y_pred).reshape(len(y_pred)*93,2)

    unscaled_labels = np.array(get_targets_unscaled(labels,galaxy_labels_used))
    unscaled_predictions = np.array(get_targets_unscaled(predictions,galaxy_labels_used))

    df['mstar'] = labels[:,0]
    df['SFR'] = labels[:,1]
    df['mstar_pred'] = predictions[:,0]
    df['SFR_pred'] = predictions[:,1]
    df['mstar_unscaled'] = unscaled_labels[:,0]
    df['SFR_unscaled'] = unscaled_labels[:,1]
    df['mstar_pred_unscaled'] = unscaled_predictions[:,0]
    df['SFR_pred_unscaled'] = unscaled_predictions[:,1]

    m1 = df.eq(0).any(axis=1)
    df.loc[m1] = 0
    #df_labels = np.array(df.loc[:, 'mstar' : 'SFR']).reshape(len(y),93,2)
    #df_predictions = np.array(df.loc[:, 'mstar_pred' : 'SFR_pred']).reshape(len(y),93,2)
    df_unscaled_labels = np.array(df.loc[:, 'mstar_unscaled' : 'SFR_unscaled']).reshape(len(y),93,2)
    df_unscaled_predictions = np.array(df.loc[:, 'mstar_pred_unscaled' : 'SFR_pred_unscaled']).reshape(len(y),93,2)
    
    return  df_unscaled_labels, df_unscaled_predictions

def f_loss(t):
    loss = 0.05*np.log(1 + (t/0.0014))
    return loss

def add_unscaled_predictions(Reduced_trees, unscaled_pred):
    for counter, tree in enumerate(unscaled_pred):
        tree = pd.DataFrame(tree)
        tree = tree.loc[(tree!=0).any(1)]
        tree = np.array(tree)
        Reduced_trees[counter]['mstar_pred'] = tree[:,0][::-1]
        Reduced_trees[counter]['sfr_pred'] = tree[:,1][::-1]
    return Reduced_trees

def create_full_merger_tree(Reduced_trees):
    with open('pkl_Data/mstar_integrated/Preprocess/lengths.pkl', 'rb') as f:
        lengths = pickle.load(f)
    full_trees = []
    merge_list = []
    indices = []
    length = 0
    counter = 0
#### if length file doesnt exist:
#     lengths = []
#     for tree in whole_trees:
#         lengths.append(len(tree))
    for tree2 in Reduced_trees:
        length = length + len(tree2)
        if length < lengths[counter]:
            merge_list.append(tree2)
        else:
            merge_list.append(tree2)
            counter = counter + 1
            concat = pd.concat(merge_list)
            full_trees.append(concat)
            indices2 = []
            for i in merge_list:
                indices2.append(i['ID'].tolist())
            indices.append(indices2)
            merge_list = []
            length = 0
    return full_trees, indices

def calculate_mstar(trees,a_scale, indices=None, exsitu=True): 
    progress_counter = 0 
    final_list = []
    ages, deltas = get_cosmo_age(a_scale)
    if indices == None:
        with open('pkl_Data/mstar_integrated/Preprocess/indices.pkl', 'rb') as f:
            indices = pickle.load(f)
    for tree_counter, tree in enumerate(trees):
        progress_counter += 1
        index_list = indices[tree_counter]
        tree_list = []
        for index, row in tree.iterrows():
            time = ages[a_scale.tolist().index(np.round(row[0],2))]
            identifier_id = row[1]
            mstar_sum = 0
            for counter, value in enumerate(index_list):
                if identifier_id in index_list[counter]:
                    identifier_list = index_list[counter]
            first_position = identifier_list[-1:][0]
            if identifier_id == first_position:
                mstar_sum += row[11] * 1e+9 * cosmo_age(row[0])[1]
            else:
                for index2, row2 in tree.iterrows():
                    ide_i = row2[1]
                    if ide_i in identifier_list:
                        time_i = ages[a_scale.tolist().index(np.round(row2[0],2))]
                        if time_i < time:
                            delta_i = deltas[a_scale.tolist().index(np.round(row2[0],2))]
                            if row[7] <= 1 or exsitu == False:
                                mstar_sum += (row2[11]) * 1e+9 * (delta_i) * (1-f_loss(time-time_i))
                            elif row[7] > 1 and exsitu == True:
                                merg_identifier = []
                                id_merg, desc_merg, coprog_merg = tree[tree['Scale'] == np.round(row['Scale']-0.01,2)]['ID'].tolist(),tree[tree['Scale'] == np.round(row['Scale']-0.01,2)]['Desc_ID'].tolist(),tree[tree['Scale'] == np.round(row['Scale']-0.01,2)]['Coprog_ID'].tolist()
                                for possible_merger in range(len(id_merg)):
                                    if desc_merg[possible_merger] == identifier_id and id_merg[possible_merger] != identifier_id+1:
                                        merg_identifier.append(id_merg[possible_merger])
                                merger_list = [tree[tree['ID'] == i] for i in merg_identifier]
                                for merger in merger_list:                                                                   
                                    mstar_sum += (merger['sfr_pred'].tolist()[0]) * 1e+9 * (delta_i) * (1-f_loss(time-time_i))
                                mstar_sum += (row2[11]) * 1e+9 * (delta_i) * (1-f_loss(time-time_i))
            mstar_sum_log = np.log10(mstar_sum)
            tree_list.append(mstar_sum_log)
        final_list.append(tree_list)
        print('progress {}%'.format((progress_counter/len(trees))*100))
    return final_list 

def calculate_baryon_df(model, X, y, halo_features_used, galaxy_labels_used):
    y_pred = model.predict(X, batch_size=10000)
    X_ori, _, y_pred_ori = get_data_without_zeropadding(X, y, y_pred, halo_features_used)
    
    gal_pred       = np.array(get_targets_unscaled(y_pred_ori,galaxy_labels_used))
    halos_unscaled = np.array(get_features_unscaled(X_ori,halo_features_used))
    
    df = pd.DataFrame()
    df['Redshift'] = 1./halos_unscaled[:,0]-1.
    df['Scale'] = halos_unscaled[:,0] 
    df['Halo_mass'] = halos_unscaled[:,1] 
    df['Halo_peak_mass'] = halos_unscaled[:,2] 
    df['Halo_Growth_rate'] = halos_unscaled[:,3]
    df['Halo_growth_peak'] = halos_unscaled[:,4]
    df['Scale_peak_mass'] = halos_unscaled[:,5]
    df['Concentration'] = halos_unscaled[:,6]
    df['Main'] = halos_unscaled[:,7]
    df['Central'] = halos_unscaled[:,8]
    df['SFR'] = gal_pred[:,1]
    df['Stellar_mass'] = gal_pred[:,0]
    df = df[df['Main'] == 1.0]
    df = df.reset_index(drop=True)

    df_list = []
    split_begin = 0
    split_end = 0
    a_last = 0
    for index, row in df.iterrows():
            a = row[1]
            if a > a_last:
                split_end += 1
                a_last = a
            else:
                a_last = 0
                df_list.append(df[split_begin:split_end])
                split_end += 1
                split_begin = split_end-1
    
    df_list2 = []
    for i in df_list:
        max_index =  i['Halo_mass'].argmax()
        asd = i[max_index:max_index+1]
        df_list2.append(asd)
    
    df_final = pd.concat(df_list2)
    
    return df_final

def get_example_main_branches(trees, number_trees=10, len_trees=1000):
    main_branch_list = []
    tree_list = []
    indices_list = []
    len_list = []
    with open('pkl_Data/mstar_integrated/Preprocess/indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    for counter, tree in enumerate(trees):
        if len(tree) >= len_trees:
            tree_list.append(tree)
            tree2 = tree[tree['Main_galaxies'] == 1.0]
            main_branch_list.append(tree2[::-1])
            len_list.append(len(tree2))
            indices_list.append(indices[counter])
            if len(main_branch_list) == number_trees:
                break
    return tree_list, main_branch_list, indices_list, len_list