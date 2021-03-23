import numpy as np
import tensorflow as tf
import pandas as pd

#Special log10 function for tensorflow
def tflog10(x):
    return tf.math.log(x)/tf.math.log(tf.constant(10.0,dtype=x.dtype))

#Compute standard deviation
def sigma(x):
    if (x.size>1):
        return np.sqrt(((x-x.mean())**2.0).sum()/(x.size-1))
    else:
        return np.nan

#Compute uncertainty of the mean
def sigma_mean(x):
    if (x.size>1):
        return np.sqrt(((x-x.mean())**2.0).sum()/(x.size-1))/np.sqrt(x.size)
    else:
        return np.nan

#This is a minmax scaler that scales everything between vmin and 1
#If features are out of bounds, they will get capped
#If the log flag is enabled, the scaling will be done in log space (and returned)
def minmax_scaler(x, xmin, xmax, vmin, log=False):    
    xmin = tf.constant(xmin,dtype=x.dtype)
    xmax = tf.constant(xmax,dtype=x.dtype)    
    x    = tf.clip_by_value(x,clip_value_min=xmin,clip_value_max=xmax)
    if log:
        x = tflog10(x)
        x = (x - tflog10(xmin))/(tflog10(xmax)-tflog10(xmin))*(1.0-vmin)+vmin
    else:        
        x = (x - xmin)/(xmax-xmin)*(1.0-vmin)+vmin
    return x

#This simply inverts the minmax scaling to get the original range
#If something has been capped, this function obviously does not restore that
def minmax_scaler_inv(x, xmin, xmax, vmin, log=False):
    xmin = tf.constant(xmin,dtype=x.dtype)
    xmax = tf.constant(xmax,dtype=x.dtype)    
    if log: 
        x = (x - vmin) / (1.0-vmin) * (tflog10(xmax)-tflog10(xmin)) + tflog10(xmin)
        x = tf.math.pow(10.0,x)
    else:
        x = (x - vmin) / (1.0-vmin) * (xmax-xmin) + xmin        
    return x

#This scales each halo feature and glues it together into a 2d numpy array
def get_features_scaled(halos,halo_features_used,minvalue=0.01):
    X = tf.constant([[]],dtype=halos.dtype,shape=(halos.shape[0],0))
    for i in range(len(halo_features_used)):
        if halo_features_used[i][1] == 'categorical':
            feature = halos[:,i]
            X = tf.concat([X,tf.reshape(feature,shape=(feature.shape[0],1))],1)
        else:
            feature = minmax_scaler(
                halos[:,i],
                halo_features_used[i][1],
                halo_features_used[i][2],
                minvalue,
                log=halo_features_used[i][3]
            )
            X = tf.concat([X,tf.reshape(feature,shape=(feature.shape[0],1))],1)
    return X

#This unscales a halo 2d numpy array to the original range
def get_features_unscaled(X_scaled,halo_features_used,minvalue=0.01):
    halo = tf.constant([[]],dtype=X_scaled.dtype,shape=(X_scaled.shape[0],0))
    for i in range(len(halo_features_used)):
        if halo_features_used[i][1] == 'categorical':
            feature = X_scaled[:,i]
            halo = tf.concat([halo,tf.reshape(feature,shape=(feature.shape[0],1))],1)   
        else: 
            feature = minmax_scaler_inv(
                X_scaled[:,i],
                halo_features_used[i][1],
                halo_features_used[i][2],
                minvalue,
                log=halo_features_used[i][3]
            )
            halo = tf.concat([halo,tf.reshape(feature,shape=(feature.shape[0],1))],1)
    return halo

#This scales the labels and returns a galaxy 2d numpy array
def get_labels_scaled(galaxies,galaxy_labels_used,minvalue=0.01):
    y = tf.constant([[]],dtype=galaxies.dtype,shape=(galaxies.shape[0],0))
    for i in range(len(galaxy_labels_used)):
        label = minmax_scaler(
            galaxies[:,i],
            galaxy_labels_used[i][1],
            galaxy_labels_used[i][2],
            minvalue,
            log=galaxy_labels_used[i][3]
        )
        y = tf.concat([y,tf.reshape(label,shape=(label.shape[0],1))],1)
    return y

#This restores a scaled target array back to the original range
def get_targets_unscaled(y_scaled,galaxy_labels_used,minvalue=0.01):
    gal = tf.constant([[]],dtype=y_scaled.dtype,shape=(y_scaled.shape[0],0))
    for i in range(len(galaxy_labels_used)):
        target = minmax_scaler_inv(
            y_scaled[:,i],
            galaxy_labels_used[i][1],
            galaxy_labels_used[i][2],
            minvalue,
            log=galaxy_labels_used[i][3]
        )
        gal = tf.concat([gal,tf.reshape(target,shape=(target.shape[0],1))],1)
    return gal


#This splits the feature and label arrays into training, validation and test set
def split_data(X, y, w, vali_ratio=0.1, test_ratio=0.1):

    #Get the range for each set
    total_size = X.shape[0]
    split_vali = int(X.shape[0]*(1.0-vali_ratio-test_ratio))
    split_test = int(X.shape[0]*(1.0-test_ratio))

    #Get a shuffled index array to randomise the arrays (both in the same way of course)
    tf.random.set_seed(42)
    index = tf.range(0,total_size,1)
    index = tf.random.shuffle(index)

    #Split feature array
    X_shuffled = tf.gather(X,index)
    X_train    = X_shuffled[0:split_vali]
    X_vali     = X_shuffled[split_vali:split_test]
    X_test     = X_shuffled[split_test:]

    #Split label array
    y_shuffled = tf.gather(y,index)
    y_train    = y_shuffled[0:split_vali]
    y_vali     = y_shuffled[split_vali:split_test]
    y_test     = y_shuffled[split_test:]

    #Split weight array
    w_shuffled = w[np.array(index)]
    w_train    = w_shuffled[0:split_vali]
    w_vali     = w_shuffled[split_vali:split_test]
    w_test     = w_shuffled[split_test:]
    
    return X_train,X_vali,X_test,y_train,y_vali,y_test,w_train,w_vali,w_test,index