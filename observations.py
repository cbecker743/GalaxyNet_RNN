import h5py as hdf5
import numpy as np
import tensorflow as tf

def load_statistics_file(file):
    
    #Load statistics file
    statfile     = hdf5.File(file,'r')
    statfilekeys = [key for key in statfile.keys()]
    uni0         = statfile[statfilekeys[0]]
    
    return uni0

def get_bin_edges(bins_centers):
    
    bin_edges      = np.zeros(bins_centers.size+1)
    bin_edges[:-1] = bin_edges[:-1] + bins_centers * 0.5
    bin_edges[1:]  = bin_edges[1:]  + bins_centers * 0.5
    bin_edges[0]   = 2 * bin_edges[1] - bin_edges[2]
    bin_edges[-1]  = 2 * bin_edges[-2] - bin_edges[-3]
    
    return bin_edges

def average_csfrd_in_z_bins(universe,redshifts):
    
    #Open CSFRD group
    csfrd = universe['CSFRD']
    #Open Data group
    dataset = csfrd['Data']
    datakeys = [key for key in dataset.keys()]
    
    mean  = np.array([])
    sigma = np.array([])
    
    redshift_bin_edges = get_bin_edges(redshifts)

    for iz in range(redshifts.size):
        yy = np.array([])
        ss = np.array([])
        for setnum in datakeys:
            dset = dataset[setnum]
            for point in dset:                
                x = point['Redshift']
                y = point['Csfrd_observed']
                s = point['Sigma_observed']
                i = np.digitize(x,redshift_bin_edges)-1
                if i == iz:
                    yy = np.append(yy,y)
                    ss = np.append(ss,s)                
        if (1./ss**2.0).sum() == 0.0:
            weighted_mean =	np.nan
        else:
            weighted_mean = (yy/ss**2.0).sum()/(1./ss**2.0).sum()
        if (1./ss**2.0).sum() == 0.0:
           error_mean = np.inf
        else:
            error_mean    = 1./np.sqrt((1./ss**2.0).sum())
        mean          = np.append(mean, weighted_mean)
        sigma         = np.append(sigma,error_mean)
        
    output = np.concatenate([mean.reshape(1,-1),sigma.reshape(1,-1)],axis=0)

    return output

def average_smf_in_z_bins(universe,redshifts,stellar_masses):
    
    #Open CSFRD group
    smf = universe['SMF']
    #Open Data group
    dataset = smf['Data']
    datakeys = [key for key in dataset.keys()]
    
    #Get the redshift of each data set
    dset = smf['Sets']
    setzmin = dset['Redshift_min']
    setzmax = dset['Redshift_max']
    setzmean= 0.5*(setzmin+setzmax)
   
    #Get the bin edges for z and mstar
    redshift_bin_edges = get_bin_edges(redshifts)
    mstar_bin_edges    = get_bin_edges(stellar_masses)

    #Setup arrays that hold all SMFs and errors
    mean  = np.empty([0,stellar_masses.size])
    sigma = np.empty([0,stellar_masses.size])
    
    #Go through all redshift bins
    for iz in range(redshifts.size):

        #Initialise mean and sigma for this redshift bin
        mean_z  = np.array([])
        sigma_z = np.array([])
        
        #Initialise arrays that hold all data
        xx = np.array([])
        yy = np.array([])
        ss = np.array([])
        
        #Go over each set
        for iset,setnum in enumerate(datakeys):
            dset = dataset[setnum]
            
            #Get the redshift bin index for this data set
            i    = np.digitize(setzmean[iset],redshift_bin_edges)-1
            
            #If it is the one we're currently looking at append them to the data arrays
            if i == iz:
                smfset = dataset[setnum]
                x = smfset['Stellar_mass']
                y = smfset['Phi_observed']
                s = smfset['Sigma_observed']
                xx = np.append(xx,x)
                yy = np.append(yy,y)
                ss = np.append(ss,s)                
        
        #Get the stellar mass bin index for each point
        ii = np.digitize(xx,mstar_bin_edges)-1

        #Loop over all stellar mass bins
        for imass in range(stellar_masses.size):
            
            #Initialise arrays that store all data in this stellar mass bin
            yyy = np.array([])
            sss = np.array([])
            
            #Loop over the indeces and check if we're in the right stellar mass bin
            for i in range(ii.size):
                if (imass==ii[i]):

                    #Append the data to the corresponding array
                    yyy = np.append(yyy,yy[i])
                    sss = np.append(sss,ss[i])

            #Compute the weighted mean and the error
            if (1./sss**2.0).sum() == 0.0:
                weighted_mean =	np.nan
            else:
                weighted_mean = (yyy/sss**2.0).sum()/(1./sss**2.0).sum()
            if (1./sss**2.0).sum() == 0.0:
                error_mean = np.inf
            else:
                error_mean    = 1./np.sqrt((1./sss**2.0).sum())           
            #If there is no data point in this bin set the mean to 0 (error is inf anyway)
            if np.isnan(weighted_mean): weighted_mean = -np.inf
            #Append it to the mean/sigma arrays for this redshift bin
            mean_z        = np.append(mean_z, weighted_mean)
            sigma_z       = np.append(sigma_z,error_mean)

        #Append the mean/sigma for this redshift bin to the overall mean/sigma
        mean  = np.concatenate([mean, mean_z.reshape(1,-1)],axis=0)
        sigma = np.concatenate([sigma,sigma_z.reshape(1,-1)],axis=0)

        output = np.concatenate([mean.reshape(1,mean.shape[0],mean.shape[1]),sigma.reshape(1,sigma.shape[0],sigma.shape[1])])
        
    return output

def average_fq_in_z_bins(universe,redshifts,stellar_masses):
    
    #Open CSFRD group
    fq = universe['FQ']
    #Open Data group
    dataset = fq['Data']
    datakeys = [key for key in dataset.keys()]
    
    #Get the redshift of each data set
    dset = fq['Sets']
    setzmin = dset['Redshift_min']
    setzmax = dset['Redshift_max']
    setzmean= 0.5*(setzmin+setzmax)
   
    #Get the bin edges for z and mstar
    redshift_bin_edges = get_bin_edges(redshifts)
    mstar_bin_edges    = get_bin_edges(stellar_masses)

    #Setup arrays that hold all FQs and errors
    mean  = np.empty([0,stellar_masses.size])
    sigma = np.empty([0,stellar_masses.size])
    
    #Go through all redshift bins
    for iz in range(redshifts.size):

        #Initialise mean and sigma for this redshift bin
        mean_z  = np.array([])
        sigma_z = np.array([])
        
        #Initialise arrays that hold all data
        xx = np.array([])
        yy = np.array([])
        ss = np.array([])
        
        #Go over each set
        for iset,setnum in enumerate(datakeys):
            dset = dataset[setnum]
            
            #Get the redshift bin index for this data set
            i    = np.digitize(setzmean[iset],redshift_bin_edges)-1
            
            #If it is the one we're currently looking at append them to the data arrays
            if i == iz:
                fqset = dataset[setnum]
                x = fqset['Stellar_mass']
                y = fqset['Fq_observed']
                s = fqset['Sigma_observed']
                xx = np.append(xx,x)
                yy = np.append(yy,y)
                ss = np.append(ss,s)                
        
        #Get the stellar mass bin index for each point
        ii = np.digitize(xx,mstar_bin_edges)-1

        #Loop over all stellar mass bins
        for imass in range(stellar_masses.size):
            
            #Initialise arrays that store all data in this stellar mass bin
            yyy = np.array([])
            sss = np.array([])
            
            #Loop over the indeces and check if we're in the right stellar mass bin
            for i in range(ii.size):
                if (imass==ii[i]):

                    #Append the data to the corresponding array
                    yyy = np.append(yyy,yy[i])
                    sss = np.append(sss,ss[i])

            #Compute the weighted mean and the error
            if (1./sss**2.0).sum() == 0.0:
                weighted_mean =	np.nan
            else:
                weighted_mean = (yyy/sss**2.0).sum()/(1./sss**2.0).sum()
            if (1./sss**2.0).sum() == 0.0:
                error_mean = np.inf
            else:
                error_mean    = 1./np.sqrt((1./sss**2.0).sum())           
            #If there is no data point in this bin set the mean to 0 (error is inf anyway)
            if np.isnan(weighted_mean): weighted_mean = -np.inf
            #Append it to the mean/sigma arrays for this redshift bin
            mean_z        = np.append(mean_z, weighted_mean)
            sigma_z       = np.append(sigma_z,error_mean)

        #Append the mean/sigma for this redshift bin to the overall mean/sigma
        mean  = np.concatenate([mean, mean_z.reshape(1,-1)],axis=0)
        sigma = np.concatenate([sigma,sigma_z.reshape(1,-1)],axis=0)

    output = np.concatenate([mean.reshape(1,mean.shape[0],mean.shape[1]),sigma.reshape(1,sigma.shape[0],sigma.shape[1])])
        
    return output

def average_ssfr_in_z_bins(universe,redshifts,stellar_masses):
    
    #Open SSFR group
    ssfr = universe['SSFR']
    #Open Data group
    dataset = ssfr['Data']
    datakeys = [key for key in dataset.keys()]
       
    #Get the bin edges for z and mstar
    redshift_bin_edges = get_bin_edges(redshifts)
    mstar_bin_edges    = get_bin_edges(stellar_masses)

    #Setup arrays that hold all SSFRs and errors
    mean  = np.empty([0,stellar_masses.size])
    sigma = np.empty([0,stellar_masses.size])
    
    #Go through all redshift bins
    for iz in range(redshifts.size):

        #Initialise mean and sigma for this redshift bin
        mean_z  = np.array([])
        sigma_z = np.array([])
        
        #Initialise arrays that hold all data
        xx = np.array([])
        zz = np.array([])
        yy = np.array([])
        ss = np.array([])
     
        #Go over each set
        for iset,setnum in enumerate(datakeys):
            dset = dataset[setnum]
                        
            z = dset['Redshift']
            x = dset['Stellar_mass']
            y = dset['Ssfr_observed']
            s = dset['Sigma_observed']
            
            #Get the redshift bin index for this data set
            i = np.digitize(z,redshift_bin_edges)-1

            #Loop over all data in this set
            for j in range(z.size):
                #If the data point is in the redshift bin append it to the data arrays
                if iz == i[j]:
                    xx = np.append(xx,x[j])
                    yy = np.append(yy,y[j])
                    ss = np.append(ss,s[j])
                    zz = np.append(zz,z[j])
        
        #Get the stellar mass bin index for each point
        ii = np.digitize(xx,mstar_bin_edges)-1

        #Loop over all stellar mass bins
        for imass in range(stellar_masses.size):
            
            #Initialise arrays that store all data in this stellar mass bin
            yyy = np.array([])
            sss = np.array([])
            
            #Loop over the indices and check if we're in the right stellar mass bin
            for i in range(ii.size):
                if (imass==ii[i]):

                    #Append the data to the corresponding array
                    yyy = np.append(yyy,yy[i])
                    sss = np.append(sss,ss[i])
                    
            #Compute the weighted mean and the error
            if (1./sss**2.0).sum() == 0.0:
                weighted_mean =	np.nan
            else:
                weighted_mean = (yyy/sss**2.0).sum()/(1./sss**2.0).sum()
            if (1./sss**2.0).sum() == 0.0:
                error_mean = np.inf
            else:
                error_mean    = 1./np.sqrt((1./sss**2.0).sum())           
            #If there is no data point in this bin set the mean to 0 (error is inf anyway)
            if np.isnan(weighted_mean): weighted_mean = -np.inf
            #Append it to the mean/sigma arrays for this redshift bin
            mean_z        = np.append(mean_z, weighted_mean)
            sigma_z       = np.append(sigma_z,error_mean)

        #Append the mean/sigma for this redshift bin to the overall mean/sigma
        mean  = np.concatenate([mean, mean_z.reshape(1,-1)],axis=0)
        sigma = np.concatenate([sigma,sigma_z.reshape(1,-1)],axis=0)

    output = np.concatenate([mean.reshape(1,mean.shape[0],mean.shape[1]),sigma.reshape(1,sigma.shape[0],sigma.shape[1])])
        
    return output

def get_clustering_data(universe,wp_start,wp_stop):
    
    #Open Clustering group
    wp = universe['Clustering']
    wpkeys = [key for key in wp.keys()]
    #print(wpkeys)
    
    wps = wp['Sets']
    min_mass = np.array(wps['Minimum_Mass'])
    max_mass = np.array(wps['Maximum_Mass'])
    wp_mass = np.array([min_mass,max_mass])
    
    #Open Data group
    wpdata = wp['Data']
    wpdatakeys = [key for key in wpdata.keys()]
    
    #Check for the maximum number of entries for the data sets
    imax = 0
    for i in wpdatakeys:
        if wpdata[i].shape[0] > imax: imax = wpdata[i].shape[0]

    #Create array to store all data
    data = np.empty([0,imax,3])
    
    #Loop over the datasets
    for i,key in enumerate(wpdatakeys):
        
        #Only return selected sets
        if i >= wp_start and i <= wp_stop:
        
            wpset = wpdata[key]
            x = wpset['Radius']
            y = wpset['Wp_observed']
            s = wpset['Sigma_observed']

            #Pad the first set
            if i == 0:
                x = np.pad(x, (1,1), 'constant', constant_values=(0.0, 0.0))
                y = np.pad(y, (1,1), 'constant', constant_values=(0.0, 0.0))
                s = np.pad(s, (1,1), 'constant', constant_values=(np.inf, np.inf))

            #Pad the second set
            if i == 3:
                x = np.pad(x, (2,0), 'constant', constant_values=(0.0, 0.0))
                y = np.pad(y, (2,0), 'constant', constant_values=(0.0, 0.0))
                s = np.pad(s, (2,0), 'constant', constant_values=(np.inf, np.inf))

            #Pad the third set
            if i == 4:
                x = np.pad(x, (4,0), 'constant', constant_values=(0.0, 0.0))
                y = np.pad(y, (4,0), 'constant', constant_values=(0.0, 0.0))
                s = np.pad(s, (4,0), 'constant', constant_values=(np.inf, np.inf))

            #Combine x,y,s
            dset = np.concatenate([x.reshape(-1,1),y.reshape(-1,1),s.reshape(-1,1)],axis=1)
            #Add data to data array
            data = np.concatenate([data,dset.reshape(1,-1,3)],axis=0)

    return data.transpose(2,0,1), wp_mass.T[wp_start:wp_stop+1]
