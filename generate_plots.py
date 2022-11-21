import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import Tensor, ones, stack, load, nn
import scipy.io
from pathlib import Path
from matplotlib.colors import LogNorm
import os
import imageio
import sys
import time
sys.path.append("fno")




def evaluate(data_og, data,T_in,T_pred,path, ds="test"):
    #%matplotlib inline
    
    # ground-truth values from original data
    gt_og = data_og['density']
    x = data_og['x']*0.8/(2*np.pi)
    px = data_og['px']
    d = data_og['d']
    time_og = data_og['time']
    
    # prediction of models for test data
    pred = data['prediction']
    gt = data['gt']
    pred = np.transpose(pred, (0, 3, 1, 2))
    gt = np.transpose(gt, (0, 3, 1, 2))
    
    # preditction of models for train data
    pred_tr = data['prediction_tr']
    gt_tr = data['gt_tr']
    pred_tr = np.transpose(pred_tr, (0, 3, 1, 2))
    gt_tr = np.transpose(gt_tr, (0, 3, 1, 2))
    
    # test index
    test_index = data['test_index'].reshape(-1)
    # train index
    train_index = data['train_index'].reshape(-1)   
    
    # number of data points for total, train and test data
    ntotal = d.shape[1]
    ntrain= train_index.shape[0]
    ntest = test_index.shape[0]

    # create meshgrid from x, px values
    X, PX = np.meshgrid(x, px)
    # number of predicted times
    T = pred.shape[1]
    
    D = pred.shape[0]

    extent = np.min(X), np.max(X), np.min(PX), np.max(PX)


    d_train = d[0,train_index]

    n_t_in = np.round(time_og[0,:T_in]*2.667/(2*np.pi),2)
    n_t_out = np.round(time_og[0,T_in:T_pred+T_in]*2.667/(2*np.pi),2)
    

    d_test = d[0,test_index]
    
    
    gt_og_train = gt_og[train_index,T_in:T_pred+T_in,:,:]
    gt_og_test = gt_og[test_index,T_in:T_pred+T_in,:,:]
    
    gt_og_train_IC = gt_og[train_index,0,:,:]
    gt_og_test_IC = gt_og[test_index,0,:,:]


    def make_plots(X,PX, N, pred,gt_og,gt_og_IC, d, T, n_t_out,ds):
        """
        X: spatial coordinate
        PX: momentum
        N: ntrain or ntest, number of d values
        pred: prediction
        gt_og: ground-truth
        gt_og: ground-truth initial condtion
        d: d_train or d_test
        T: time
        """
        filenames_all = []
        filenames_l1_all = []
        filenames_rel_l1_all = []
        norm_inf_all = []
        norm_l1_all = []
        norm_rel_l1_all = []
        energy_all = []
        for i in range(N):
            print(i)
            filenames = []
            filenames_l1 = []
            filenames_rel_l1 = []
            norm_inf = []
            norm_l1 = []
            norm_rel_l1 = []
            energy = []
            for t in range(T):
                
                fig = plt.figure(figsize=(10, 20))
                # Adds subplot on position 1
                ax = fig.add_subplot(212)

                # Adds subplot on position 2
                ax2 = fig.add_subplot(211)
                
                ########### model prediction ###############
                cp = ax.imshow(pred[i,t,:,:].T, interpolation='nearest', cmap='YlGnBu', origin = 'lower',   
                    extent = extent,  
                    aspect='auto',
                    norm=LogNorm(vmin=1e-3, vmax=10)
                    )
                ax.set_xlim(15,35)
                cb = fig.colorbar(cp, ax=ax)
                cb.set_label('log f(x, p$_x$) [a.u.]',fontsize=40) # Add a colorbar to a plot
                #ax.tick_params(axis='both', which='major', labelsize=15)
                cb.ax.tick_params(labelsize=30)
                ax.tick_params(labelsize=30)
                ax.set_xlabel('x (microns)',fontsize=40)
                ax.set_ylabel('p$_x$ (m$_e$ c)',fontsize=40)
#                 ax.set_title('Predicted Phase Space at t = {} fs'.format(n_t_out[t]),fontsize=25)
                ############################################
                
                ########### ground truth ###############
                cp2 = ax2.imshow(gt_og[i,t,:,:].T, interpolation='nearest', cmap='YlGnBu', origin = 'lower',   
                    extent = extent,  
                    aspect='auto',
                    norm=LogNorm(vmin=1e-3, vmax=10)
                   )
                cb2 = fig.colorbar(cp2, ax =ax2)
                cb2.set_label('log f(x, p$_x$) [a.u.]',fontsize=40) # Add a colorbar to a plot
                cb2.ax.tick_params(labelsize=30)
                ax2.set_xlim(15,35)
                ax2.tick_params(labelsize=30)
                ax2.set_xlabel('x (microns)',fontsize=40)
                ax2.set_ylabel('p$_x$ (m$_e$ c)',fontsize=40)
#                 ax2.set_title('Ground-truth Phase Space at t = {} fs'.format(n_t_out[t]),fontsize=25)
                if t+1 == 10:
                    ax2.set_title('t$_{10}$ = ' + str(n_t_out[t])+ ' fs',fontsize=40)
                else:
                    ax2.set_title('t$_{}$ = {} fs'.format(t+1,n_t_out[t]),fontsize=40)
                ########################################
                
                ########## L1 plot ####################
                
                #### calculate l1 norm ####
                l1 = abs(pred[i,t,:,:] - gt_og[i,t,:,:])
                
                                
                dir_path = 'fno/fnoresults/'+path+'/'+ds+'/'+str(d[i])
                # Instantiate the Path class
                obj = Path(dir_path)
                
                # Check if path points to an existing file or directory
                if not obj.exists():   
                    os.makedirs(dir_path)
                
                #################### Calculate infinity norm ########################
                #norm_inf_temp = np.linalg.norm(pred[i,t,:,:] - gt_og[i,t,:,:], np.inf)
                norm_inf_temp = np.max(np.abs(pred[i,t,:,:] - gt_og[i,t,:,:]))
                norm_inf.append(norm_inf_temp)
                
                ################### Calculate l1 norm ##############################
                #norm_l1_temp = np.linalg.norm(pred[i,t,:,:] - gt_og[i,t,:,:], 1)
                norm_l1_temp = np.mean(np.abs(pred[i,t,:,:] - gt_og[i,t,:,:]))
                norm_rel_l1_temp = np.mean(np.abs((pred[i,t,:,:] - gt_og[i,t,:,:])/gt_og[i,t,:,:].max()))
                norm_l1.append(norm_l1_temp)
                norm_rel_l1.append(norm_rel_l1_temp)
                
                ################ Calculate energy ###############
                energy_temp = np.sum(np.abs(pred[i,t,:,:]))
                energy.append(energy_temp)
                
                #l1 = abs(pred[i,t,:,:] - gt_og[i,t,:,:])
                
                fig.tight_layout()
                plt.savefig(dir_path+'/Pred_vs_gt_d{}_t_{}.png'.format(d[i],n_t_out[t]))
                filenames.append(dir_path+'/Pred_vs_gt_d{}_t_{}.png'.format(d[i],n_t_out[t]))
                plt.close()
                
#                 ######## Individual L1 plot #######
#                 fig2 = plt.figure(figsize=(12, 6))
#                 cp3 = plt.imshow(l1.T, interpolation='nearest', cmap='jet', origin = 'lower',   
#                     extent = extent,  
#                     aspect='auto',
#                     #vmin =0, vmax=10,
#                     norm=LogNorm(vmin=1e-3, vmax=10)
#                     )
#                 fig2.colorbar(cp3).set_label('L$_1$',fontsize=25) # Add a colorbar to a plot
#                 plt.xlabel('x (microns)',fontsize=25)
#                 plt.ylabel('p$_x$ (m$_e$ c)',fontsize=25)
#                 plt.title('L$_1$ between predicted and ground-truth Phase Space for (d={} microns) at t = {} fs'.format(d[i],n_t_out[t]),fontsize=25)
#                 plt.savefig(dir_path+'/L1_d{}_t_{}.png'.format(d[i],n_t_out[t]))
#                 plt.close()
#                 filenames_l1.append(dir_path+'/L1_d{}_t_{}.png'.format(d[i],n_t_out[t]))
# #                 plt.show()
#                 ######## Individual L1 plot #######
                
                ######## Individual relative L1 plot #######
                rel_l1 = np.abs((pred[i,t,:,:] - gt_og[i,t,:,:])/gt_og[i,t,:,:].max())
#                 fig4 = plt.figure(figsize=(12, 6))
#                 cp4 = plt.imshow(rel_l1.T, interpolation='nearest', cmap='jet', origin = 'lower',   
#                     extent = extent,  
#                     aspect='auto',
#                     #vmin =0, vmax=10,
#                     norm=LogNorm(vmin=1e-3, vmax=10)
#                     )
#                 fig4.colorbar(cp4).set_label('Relative L$_1$',fontsize=25) # Add a colorbar to a plot
#                 plt.xlabel('x (microns)',fontsize=25)
#                 plt.ylabel('p$_x$ (m$_e$ c)',fontsize=25)
#                 plt.title('Relative L$_1$ error between predicted and ground-truth Phase Space for (d={} microns) at t = {} fs'.format(d[i],n_t_out[t]),fontsize=25)
#                 plt.savefig(dir_path+'/rel_L1_d{}_t_{}.png'.format(d[i],n_t_out[t]))
#                 plt.close()
#                 filenames_rel_l1.append(dir_path+'/rel_L1_d{}_t_{}.png'.format(d[i],n_t_out[t]))
                ######## Individual relative L1 plot #######
                
            
            filenames_all.append(filenames)
            filenames_l1_all.append(filenames_l1)
            filenames_rel_l1_all.append(filenames_rel_l1)
            norm_inf_all.append(norm_inf)
            norm_l1_all.append(norm_l1)
            norm_rel_l1_all.append(norm_rel_l1)
            
            # calculate max number of particles for ground truth initial condition
            energy_ic = np.sum(np.abs(gt_og_IC[i]))
            
            energy = [x / energy_ic*100 for x in energy]
            
            energy_all.append(energy)
            
#             # Initial condition
#             fig4 = plt.figure(figsize=(10, 10))
#             cp4 = plt.imshow(gt_og_IC[i].T, interpolation='nearest', cmap='YlGnBu', origin = 'lower',   
#                 extent = extent,  
#                 aspect='auto',
#                 norm=LogNorm(vmin=1e-3, vmax=10
#                 ))
#             cb4 = fig4.colorbar(cp4)
#             cb4.set_label('log f(x, p$_x$) [a.u.]',fontsize=40) # Add a colorbar to a plot
#             cb4.ax.tick_params(labelsize=30)
#             plt.tick_params(labelsize=30)
#             plt.xlabel('x (microns)',fontsize=40)
#             plt.ylabel('p$_x$ (m$_e$ c)',fontsize=40)
#             plt.title('Initial condition at t$_0$ = {} fs'.format(n_t_in[0]),fontsize=40)
#             plt.savefig(dir_path+'/IC_d{}_t_{}.png'.format(d[i],n_t_in[0]))
#             plt.close()
            
#             ######################################### new initial condition            
#             fig5 = plt.figure(figsize=(10, 20))
#             # Adds subplot on position 1
#             ax5 = fig5.add_subplot(212)

#             # Adds subplot on position 2
#             ax6 = fig5.add_subplot(211)
            
#             cp5 = ax6.imshow(gt_og_IC[i].T, interpolation='nearest', cmap='YlGnBu', origin = 'lower',   
#                 extent = extent,  
#                 aspect='auto',
#                 norm=LogNorm(vmin=1e-3, vmax=10)
#                 )
#             ax6.set_xlim(15,35)
#             cb5 = fig5.colorbar(cp5, ax=ax6)
#             cb5.set_label('log f(x, p$_x$) [a.u.]',fontsize=40) # Add a colorbar to a plot
#             #ax.tick_params(axis='both', which='major', labelsize=15)
#             cb5.ax.tick_params(labelsize=30)
#             ax6.tick_params(labelsize=30)
#             ax6.set_xlabel('x (microns)',fontsize=40)
#             ax6.set_ylabel('p$_x$ (m$_e$ c)',fontsize=40)
#             ax6.set_title('Initial condition at t$_0$ = {} fs'.format(n_t_in[0]),fontsize=40)
#             fig5.tight_layout()
#             plt.savefig(dir_path+'/IC_v2_d{}_t_{}.png'.format(d[i],n_t_in[0]))                
#             ######################################### new initial condition
            
            # build gif
            with imageio.get_writer(dir_path + '/movie.gif', mode='I',fps=10) as writer:
                for filename in filenames:
                    image = imageio.imread(filename)
                    writer.append_data(image)
            
#             # build gif
#             with imageio.get_writer(dir_path + '/l1_movie.gif', mode='I',fps=10) as writer:
#                 for filename in filenames_l1:
#                     image = imageio.imread(filename)
#                     writer.append_data(image)
                    
#             # build gif
#             with imageio.get_writer(dir_path + '/rel_l1_movie.gif', mode='I',fps=10) as writer:
#                 for filename in filenames_rel_l1:
#                     image = imageio.imread(filename)
#                     writer.append_data(image)
                    
        return filenames_all, norm_inf_all,norm_l1_all, norm_rel_l1_all, energy_all
    
    def make_l_inf_plot(norm_inf_all,d,n_t_out,div):   
        j=0
        fig = plt.figure(figsize=(30, 12))
        for norm_inf,dd in zip(norm_inf_all,d):
            if (j+1)%div == 0 or j==0:            
                plt.plot(n_t_out, norm_inf, label = dd, linestyle="-", linewidth=5)
                plt.legend(fontsize = 25)
                #plt.title('L$_\infty$ norm for different thickness values d (microns)',fontsize=30)
                plt.xlabel('t (fs)',fontsize=50)
                plt.ylabel('L$_\infty$ norm',fontsize=50)
                plt.tick_params(axis='both',labelsize=30)
            j+=1
        plt.savefig('fno/fnoresults/'+path+'/'+ds+'/'+path+'_l_inf_plot.png')
        plt.show()
        
    def make_l1_plot(norm_l1_all,d,n_t_out,div):
        indices = np.argsort(d)
        d=d[indices]
        norm_l1_all = np.asarray(norm_l1_all)
        norm_l1_all=norm_l1_all[indices]
        j=0
        fig = plt.figure(figsize=(30, 12))
        for norm_l1,dd in zip(norm_l1_all,d):
            if (j+1)%div == 0 or j==0:
                plt.plot(n_t_out, norm_l1, label = dd, linestyle="-", linewidth=5)
                # plt.plot(n_t_out, np.cos(x), label = "curve 2", linestyle=":")
                plt.legend(fontsize = 25)
            #     plt.set_title('damped')
            #     plt.set_xlabel('time (s)')
            #     plt.set_ylabel('amplitude')
                #plt.title('L$_1$ norm for different thickness values d (microns)',fontsize=30)
                plt.xlabel('t (fs)',fontsize=50)
                plt.ylabel('L$_1$ norm',fontsize=50)
                plt.tick_params(axis='both',labelsize=30)
            j+=1
        plt.savefig('fno/fnoresults/'+path+'/'+ds+'/'+path+'_l1_plot.png')
        plt.show()
        
        
    def make_energy_plot(energy_all,d,n_t_out,div):
        indices = np.argsort(d)
        d=d[indices]
        energy_all = np.asarray(energy_all)
        energy_all=energy_all[indices]
        j=0
        fig = plt.figure(figsize=(30, 12))
        for energy,dd in zip(energy_all,d):
            if (j+1)%div == 0 or j==0:            
                plt.plot(n_t_out, energy, label = dd, linestyle="-", linewidth=5)
                # plt.plot(n_t_out, np.cos(x), label = "curve 2", linestyle=":")
                plt.legend(fontsize = 25)
                #plt.title('Total number of ions for different thickness values d (microns)',fontsize=30)
                plt.xlabel('t (fs)',fontsize=50)
                plt.ylabel('Percentage of ions (%)',fontsize=50)
                plt.tick_params(axis='both',labelsize=30)
            j+=1
        plt.savefig('fno/fnoresults/'+path+'/'+ds+'/'+path+'_energy_plot.png')
        plt.show()

    if ds == "train":
        filenames_all, norm_inf_all,norm_l1_all,norm_rel_l1_all,energy_all = make_plots(X,PX,ntrain,pred_tr,gt_og_train,gt_og_train_IC,d_train,T,n_t_out,ds)
        make_l_inf_plot(norm_inf_all,d_train,n_t_out,div=10)
        make_l1_plot(norm_l1_all,d_train,n_t_out,div=10)
        make_energy_plot(energy_all,d_train,n_t_out,div=10)
    elif ds == "test":
        filenames_all,norm_inf_all,norm_l1_all,norm_rel_l1_all,energy_all = make_plots(X,PX,ntest,pred,gt_og_test,gt_og_test_IC,d_test,T,n_t_out,ds)
        make_l_inf_plot(norm_inf_all,d_test,n_t_out,div=4) 
        make_l1_plot(norm_l1_all,d_test,n_t_out,div=4)
        make_energy_plot(energy_all,d_test,n_t_out,div=4)
    
    return filenames_all, norm_inf_all, norm_l1_all, norm_rel_l1_all, energy_all, d_train, d_test, n_t_out
     


def make_colormesh_plot(norm_all_test,norm_all_train,d_test,d_train,n_t_out,path,title):
    
    norm_all_test_arr = np.array(norm_all_test)
    norm_all_train_arr = np.array(norm_all_train)
    
    X_train, Y_train = np.meshgrid(n_t_out, d_train)
    extent_train = np.min(X_train), np.max(X_train), np.min(Y_train), np.max(Y_train)
    
    X_test, Y_test = np.meshgrid(n_t_out, d_test)
    extent_test = np.min(X_test), np.max(X_test), np.min(Y_test), np.max(Y_test)
    
    #fig, ax = plt.subplots(figsize=(12, 6))
    fig= plt.figure(figsize=(10, 10))
    #ax.pcolormesh(x, y, Z)

    # Adds subplot on position 1
    ax = fig.add_subplot(211)

    # Adds subplot on position 2
    ax2 = fig.add_subplot(212)


    #cp = ax.pcolormesh(n_t_out, d_test, norm_all_test_arr, shading='auto')
    cp = ax.imshow(norm_all_test_arr,interpolation='nearest', cmap='jet', origin = 'lower', extent = extent_test,
                    aspect='auto', #vmin = 0, vmax =25
                   )
#     fig.colorbar(cp, ax = ax)
    cbar = fig.colorbar(cp, ax = ax)
    cbar.set_label(title + ' norm', fontsize = 25)
    cbar.ax.tick_params(labelsize=15)
    
    ax.set_xlabel('t (fs)', fontsize = 30)
    ax.set_ylabel('d (microns)', fontsize = 30)
    #ax.set_title('L$_\infty$ norm for different d values (test data)', fontsize = 18)
    #ax.set_title(title + ' norm for different d values (test data)', fontsize = 30)
    ax.tick_params(axis='both',labelsize=18)


#     cp2 = ax2.pcolormesh(n_t_out, d_train, norm_all_train_arr, shading='auto')
    cp2 = ax2.imshow(norm_all_train_arr,interpolation='nearest', cmap='jet', origin = 'lower', extent = extent_train,
                aspect='auto', #vmin = 0, vmax =25
               )
#     fig.colorbar(cp2, ax = ax2)
    cbar2 = fig.colorbar(cp2, ax = ax2)
    cbar2.set_label(title + ' norm', fontsize = 25)
    cbar2.ax.tick_params(labelsize=15)

    ax2.set_xlabel('t (fs)', fontsize = 30)
    ax2.set_ylabel('d (microns)', fontsize = 30)
    #ax2.set_title('L$_\infty$ norm for different d values (train data)', fontsize = 18)
    #ax2.set_title(title + ' norm for different d values (train data)', fontsize = 30)
    ax2.tick_params(axis='both',labelsize=18)
    fig.tight_layout()
    plt.savefig('fno/fnoresults/'+path)
    plt.show()
    

def make_conc_colormesh_plot(norm_all_test,norm_all_train,d_test,d_train,n_t_out,path,mode,title):
    
    norm_all_test_arr = np.array(norm_all_test)
    norm_all_train_arr = np.array(norm_all_train)

    X_train, Y_train = np.meshgrid(n_t_out, d_train)
    extent_train = np.min(X_train), np.max(X_train), np.min(Y_train), np.max(Y_train)

    X_test, Y_test = np.meshgrid(n_t_out, d_test)
    extent_test = np.min(X_test), np.max(X_test), np.min(Y_test), np.max(Y_test)

    norm_all = np.concatenate((norm_all_train_arr,norm_all_test_arr), axis=0)


    d_all = np.concatenate((d_train, d_test), axis=0)

    X_all, Y_all = np.meshgrid(n_t_out, d_all)

    extent_all = np.min(X_all), np.max(X_all), np.min(Y_all), np.max(Y_all)

    fig= plt.figure(figsize=(10, 10))

    cp = plt.imshow(norm_all, interpolation='nearest', cmap='jet', origin = 'lower', extent = extent_all,
                    aspect='auto'
                   )


    cbar = fig.colorbar(cp)
    cbar.set_label(title + ' norm', fontsize = 25)
    cbar.ax.tick_params(labelsize=15)
    
    if mode == 'extrapolation':   
        plt.axhspan(ymin = d_train.max(), ymax = d_test.min(), color='r', linestyle='-')

    plt.xlabel('t (fs)', fontsize = 40)
    plt.ylabel('d (microns)', fontsize = 40)
    #plt.legend(loc = 'upper left')
    #plt.title(title + ' norm for different d values', fontsize = 30)
    plt.tick_params(axis='both',labelsize=14)
    fig.tight_layout()
    plt.savefig('fno/fnoresults/'+path)
    plt.show()

    
    
if __name__ == "__main__":

    path = 'hh49zpnh'
    
    mode = 'interpolation'
    T_in = 1
    T_pred = 10
    
    #load the data for fno model results
    data = scipy.io.loadmat('ns_fourier_ion_400_400_80_ep500_m12_w80_'+path+'.mat')
    
    #Load the original simulation results
    data_og = loadmat("ion_N_100_0.5_10_400_400.mat")
    
    
#     filenames_all_train, norm_inf_all_train, norm_l1_all_train, norm_rel_l1_all_train, energy_all_train, d_train, d_test, n_t_out  = evaluate(data_og, data, T_in, T_pred,path=path, ds="train")
    
    filenames_all_test, norm_inf_all_test, norm_l1_all_test, norm_rel_l1_all_test, energy_all_test, d_train, d_test, n_t_out  = evaluate(data_og, data, T_in, T_pred,path =path, ds="test")
    
    