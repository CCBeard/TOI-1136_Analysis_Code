import os
os.environ["OMP_NUM_THREADS"] = "1"


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import ttvfast
from ttvfast import models
import astropy.modeling
import astropy.units as u
import astropy.constants as ac
from astropy.time import Time
from datetime import datetime

import lmfit
import time
import emcee
import radvel
from scipy.linalg import cho_factor, cho_solve
from scipy import spatial
import scipy
import warnings
from matplotlib.gridspec import GridSpec


#########################
#    All Our Plot Func  #
#########################

def plot_results(para,version, samples=None, nindx=10, overplot=False):
    '''
    A function for plotting TTV O-C plots for the inner 6 planets

    para (numpy array): An array of the N model parameters, in the order as defined in the main code

    version (str): Name to be associated with plot (i.e. initial, final, etc)

    samples (str): A post-mcmc samples object for plotting a variety of chains as well

    nindx (int): Number of random chains to plot, if samples is not none

    overplot (bool): True if you wish to overplot the 6-planet results as well

    '''

    #########################
    #      Plot Stuff.      #
    #########################


    fig = plt.figure(figsize=(12,14))

    gs1 = GridSpec(1,1)
    gs1.update(top=0.99, bottom = 0.67,hspace=0.8,wspace=0.8,left=0.01,right=0.48)

    gs2 = GridSpec(1,1)
    gs2.update(top=0.99, bottom = 0.67,hspace=0.8,wspace=0.8,left=0.52,right=0.99)

    gs3 = GridSpec(1,1)
    gs3.update(top=0.65, bottom = 0.34,hspace=0.8,wspace=0.8,left=0.01,right=0.48)

    gs4 = GridSpec(1,1)
    gs4.update(top=0.65, bottom = 0.34,hspace=0.8,wspace=0.8,left=0.52,right=0.99)

    gs5 = GridSpec(1,1)
    gs5.update(top=0.32, bottom = 0.01,hspace=0.8,wspace=0.8,left=0.01,right=0.48)

    gs6 = GridSpec(1,1)
    gs6.update(top=0.32, bottom = 0.01,hspace=0.8,wspace=0.8,left=0.52,right=0.99)



    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs2[0])
    ax3 = plt.subplot(gs3[0])
    ax4 = plt.subplot(gs4[0])
    ax5 = plt.subplot(gs5[0])
    ax6 = plt.subplot(gs6[0])

    ax_tot = [ax1,ax2,ax3,ax4,ax5,ax6]
    ylimits = [[-100, 120],[-30,90],[-60,400],[-1200,200],[-100,500],[-300,30]]
    let_pos = [(2000,80),(800,70),(1750,320),(2200,0),(1750,450),(2250,-20)]


    ttvfast_results = para_to_model(para) #N body integration
    which_planet = np.array(ttvfast_results['positions'][0],'i') #which planet is it?
    which_epoch = np.array(ttvfast_results['positions'][1],'i') #epoch
    transit_times = np.array(ttvfast_results['positions'][2],'d') #transit times


    for i in range(6): #go through only systems with TTV measurements
        ylabel = None
        xlabel = None

        if i >3:
            xlabel='BJD - 2458000'

        if i % 2 == 0:
            ylabel= 'TTV (min)'


        match = np.where((which_planet==i) & (transit_times>-2.)) #TTVfast returns -2 if there is no transit
        match_plot = np.where((which_planet==i))
        epoch_tmp = which_epoch[match]
        transit_time_tmp = transit_times[match]

        epoch_plot = which_epoch[match_plot]
        transit_time_plot = transit_times[match_plot]
        thing = np.argsort(epoch_plot)



        ax = ax_tot[i]

        #these are the datapoints (transits)
        ax.errorbar(epoch_obs[i]*p_ave[i] + tc_ave[i], #make the x axis bjd
                    (transit_time_obs[i]-epoch_obs[i]*p_ave[i]-tc_ave[i])*(24*60), #convert to minutes
                    yerr=o_c_unc[i]*(24*60),
                     fmt = 'o', color ='black', label = 'Observed TTVs', mec='dodgerblue', mew=0.1)
        #model predictions
        ax.errorbar(epoch_plot[thing]*p_ave[i] + tc_ave[i],
                (transit_time_tmp[epoch_plot]-epoch_plot*p_ave[i]-tc_ave[i])[thing]*(24*60),
                yerr=None, marker='d', ms=3,
                color ='red', label = '7p Model', zorder=-10)

        letters = ['b','c','d', 'e','f','g']


        #overplot 6p model
        if overplot:
            mod_6p = pd.read_csv('RVPlots/TTV_fit_6p_{}.txt'.format(i), sep=' ', header=None)
            times_6p = mod_6p[0].to_numpy(dtype=float)
            ttv_6p = mod_6p[1].to_numpy(dtype=float)
            #plot the 6p model
            ax.errorbar(times_6p,
                    ttv_6p,
                    yerr=None, marker='d', ms=3,
                    color ='green', label = '6p Model', zorder=-10, alpha=0.5)

        #plot TESS Sector 75
        ax.axvspan(2460339.50000 - 2458000, 2460339.50000+27 - 2458000, color='gold', alpha=0.3, label='TESS Sector 75')

        ax.annotate(letters[i], xy=(let_pos[i]), color='red', fontsize=30)


        if samples is not None:
            #plot MCMC chains

            for j in range(nindx):

               label=None
               if j == 0:
                   label='Chain Predictions'
               para_o = samples[j,:]


               try:

                    ttvfast_results_o = para_to_model(para_o)
                    which_planet_o = np.array(ttvfast_results_o['positions'][0],'i')#which system
                    which_epoch_o = np.array(ttvfast_results_o['positions'][1],'i')#epoch
                    transit_times_o = np.array(ttvfast_results_o['positions'][2],'d')#transit times

                    match_o = np.where((which_planet_o==i) & (transit_times_o>-2.))
                    match_plot_o = np.where((which_planet_o==i))
                    epoch_tmp_o = which_epoch_o[match_o]
                    transit_time_tmp_o = transit_times_o[match_o]

                    epoch_plot_o = which_epoch_o[match_plot_o]
                    transit_time_plot_o = transit_times_o[match_plot_o]
                    thing_o = np.argsort(epoch_plot_o)


                    ax.errorbar(epoch_plot_o[thing_o]*p_ave[i] + tc_ave[i],
                            (transit_time_tmp_o[epoch_plot_o]-epoch_plot_o*p_ave[i]-tc_ave[i])[thing_o]*(24*60),
                            yerr=None, fmt='-', ms=3, label=label,
                            color ='dodgerblue', zorder=-100, alpha=0.1)

               except IndexError:
                   continue

        ax.set_ylim(ylimits[i])
        ax.legend(fontsize=14)
        ax.set_xlabel(xlabel,fontsize=24)
        ax.set_ylabel(ylabel, fontsize=24)

    plt.savefig('7pPlots/O-C_{}.png'.format(version),bbox_inches='tight')
    plt.close()


#massive function to plot RV results :/
def plot_rv_results(para, t, version, error=None):
    '''
    Function for plotting RV plots in the paper

    para (numpy float): List of N model values for which to plot

    t (numpy float): times to predict RV values at for purposes of plotting

    version (str): Name for the plots (i.e. initial, final, etc)

    error (numpy float): Uncertainties in RV parameters to plot 1 sigma upper/lower limits


    '''

    inst_names = {'hires_j':'HIRES', 'apf':'APF', 'harpsn':'HARPS-N'}
    cs = {'hires_j':'dodgerblue', 'apf':'green', 'harpsn':'indianred'}
    alphas={'hires_j':1., 'apf':0.5, 'harpsn':1.}

    ttvfast_results = para_to_model(para, rvs=rvtimes) #calculate RV values at the RV datapoint timesteps
    model_rv = np.array(ttvfast_results['rv']) * (au2m)/(60*60*24) #change from AU/day to m/s

    ttvfast_results = para_to_model(para, rvs=t) #calculate RV values at datapoints we want for plotting purposes
    predicted_rv = np.array(ttvfast_results['rv'])*(au2m)/(60*60*24) #change from AU/day to m/s
    t = np.array(t)

    #for now, while gammas are not folded in
    gammas = np.zeros(len(rvtimes)) #create an array of gammas
    jitters = np.ones(len(rvtimes)) #create an array of jitters

    gam = {}
    jit = {}

    for i in range(len(telnames)):
        this_gamma = para[-(6 + 2*len(telnames) - 2*i)]
        this_jitter = para[-(6 + 2*len(telnames) - 2*i - 1)]
        gam[telnames[i]] = this_gamma
        jit[telnames[i]] = this_jitter

    for j in range(len(rvtimes)):
        gammas[j] += gam[tel[j]]
        jitters[j] += jit[tel[j]]




    ###########################
    #.  Generate RV Model.    #
    ###########################

    #get the necessary parameters for para
    periods = [] #orbital period
    periods_up = [] #period 1 sigma upper limit
    periods_down = [] #period 1 sigma lower
    tc = [] #time of inferior conjuction
    tc_up = [] #tc 1 sigma upper limit
    tc_down = [] #tc 1 sigma lower limit
    e = [] #orbital eccentricity
    e_up = [] #eccentricity upper limit
    e_down = [] #eccentricity lower limit
    w = [] #mean anomaly
    w_up = []
    w_down = []
    m_p = [] #planet mass (in m earth)
    m_p_up = []
    m_p_down = []
    inc = [] #orbital inclination (in degrees)
    inc_up = []
    inc_down = []
    for i in range(n_p):
        m_p.append(para[0+i*6])
        periods.append(para[1+i*6])
        e.append(para[2+i*6]**2+para[3+i*6]**2)
        w.append(np.arctan2(para[2+i*6],para[3+i*6])) #in radians for radvel
        inc.append(para[4+i*6])
        if error is not None:
            m_p_up.append(para[0+i*6] + error[0+i*6])
            periods_up.append(para[1+i*6] + error[0+i*6])
            e_up.append((para[2+i*6]+error[2+i*6])**2+(para[3+i*6]+error[3+i*6])**2)
            w_up.append(np.arctan2(para[2+i*6]+error[2+i*6],para[3+i*6]+error[3+i*6])) #in radians for radvel
            inc_up.append(para[4+i*6] + error[4+i*6])
            m_p_down.append(para[0+i*6] - error[0+i*6])
            periods_down.append(para[1+i*6] - error[0+i*6])
            e_down.append((para[2+i*6]-error[2+i*6])**2+(para[3+i*6]-error[3+i*6])**2)
            w_down.append(np.arctan2(para[2+i*6]-error[2+i*6],para[3+i*6]-error[3+i*6])) #in radians for radvel
            inc_down.append(para[4+i*6]-error[4+i*6])
    m_p = np.array(m_p) * (stellar_mass * Msun2Mearth)
    if error is not None:
        m_p_up = np.array(m_p_up) * (stellar_mass * Msun2Mearth)
        m_p_down = np.array(m_p_down) * (stellar_mass * Msun2Mearth)
    #convert m_p into K amplitude!
    Kamps = []
    Kamps_up = []
    Kamps_down = []
    for i in range(n_p):
        Kamps.append(Kamp(m_p[i], periods[i], stellar_mass, inc[i]))
        if error is not None:
            Kamps_up.append(Kamp(m_p_up[i], periods_up[i], stellar_mass, inc_up[i]))
            Kamps_down.append(Kamp(m_p_down[i], periods_down[i], stellar_mass, inc_down[i]))

    #now get times of periastron
    #RadVel wants periastron times, not times of inferior conjunction
    #luckily we can convert conveniently
    tp = []
    tp_up = []
    tp_down = []
    for i in range(n_p):
        tp.append(radvel.orbit.timetrans_to_timeperi(tc_ave[i], periods[i], e[i], w[i])) #radvel conversion
        if error is not None:
            tp_up.append(radvel.orbit.timetrans_to_timeperi(tc_ave[i], periods_up[i], e_up[i], w_up[i]))
            tp_down.append(radvel.orbit.timetrans_to_timeperi(tc_ave[i], periods_down[i], e_down[i], w_down[i]))


    #now we append to orbel lists for use in radvel RV drive
    orbels = []
    orbels_up = []
    orbels_down = []
    for i in range(n_p):
        orbels.append([periods[i], tp[i], e[i], w[i], Kamps[i]]) #orbital elements needed to compute an RV orbit in RadVel
        if error is not None:
            orbels_up.append([periods_up[i], tp_up[i], e_up[i], w_up[i], Kamps_up[i]])
            orbels_down.append([periods_down[i], tp_down[i], e_down[i], w_down[i], Kamps_down[i]])


    RVmodelp = [] #the RV values at each measured RV timestep, predicted
    RVmodelp_up = []
    RVmodelp_down = []
    RVmodel = []
    for i in range(n_p):
        RVmodelp.append(radvel.kepler.rv_drive(t, orbels[i]))
        if error is not None:
            RVmodelp_up.append(radvel.kepler.rv_drive(t, orbels_up[i]))
            RVmodelp_down.append(radvel.kepler.rv_drive(t, orbels_down[i]))

        RVmodel.append(radvel.kepler.rv_drive(x_rv, orbels[i]))

    RVmodelp = np.array(RVmodelp)
    if error is not None:
        RVmodelp_up = np.array(RVmodelp_up)
        RVmodelp_down = np.array(RVmodelp_down)




    #############################
    #    Generate GP Model.     #
    #############################

    #instrument gp amplitudes
    amps = {'hires_j':para[-6], 'apf': para[-5], 'harpsn':para[-4]} #gp amplitudes

    #list the gp hyperparameters
    perlength = para[-1]
    explength = para[-3]
    per = para[-2]

    #generate the covmatrix for each instrument
    #this covariance matrix is the Chromatic KJ1 kernel from Cale et al. 2021
    def compute_covmatrix(amp_i, amp_j, per, perlength, explength, dist_se, dist_p, errors):

            K = np.array(np.outer(amp_i, amp_j)
                         * np.exp(-dist_se/(2*explength**2))
                         * np.exp((-np.sin(np.pi*dist_p/per)**2.) / (2*perlength**2)))

            # add errors along the diagonal
            try:
                K += (errors**2) * np.identity(K.shape[0])
            except ValueError:  # errors can't be added along diagonal to a non-square array
                pass

            return K


    mu = {} #saving gp predictions for subtracting from RVs
    mu_pred = {} #saving gp predictions for plotting
    mu_tot = []
    x_temp = []

    r = (y_rv - model_rv - gammas) #model residuals without GP

    X1 = np.array([x_rv]).T
    X2 = np.array([x_rv]).T
    dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
    dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

    errorbars = np.sqrt(jitters**2 + yerr_rv**2)

    #make amp matrices
    amp_i = []
    for i in range(len(tel)):
        amp_i.append(amps[tel[i]])
    amp_j = amp_i #for this part, we use the same


    K = compute_covmatrix(amp_i, amp_j, per, perlength, explength, dist_se, dist_p, errorbars)

    Ks = compute_covmatrix(amp_i, amp_j, per, perlength, explength, dist_se, dist_p, 0.)

    L = cho_factor(K)
    alpha = cho_solve(L, r)
    mu = np.dot(Ks, alpha).flatten()

    #do this again for the predictions

    for inst in inst_names.keys():

        r = (y_rv - model_rv - gammas)

        X1 = np.array([x_rv]).T
        X2 = np.array([x_rv]).T
        dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

        errorbars = np.sqrt(jitters**2 + yerr_rv**2)

        #make amp matrices
        amp_i = []
        for i in range(len(tel)):
            amp_i.append(amps[tel[i]])
        amp_j = amp_i #for this part, we use the same

        K = compute_covmatrix(amp_i, amp_j, per, perlength, explength, dist_se, dist_p, errorbars)

        #this time we need to evaluate dist_se and dist_p at predicted times

        X1 = np.array([t]).T # the prediction times
        X2 = np.array([x_rv]).T
        dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

        amp_i = np.repeat(amps[inst], len(t)) #this is for predictions


        Ks = compute_covmatrix(amp_i, amp_j, per, perlength, explength, dist_se, dist_p, 0.)

        L = cho_factor(K)
        alpha = cho_solve(L, r)
        mu_pred[inst] = np.dot(Ks, alpha).flatten()



    #########################
    #      Plot Stuff.      #
    #########################



    fig = plt.figure(figsize=(14,24))

    gs1 = GridSpec(1,1)
    gs1.update(top=0.99, bottom = 0.82,hspace=0.8,wspace=0.8,left=0.01,right=0.99)

    gs2 = GridSpec(1,1)
    gs2.update(top=0.80, bottom = 0.74,hspace=0.8,wspace=0.8,left=0.01,right=0.99)

    gs3 = GridSpec(1,1)
    gs3.update(top=0.7, bottom = 0.54,hspace=0.8,wspace=0.8,left=0.01,right=0.49)

    gs4 = GridSpec(1,1)
    gs4.update(top=0.7, bottom = 0.54,hspace=0.8,wspace=0.8,left=0.51,right=0.99)

    gs5 = GridSpec(1,1)
    gs5.update(top=0.525, bottom = 0.365,hspace=0.8,wspace=0.8,left=0.01,right=0.49)

    gs6 = GridSpec(1,1)
    gs6.update(top=0.525, bottom = 0.365,hspace=0.8,wspace=0.8,left=0.51,right=0.99)

    gs7 = GridSpec(1,1)
    gs7.update(top=0.35, bottom = 0.19,hspace=0.8,wspace=0.8,left=0.01,right=0.49)

    gs8 = GridSpec(1,1)
    gs8.update(top=0.35, bottom = 0.19,hspace=0.8,wspace=0.8,left=0.51,right=0.99)

    gs9 = GridSpec(1,1)
    gs9.update(top=0.175, bottom = 0.01,hspace=0.8,wspace=0.8,left=0.01,right=0.49)



    ax1 = plt.subplot(gs1[0])
    ax2 = plt.subplot(gs2[0])
    ax3 = plt.subplot(gs3[0])
    ax4 = plt.subplot(gs4[0])
    ax5 = plt.subplot(gs5[0])
    ax6 = plt.subplot(gs6[0])
    ax7 = plt.subplot(gs7[0])
    ax8 = plt.subplot(gs8[0])
    ax9 = plt.subplot(gs9[0])


    #The RV scatterplot

    for inst in inst_names.keys():
        mask = tel == inst
        markers, caps, bars = ax1.errorbar(x_rv[mask], y_rv[mask] - gammas[mask],
                     yerr=np.sqrt(yerr_rv**2)[mask],
                     fmt="o", color=cs[inst], mew=1.,mec='black',linewidth=0.5,ms=8,label=inst_names[inst],
                                          )

    #GP Part
    for inst in inst_names.keys():
        ax1.plot(t, mu_pred[inst] + predicted_rv, color=cs[inst], alpha=0.5, lw=2)

    ax1.set_ylabel(r"RV Amplitude (m s$^{-1}$)",fontsize=20)
    ax1.axhline(0, color='gray',ls='--', zorder=-5)
    ax1.legend(loc='upper left', fontsize=14).set_zorder(2000)

    ax1.set_xticklabels(labels=[],fontsize=8)



    #######################
    ####    Residuals   ####
    #######################


    residuals = y_rv - model_rv - gammas

    for inst in inst_names.keys():
        mask = tel == inst
        markers, caps, bars = ax2.errorbar(x_rv[mask], residuals[mask] - mu[mask],
                     yerr=np.sqrt(yerr_rv**2)[mask],
                     fmt="o", color=cs[inst], mew=1.,mec='black',linewidth=0.5,ms=8,
                     label='RMS = '+str(np.round(np.std(residuals[mask] - mu[mask]),2))+ r' m s$^{-1}$',
                                          )



    ax2.axhline(0, color='gray',ls='--',zorder=-5)



    ax2.set_xlabel("BJD - 2450000 (days)",fontsize=20)
    ax2.set_ylabel(r"Residuals (m s$^{-1}$)",fontsize=16)



    # Folding By Planet

    #######################
    ####    Planets    ####
    #######################

    t0s = tc_ave
    axes = [ax3, ax4, ax5, ax6, ax7, ax8, ax9]
    letters = ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    ks = K

    for n in range(7):

        ax = axes[n]

        period = periods[n]
        t0 = t0s[n]
        other = np.zeros(len(x_rv))
        for i in range(6):
            if i == n:
                continue
            else:
                other += RVmodel[n]
        detrended = y_rv - gammas - other
        folded = ((x_rv - t0 + 0.5 * period) % period) / period

        for inst in inst_names:
            mask = tel == inst
            markers, caps, bars = ax.errorbar(folded[mask], (y_rv - gammas - model_rv + RVmodel[n] - mu)[mask],
                         yerr=np.sqrt(yerr_rv**2)[mask],
                         fmt="o", color=cs[inst], mew=1.,mec='black',linewidth=1,ms=8, alpha=alphas[inst],
                                              )


        ax.axhline(0, color='gray',ls='--',zorder=-5)
        t_folded = ((t - t0 + 0.5 * period) % period) / period
        inds = np.argsort(t_folded)


        ax.plot(t_folded[inds], RVmodelp[n][inds], '-k', label="Planet {}".format(letters[n]))
        if error is not None:

            plot_up = []
            plot_down = []
            for p in range(len(t_folded[inds])):
                if t_folded[inds][p] < 0.5:
                    plot_up.append(RVmodelp_up[n][inds][p])
                    plot_down.append(np.max([RVmodelp_down[n][inds][p], 0]))
                elif t_folded[inds][p] >= 0.5:
                    plot_up.append(RVmodelp_up[n][inds][p])
                    plot_down.append(np.min([RVmodelp_down[n][inds][p], 0]))
            ax.fill_between(t_folded[inds], plot_up, plot_down, color='gray', alpha=0.5, label=r'1$\sigma$ confidence')





        bins = np.linspace(0, 1, 10)
        denom, _ = np.histogram(folded, bins)
        num, _ = np.histogram(folded, bins, weights=(y_rv - gammas - model_rv - mu + RVmodel[n]))
        denom[num == 0] = 1.0
        ax.plot(
            0.5 * (bins[1:] + bins[:-1]), num / denom, "o",
            markersize=15, color="coral", label="Binned", mew=3, mec='white', zorder=3000
        )

        ax.set_xlim(0, 1)
        if n > 4:
            ax.set_xlabel("Orbital Phase",fontsize=20)
        if n % 2 == 0:
            ax.set_ylabel(r"RV Amplitude (m s$^{-1}$)",fontsize=20)
        if n % 2 == 1:
            ax.axes.yaxis.set_ticklabels([None, None, None,None,None,None])
        ax.legend(fontsize=12,loc='upper right').set_zorder(2000)
        ax.set_ylim(-15,15)

    plt.savefig('7pPlots/RV_Fit_{}.png'.format(version),bbox_inches='tight')
    plt.close()



###########################
#   Utility Functions     #
###########################

def determine_period(epoch_fit,transit_time_fit,unc_fit):
        fit = astropy.modeling.fitting.LinearLSQFitter()

        # initialize a linear model
        line_init = astropy.modeling.models.Linear1D()

        # fit the data with the fitter
        fitted_line = fit(line_init, epoch_fit, transit_time_fit, weights=1.0/unc_fit)
        #plt.errorbar(epoch_fit, transit_time_fit,yerr= unc_fit,fmt= '.')
        #print(fitted_line)
        return fitted_line.slope.value, fitted_line.intercept.value



def Kamp(M_p,P,M_s, inc):
    '''
    Calculates K-amplitude from planet mass for RV plotting

    M_p (float): planet mass (M earth)

    P (float): Planet orbital period (days)

    M_s (float): Stellar mass (M sun)

    inc (float): inclination (degrees)

    '''
    M_p = np.array(M_p) * np.sin(inc * np.pi/180.) #make it msini, since we're fitting inclination too
    P = np.array(P)
    M_s = np.array(M_s)

    K = (((2*np.pi*ac.G.value)/(float(P)*86400))**(1/3)) * ((float(M_p)*Mearth2Mkg)/((float(M_s)*Msun2Mkg)**(2/3)))
    return K


def para_to_model(para, rvs=None, include_h=False):
    '''
    Take in an array of parameters and organize it, ingest it into TTVFast, and return results

    para (numpy float): Model free Parameters

    rvs (numpy float): RV times for TTVFast to comput predictions

    include_h (bool): Use the seventh planet in the model computation?

    '''

    if include_h == False:
        planet_n = 6
    else:
        planet_n = 7


    for i in range(planet_n):
        planets[i].mass = para[0+i*6]
        planets[i].period = para[1+i*6]
        planets[i].eccentricity = para[2+i*6]**2+para[3+i*6]**2 #convert from secosw and sesinw to e
        if para[2+i*6]**2+para[3+i*6]**2>1.: planets[i].eccentricity = 0.99 #don't let eccentricity be bigger than 1
        planets[i].argument = np.arctan2(para[2+i*6],para[3+i*6])*180./np.pi #convert from secosw and sesinw
        planets[i].inclination = para[4+i*6]
        planets[i].longnode = 0 #We have this fixed at 0 for our model
        planets[i].mean_anomaly = para[5+i*6]

    if rvs is not None:
        ttvfast_results = ttvfast.ttvfast(planets, stellar_mass, T_st, dt, T_fi, rv_times=rvs)
    else:
        ttvfast_results = ttvfast.ttvfast(planets, stellar_mass, T_st, dt, T_fi, rv_times=rvtimes) #this is just the times when our RVs were taken
    return ttvfast_results


def apply_prior(parameter_val, prior_type, a, b):
    '''
    Function that adds a prior to the likelihood

    parameter_val (float): current value of the free parameter

    prior_type (str): Either uniform or gaussian. Only these priors are supported right now, but one could add more easily

    a (float): first prior parameter, minimum for uniform, mean for gaussians

    b (float): second prior parameter, maximum for uniform, sd for gaussian

    '''

    if prior_type == 'uniform':
        #first check that the parameter is inside the region
        if parameter_val < a or parameter_val > b:
            out = -np.inf
        else:
            out = np.log(1/(b - a)) #normalize

    if prior_type == 'gaussian':
        out = -0.5 * ((parameter_val - a) / b)**2 - 0.5*np.log((b**2)*2.*np.pi)

    return out



def residual(paras):
    '''
    Calculate a model "residual" for use with lmfit
    Kind of like a chi2, but it returns a list of values at each datapoint

    paras (lmfit paras): An lmfit object, as defined in the main code


    '''


    ######################
    #.   TTV Portion.    #
    ######################

    #convert the lmfit object to a list
    tmp = paras.valuesdict()
    para = []
    for i in range(len(tmp)):
        para.append(tmp[para_name[i]])

    ttvfast_results = para_to_model(para, include_h=True)
    which_planet = np.array(ttvfast_results['positions'][0],'i')#which system
    which_epoch = np.array(ttvfast_results['positions'][1],'i')#epoch
    transit_times = np.array(ttvfast_results['positions'][2],'d')#transit times

    resi = []
    for i in ttv_fit: #This goes through all 7 planets
        match = np.where((which_planet==i) & (transit_times>-2.))
        epoch_tmp = which_epoch[match]
        transit_time_tmp = transit_times[match]

        try:
            resi = np.concatenate((resi, ((transit_time_obs[i]-transit_time_tmp[epoch_obs[i]])/o_c_unc[i])**2 + np.log(2*np.pi*o_c_unc[i]**2)), axis=0)
        except IndexError:
            print('Index Error Occured')
            resi = np.concatenate([resi, np.repeat(100000,len(transit_time_obs[i]))], axis=0)



    ######################
    #.    RV Portion.    #
    ######################

    ttvfast_results_h = para_to_model(para, include_h=True)
    model_rvs = np.array(ttvfast_results_h['rv']) * (au2m)/(60*60*24) #change from AU/day to m/s


    gam = {}
    jit = {}

    #create gamma array and jitter array
    for i in range(len(telnames)):
        this_gamma = para[-(0 + 2*len(telnames) - 2*i)]
        this_jitter = para[-(0 + 2*len(telnames) - 2*i - 1)]

       	gam[telnames[i]] = this_gamma
        jit[telnames[i]] = this_jitter

    gammas = []
    jitters = []


    for j in range(len(rvtimes)):
        gammas.append(gam[tel[j]])
        jitters.append(jit[tel[j]])

    gammas = np.array(gammas)
    jitters = np.array(jitters)

    amps = {'hires_j':gp_amp_input, 'apf': gp_amp_input, 'harpsn':gp_amp_input}



    r = y_rv - model_rvs - gammas


    #get the distances
    def compute_distances(x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

        return dist_p, dist_se

    dist_p, dist_se = compute_distances(x_rv, x_rv)

    #make amp matrices


    amp_i = []
    for i in range(len(tel)):
        amp_i.append(amps[tel[i]])
    amp_j = amp_i #for this part, we use the same

    explength = gp_explength_input
    per = gp_per_input
    perlength = gp_perlength_input

    K = np.array(np.outer(amp_i, amp_j)
                 * scipy.exp(-dist_se/(explength**2))
                 * scipy.exp((-np.sin(np.pi*dist_p/per)**2.) / (2.*perlength**2)))


    K += (yerr_rv**2 + jitters**2) * np.identity(K.shape[0])

    dist_p, dist_se = compute_distances(x_rv, x_rv) #now get predictions
    Ks = np.array(np.outer(amp_i, amp_j)
                 * scipy.exp(-dist_se/(explength**2))
                 * scipy.exp((-np.sin(np.pi*dist_p/per)**2.) / (2.*perlength**2))) #don't add errors this time


    L = cho_factor(K)
    alpha = cho_solve(L, r)
    mu = np.dot(Ks, alpha).flatten()


    resi = np.concatenate((resi, np.abs((r - mu)/(yerr_rv))), axis=0)


    return resi

#the log likelihood
def lnprob(para):
    '''
    Log likelihood of the model evaluated at the parameter valuesdict

    para (numpy float): parameter values to calculate the log likelihood at

    '''

    ######################
    #.   TTV Portion.    #
    ######################

    ttvfast_results = para_to_model(para,include_h=True)
    which_planet = np.array(ttvfast_results['positions'][0],'i')#which system
    which_epoch = np.array(ttvfast_results['positions'][1],'i')#epoch
    transit_times = np.array(ttvfast_results['positions'][2],'d')#transit times


    resi = []
    for i in ttv_fit: #go through only systems with TTV measurements
        match = np.where((which_planet==i) & (transit_times>-2.))
        epoch_tmp = which_epoch[match]
        transit_time_tmp = transit_times[match]

        try:
            resi = np.concatenate((resi, ((transit_time_obs[i]-transit_time_tmp[epoch_obs[i]])/o_c_unc[i])**2 + np.log(2*np.pi*o_c_unc[i]**2)), axis=0)
        except IndexError:
            #sometimes, TTVFast doesn't generate the right number of transits, and there is an IndexError
            #I believe this may be a bug in TTVFast, as this occurence can be caused merely by changing the integration timestep to a smaller value
            #It happens rarely, and it is hard to predict what combination of parameters causes the result
            #Fortunately, an MCMC with many chains has no issue simply rejecting any move that would cause this result
            print('Index Error Occured')
            return -np.inf
    chi2 = np.sum(resi)

    logprob = -0.5*chi2


    ######################
    #.    RV Portion.    #
    ######################


    model_rvs = np.array(ttvfast_results['rv']) * (au2m)/(60*60*24) #change from AU/day



    gam = {}
    jit = {}

    #create gamma array and jitter array
    for i in range(len(telnames)):
        this_gamma = para[-(6 + 2*len(telnames) - 2*i)]
        this_jitter = para[-(6 + 2*len(telnames) - 2*i - 1)]

        gam[telnames[i]] = this_gamma
        jit[telnames[i]] = this_jitter

    #print(gam)

    gammas = []
    jitters = []
    for j in range(len(rvtimes)):
        gammas.append(gam[tel[j]])
        jitters.append(jit[tel[j]])

    gammas = np.array(gammas)
    jitters = np.array(jitters)

    amps = {'hires_j':para[-6], 'apf': para[-5], 'harpsn':para[-4]}



    r = y_rv - model_rvs - gammas


    #get the distances
    def compute_distances(x1, x2):
        X1 = np.array([x1]).T
        X2 = np.array([x2]).T
        dist_p = scipy.spatial.distance.cdist(X1, X2, 'euclidean')
        dist_se = scipy.spatial.distance.cdist(X1, X2, 'sqeuclidean')

        return dist_p, dist_se

    dist_p, dist_se = compute_distances(x_rv, x_rv)

    #compute the covariance matrix

    #make amp matrices
    amp_i = []
    for i in range(len(tel)):
        amp_i.append(amps[tel[i]])
    amp_j = amp_i #for this part, we use the same

    explength = para[-3]
    per = para[-2]
    perlength = para[-1]

    K = np.array(np.outer(amp_i, amp_j)
                 * scipy.exp(-dist_se/(explength**2))
                 * scipy.exp((-np.sin(np.pi*dist_p/per)**2.) / (2.*perlength**2)))


    K += (yerr_rv**2 + jitters**2) * np.identity(K.shape[0])


    try:

        alpha = cho_solve(cho_factor(K),r)



            # compute determinant of K
        (s,d) = np.linalg.slogdet(K)
        logprob += -.5 * (np.dot(r, alpha) + d + N*np.log(2.*np.pi))


    except (np.linalg.linalg.LinAlgError, ValueError):
        warnings.warn("Non-positive definite kernel detected.", RuntimeWarning)
        logprob += -np.inf

    #now apply priors
    for p in range(len(para)):
        logprob += apply_prior(para[p], mcmc_prior[p], prior_a[p], prior_b[p])

    return logprob

###########################
#.      Constants.        #
###########################

gravity = 0.000295994511 # AU^3/day^2/M_sun fixed
Mearth2Msun = (ac.M_earth/ac.M_sun).value
Msun2Mkg = ac.M_sun.value
Msun2Mearth = 1/(Mearth2Msun)
Mearth2Mkg = ac.M_earth.value
au2m = (ac.au.value)
Rsun2Rearth = (ac.R_sun/ac.R_earth).value
RsunPerDay = ((ac.R_sun/u.d).to(u.m/u.s)).value
Rsun2AU = (ac.R_sun).to(u.au).value
Mjup2Mearth = (ac.M_jup/ac.M_earth).value
deg2rad = (np.pi * 2) / 360
Lsun2W = 3.827e26
AU2m = 1.496e11
Insol_earth = 1360

###########################
#  Param Starting Values  #
###########################

desig = 'TOI-1136'
h_period = 160. #starting value for planet h orbital period
nsteps = 0 #how many mcmc steps to do
multi_cpu = False


global stellar_mass,T_st, dt, T_fi, n_p, ttv_fit, i_prior, i_unc_prior

stellar_mass = 1.022 # M_sun, taken from D23
t_ref = 680. #all Keplerian elements are calculated here at this epoch, also where integration starts
#data points are already BJD -2458000, and so is this value ^
n_p = 7 #model 7 planets
ttv_fit = [0,1,2,3,4,5,6]

period_input = np.array([4.17278,6.25725,12.51937,18.7992,26.3162,39.5387,h_period],'d') #taken from D23
e_input = np.array([0.031,0.117,0.016,0.057,0.012,0.036, 0.01],'d')#All taken from D23, except e_h
i_input = np.array([86.44,89.42,89.41,89.31,89.38,89.65,90.0],'d')
longnode_input = np.array([0., 0., 0., 0., 0., 0.,0.],'d') #longitude of ascending node, the inner should be fixed to 0 if varying



argument_input = np.array([45, -113, 118, -66, 140, -87, 0.],'d') #argument of pericenter in deg between -180 and 180
tc_input = np.array([684.2689, 688.7211, 686.0671, 697.7758, 699.3854, 711.9393, 1435.10],'d') # time of conjunction for each planet
mass_input = np.array([3.01, 6.0, 8.0, 5.4, 8.3, 4.8, 25.],'d')/stellar_mass*Mearth2Msun #in jupiter mass divided by stellar mass
#calculate the mean anomaly at time of reference
f_tran = 0.5*np.pi - argument_input/180.*np.pi
EE_tran = 2.0*np.arctan(((1-e_input)/(1+e_input))**0.5*np.tan(f_tran/2.0))
M_tran = EE_tran-e_input*np.sin(EE_tran)
t_peri = tc_input-M_tran/2.0/np.pi*period_input
M_ref_input = (t_ref-t_peri)*2*np.pi/period_input*180./np.pi #again deg
M_ref_input =M_ref_input % 360.
M_ref_input[5] -= 360 #make it between -180 and 180, for some reason this one came out outside the range


######################
#    RV Inst.        #
######################

gammas = {'hires_j': 0., 'apf': 0., 'harpsn': 0.} #RV linear offsets
gamma_max = 1000. #prior max
gamma_min = -1000. #prior min

jitters = {'hires_j': 10., 'apf': 10., 'harpsn': 10.} #RV jitters
jitter_max = 50. #prior max
jitter_min = 0.1 #prior min


######################
#        KJ1         #
######################

gp_amp_input = 32.55 #from FF' fits
gp_explength_input = 9.6188 #from FF' fits
gp_per_input = 8.429 #from ACF
gp_perlength_input = 0.402 #from FF' fits

#hyperparameter maximums
gp_amp_max = 100.
gp_explength_max = 10000
gp_per_max = 9.
gp_perlength_max = 1.

#hyperparameter minimums
gp_amp_min = 1.
gp_explength_min = 8.7
gp_per_min = 8
gp_perlength_min = 0.1

#guassian prior means
gp_amp_mu = 32.55
gp_explength_mu = 9.6188
gp_per_mu = 8.429
gp_perlength_mu = 0.4402

#gaussian prior standard deviations
gp_amp_sd = 6.15
gp_explength_sd = 0.871
gp_per_sd = 0.094
gp_perlength_sd = 0.0499



###########################
#    Start with TTVFast   #
###########################

#we define the planets object for each planet in our system
planets = []
for i in range(n_p):
    planet_tmp = models.Planet(
        mass=mass_input[i], #0.00002878248,                         # M_sun
        period=period_input[i], #1.0917340278625494e+01,              # days
        eccentricity=e_input[i], #5.6159310042858110e-02,
        inclination=i_input[i], #9.0921164935951211e+01,         # degrees
        longnode=longnode_input[i], #-1.1729336712101943e-18,           # degrees
        argument=argument_input[i], #1.8094838714599581e+02,            # degrees
        mean_anomaly=M_ref_input[i] #-8.7093652691581923e+01,       # degrees mean anomaly at the reference time not at the time of transit
    )
    planets.append(planet_tmp)



T_st = t_ref # days start of integration
dt = np.min(period_input)/25. # days <1/20 of the smallest period
T_fi = 1800+T_st # days end point for integration (days)




###########################
#      Read Datafiles     #
###########################

#read in transit times
global epoch_obs, transit_time_obs, o_c, o_c_unc, n_data #these values do not change

#TTVFast wants these in list format
epoch_obs= []
transit_time_obs= []
o_c= []
o_c_unc= []
n_data =0

#loop through each planet's textfile and grab the transit times
for i in range(n_p):
    newdata = pd.read_csv('Data/ttv_'+str(i)+'.txt',sep = '\t')
    epoch_obs_tmp = np.array(newdata['epoch'].to_numpy(dtype=int))
    if i == 6:
    #we manually set the single epoch value for the seventh planet
    #This is the number of transits since the reference time defined above
        if h_period == 80.:
            epoch_obs_tmp = np.array([9]) #the epoch if the period is near 80
        elif h_period == 160.:
            epoch_obs_tmp = np.array([4]) #the epoch if near 160

    #generate predicted transit times from a linear fit
    #we do this to generate o-c plots
    transit_time_linear = (epoch_obs_tmp)*period_input[i] + tc_input[i]
    if i == 6:
        transit_time_linear = tc_input[i] #only the one transit


    transit_time_obs_tmp = newdata['tc'].to_numpy(dtype=float) #observed transit times

    o_c_tmp =  transit_time_obs_tmp - transit_time_linear #difference between predicted and observed
    o_c_unc_tmp = newdata['unc'].to_numpy(dtype=float) #uncertainty in transit times
    epoch_obs.append(epoch_obs_tmp) #we append a list of epochs for the ith planet to a list of all transit times
    transit_time_obs.append(transit_time_obs_tmp) #ditto transit times
    o_c.append(o_c_tmp) #ditto o-c values
    o_c_unc.append(o_c_unc_tmp)#ditto uncertainties
    n_data+=len(epoch_obs_tmp) #for keeping track of our number of datapoints


#adjust the epochs of planets b and c
#these values are one smaller than they should be because of where I defined my reference time, so I adjust
epoch_obs[0] += 1
epoch_obs[1] += 1




global p_ave,tc_ave
p_ave = np.ones(n_p)
tc_ave = np.ones(n_p)
new_oc = []
for i in range(n_p):
    #O-C plots actually look much better when I calculate the average period from the data, rather than using the values in D23
    #I'm not 100% sure why, but I'm guessing it my be due to a slightly different definition of "period", since orbital periods
    #in a system with TTVs isn't extremely well defined.
    p_ave[i],tc_ave[i]= determine_period(epoch_obs[i],transit_time_obs[i],o_c_unc[i]) #calculate average

    newdata = pd.read_csv('Data/ttv_'+str(i)+'.txt',sep = '\t')

    #generate transit times
    transit_time_linear = (epoch_obs[i])*p_ave[i] + tc_ave[i] #make linear times from the average period
    transit_time_obs_tmp = newdata['tc'].to_numpy(dtype=float)

    o_c_tmp =  transit_time_obs_tmp - transit_time_linear #make the o-c for this planet
    new_oc.append(o_c_tmp)

o_c_from_avg = np.array(new_oc) #calculated from average works way better

###########################
#      Import RV Data     #
###########################

#read rv times, if no rv available just create a single epoch
rvdata = pd.read_csv('Data/TOI1136_rv.csv', sep=',')
global x_rv, y_rv, yerr_rv, tel, rvtimes, t #these values do not change
x_rv = rvdata['time'].to_numpy(dtype=float) - 2458000 #put it at the same reference point as TTVs
#for some reason we need a list, not a numpy array
rvtimes =[]
for i in range(len(x_rv)):
    rvtimes.append(x_rv[i])
t = []

thing = np.linspace(np.min(x_rv), np.max(x_rv), 10000)
for i in range(len(thing)):
    t.append(thing[i])
y_rv = rvdata['mnvel'].to_numpy(dtype=float) #radial velocity value
yerr_rv = rvdata['errvel'].to_numpy(dtype=float) #radial velocity error
tel = rvdata['tel'].to_numpy(dtype=str) #associated telescope/instrument
telnames = ['hires_j', 'apf', 'harpsn'] #the instruments in the fit

################################
#    Set up the parameters     #
################################

global para_name
parameters = []
para_name = []
para_max =[] #max allowed range during optimization, NOT during MCMC
para_min =[] #minimum allowed range during optimization, NOT during MCMC
prior_type = [] #I've coded in either uniform or gaussian

#add all the planet parameters first
#loop by planets
for i in range(n_p):

    #####################
    #  Starting Guess.  #
    #####################

    parameters += [mass_input[i], #planet-star mass ratio
                   period_input[i], #period (in days)
                   e_input[i]**0.5*np.sin(argument_input[i]/180*np.pi), #sesinw
                   e_input[i]**0.5*np.cos(argument_input[i]/180*np.pi), #secosw
                   i_input[i], #inclination
#                   longnode_input[i], #longitude of ascending node, we fix this
                   M_ref_input[i]] #mean anomaly (degrees)

    #####################
    #.    Param Max     #
    #####################
    if i < 6:
        para_max+=[0.01, #planet-star mass ratio
                   period_input[i]*1.01, #period
                   1.0, #sesinw
                   1.0, #secosw
                   90.3, #0.3,#maximum geometric transiting angle
                   #180., #long of ascending node
                   180.] #mean anomaly
    #priors are a bit different for 1136.07
    else:
        para_max+=[0.01, #plausible maximum is about a Jupiter mass
                   1000., #period
                   1.0, #sesinw
                   1.0, #secosw
                   100., #inclination
                   #180., #long of ascending node
                   180.] #mean anomaly

    #####################
    #     Param Min     #
    #####################
    if i < 6:
        para_min+=[0., #planet-star mass ratio
                   period_input[i]*0.99, #period
                   -1., #sesinw
                   -1., #secosw
                   89.3, #
                   #-180., #long of ascending node
                   -180.] #mean anomaly
    else:
        para_min+=[0., #planet-star mass ratio
                   1., #period
                   -1., #sesinw
                   -1., #secosw
                   80., #-0.3, #cosine of inclination
                   #-180., #long of ascending node
                   -180.] #mean anomaly


    ######################
    #     Prior Type     #
    ######################
    if i < 6:
        prior_type+=['uniform',
                      'uniform',
                       'uniform',
                        'uniform',
                         'gaussian', #inclination has a gaussian prior
                          'uniform']
    else:
        prior_type+=['uniform',
                      'gaussian', #this will actually have a gaussian prior
                       'uniform',
                        'uniform',
                         'uniform', #uniform prior on planet h's inclination
                          'uniform']

    #names for all
    #TTVFast Params
    para_name.append('mass'+str(i))
    para_name.append('period'+str(i))
    para_name.append('e_sqrt_sin'+str(i))
    para_name.append('e_sqrt_cos'+str(i))
    para_name.append('inclination'+str(i))  # para_name.append('cos_i'+str(i))
   # para_name.append('longnode'+str(i))
    para_name.append('mean_anomaly'+str(i))

# #instrument parameters
for inst in telnames:
    parameters.append(gammas[inst]) #RV offset
    para_max.append(gamma_max) #max value
    para_min.append(gamma_min) #min value
    para_name.append('gamma_{}'.format(inst)) #name
    prior_type.append('uniform')

    parameters.append(jitters[inst]) #RV jitter (in m/s)
    para_max.append(jitter_max)
    para_min.append(jitter_min)
    para_name.append('jitter_{}'.format(inst))
    prior_type.append('uniform')

#we only want GP Params to vary in the MCMC, not in the optimization
#this is because we've already constrained them ahead of time pretty well with FF'
#also, the optimization sometimes goes a little crazy if GP params can vary in there

#copy the stuff from before, and then add the GP parameters
mcmc_parameters = np.copy(parameters)
mcmc_max = np.copy(para_max)
mcmc_min = np.copy(para_min)
mcmc_name = np.copy(para_name)
mcmc_prior = np.copy(prior_type)

#GP Params
mcmc_parameters = np.append(mcmc_parameters, gp_amp_input) #one for reach instrument
mcmc_parameters = np.append(mcmc_parameters, gp_amp_input)
mcmc_parameters = np.append(mcmc_parameters, gp_amp_input)
mcmc_parameters = np.append(mcmc_parameters, gp_explength_input)
mcmc_parameters = np.append(mcmc_parameters, gp_per_input)
mcmc_parameters = np.append(mcmc_parameters, gp_perlength_input)


#we create arrays called prior_a and prior_b to accomodate gaussians
#bascially, everything before now was uniform, so "max" and "min" made sense for names
#This doesn't make much sense when the priors are Gaussian, so we call them prior_a and prior_b

prior_a = np.copy(mcmc_min) #lower end of uniform priors, and gaussian mean
prior_b = np.copy(mcmc_max) #upper end of uniform priors, and gaussian standard deviation

mcmc_max = np.append(mcmc_max, gp_amp_max) #one for each instrument
mcmc_max = np.append(mcmc_max, gp_amp_max)
mcmc_max = np.append(mcmc_max, gp_amp_max)
mcmc_max = np.append(mcmc_max, gp_explength_max)
mcmc_max = np.append(mcmc_max, gp_per_max)
mcmc_max = np.append(mcmc_max, gp_perlength_max)

mcmc_min = np.append(mcmc_min, gp_amp_min) #one for each instrument
mcmc_min= np.append(mcmc_min, gp_amp_min)
mcmc_min = np.append(mcmc_min, gp_amp_min)
mcmc_min = np.append(mcmc_min, gp_explength_min)
mcmc_min = np.append(mcmc_min, gp_per_min)
mcmc_min = np.append(mcmc_min, gp_perlength_min)


mcmc_name = np.append(mcmc_name, 'gp_amp_hires_j')
mcmc_name = np.append(mcmc_name, 'gp_amp_apf')
mcmc_name = np.append(mcmc_name, 'gp_amp_harpsn')
mcmc_name = np.append(mcmc_name, 'gp_explength')
mcmc_name = np.append(mcmc_name, 'gp_per')
mcmc_name = np.append(mcmc_name, 'gp_perlength')

mcmc_prior = np.append(mcmc_prior, 'gaussian')
mcmc_prior = np.append(mcmc_prior, 'gaussian')
mcmc_prior = np.append(mcmc_prior, 'gaussian')
mcmc_prior = np.append(mcmc_prior, 'gaussian')
mcmc_prior = np.append(mcmc_prior, 'gaussian')
mcmc_prior = np.append(mcmc_prior, 'gaussian')

##########################
#     Prior A and B.     #
##########################

prior_a = np.append(prior_a, gp_amp_mu) #one for each instrument
prior_a = np.append(prior_a, gp_amp_mu)
prior_a = np.append(prior_a, gp_amp_mu)
prior_a = np.append(prior_a, gp_explength_mu)
prior_a = np.append(prior_a, gp_per_mu)
prior_a = np.append(prior_a, gp_perlength_mu)

prior_b = np.append(prior_b, gp_amp_sd) #one for each instrument
prior_b = np.append(prior_b, gp_amp_sd)
prior_b = np.append(prior_b, gp_amp_sd)
prior_b = np.append(prior_b, gp_explength_sd)
prior_b = np.append(prior_b, gp_per_sd)
prior_b = np.append(prior_b, gp_perlength_sd)



#modify inclination prior stuff
#during optimization we put mins and maxes on Inclination
#Now we change them to Gaussian priors, as described in the paper
i_mu = [86.44, 89.42, 89.41, 89.31, 89.38, 89.65,89.5] #taken from D23 for b-g, planet h is a lower limit from geometry
i_sd = [2.7e-1, 5.5e-1, 2.8e-1, 2.6e-1, 2.2e-1, 1.8e-1,90.5] #planet h uniform ends are attached at the end here

for i in range(len(i_mu)):
    prior_a[i*6+4] = i_mu[i] #change to mean
    prior_b[i*6+4] = i_sd[i] #change to sd

#change planet h's period prior
prior_a[37] = 155.53 #change it to NS posterior
prior_b[37] = 5.21



#we define a p_fit object for lmfit optimization
p_fit = lmfit.Parameters()
for i in range(len(parameters)):
    p_fit.add(para_name[i],parameters[i], min = para_min[i], max =para_max[i])

#what does an RV fit look like with initial parameters
plot_rv_results(mcmc_parameters, t, 'initial')


#number of free parameters, useful later
global N
N = len(p_fit)


#optimize everything but GP hyperparameters, since those are already trained
#We utilize lmfit's least squares, and we use an estimate of the error for initial variance
results = lmfit.minimize(residual, p_fit, method='least_squares',calc_covar=True)
#check a summary
print(lmfit.fit_report(results))

#creating a list of results that isn't an lmfit object
re = []
for i in range(len(parameters)):
    re.append(results.params[para_name[i]].value)

#remember we didn't include gp parameters in our initial optimization
#now we attach them
#attach the gp parameters to the re array
re.append(gp_amp_input) #one for each instrument
re.append(gp_amp_input) #one for each instrument
re.append(gp_amp_input) #one for each instrument
re.append(gp_explength_input)
re.append(gp_per_input)
re.append(gp_perlength_input)



#plot the results after optimization
plot_results(re,version ='ttvfast_lmfit',overplot=False) #TTV O-C fit
plot_rv_results(re, t, version='post_lmfit') #RV fit


#estimnate the BIC
BIC = -2.0*lnprob(re)+len(re)*np.log(n_data)
print('BIC',BIC)

#sometimes the gamma parameters would go to unreasonably strange values after optimization
#set the gammas to 0 here and let the mcmc change them
for i in range(len(para_name)):
    if 'gamma' in para_name[i]:
        re[i] = 0
    else:
        continue


#now mcmc
#First we copy the optimized array
initial = np.array(re)

#now we need an array of "errors" estimated from the covariance matrix of the least squares fit
#We will use this to control how much initial dispersion to give the parameters
initial_err = np.copy(initial)
print(np.shape(results.covar),len(initial))
for i in range(len(p_fit)):
    initial_err[i] = np.sqrt(results.covar[i,i])

#need to add mcmc_stuff to error, because these weren't fit during the optimization
#These errors are taken as a result of our FF' fit
initial_err[-6] = 6.15 #amp error
initial_err[-5] = 6.15
initial_err[-4] = 6.15
initial_err[-3] = 0.871 #explength error
initial_err[-2] = 0.09 #period error
initial_err[-1] = 0.0499 #perlength error

ndim, nwalkers = len(initial), 1000 #use "hundreds" of walkers as foreman-mackey suggests in the paper
#We multiply the covariance by 1e-5, add some randomness
#It would probably be better to pull initial positions from, say, the priors, but this would result in
#A covariance matrix of the RVs with a high condition number fairly frequently, and was hard in practice
# We insured that our acceptance rate was initially high (~30%), and our trend plots suggest the walkers were able to leave starting positions
# And so we proceed
pos = [initial+1e-5*initial_err*np.random.randn(ndim) for i in range(nwalkers)]

#We experimented with different moves
#The GaussianMove is a basic MH move, and never worked very well. Probably could work with careful scrutiny of the
#covariance, but this is different for each parameter, and time consuming to optimize
#new_moves = emcee.moves.GaussianMove(cov=initial_err*1e-1,mode='random') #use initial error for now

#The stretch move is the default move in emcee, and worked fairly well. We found a~1.2 worked best for an initially good acceptance rate
new_moves = emcee.moves.StretchMove(a=1.2)

#The Differential Evolution Move is supposedly better in higher dimensions, and so we ended up using this as our final move
# Its results were basically consistent with the stretchmove, though it had a higher acceptance rate
#new_moves = emcee.moves.DEMove(gamma0=0.5)

try:
    #load the previous mcmc steps for convenience
    filename = "7p_chains.h5"
    backend = emcee.backends.HDFBackend(filename)
    print("Initial size: {0}".format(backend.iteration))
    #parallelize
    from multiprocessing import Pool
    if multi_cpu:
        with Pool(2) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, pool=pool, moves=new_moves)
            samples = sampler.run_mcmc(None, nsteps, progress=True)
        print("Final size: {0}".format(backend.iteration))
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, moves=new_moves)
        samples = sampler.run_mcmc(None, nsteps, progress=True)
        print("Final size: {0}".format(backend.iteration))
except (FileNotFoundError, OSError):
    #no previous steps?
    #make a new one and save it
    print('Starting chain from scratch')
    filename = "7p_chains.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    from multiprocessing import Pool
    #parallelize
    if multi_cpu:
        with Pool(2) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend,pool=pool,moves=new_moves)
            samples = sampler.run_mcmc(pos, nsteps, progress=True)

    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend, moves=new_moves)
        samples = sampler.run_mcmc(pos, nsteps, progress=True)


totlength = backend.iteration #total lenght of chains

print('Generating first samples for trend plots')
trend_cols = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11']
samples = sampler.get_chain(discard=0) #don't want to discard for trend plots
labels = mcmc_name
plt.figure(figsize=(14, 5))
for i in range(ndim):
    plt.figure(figsize=(14,5))
    pick_counter = 0
    #randomly pick 10 walkers to follow and inspect thoughout
    nrand = np.random.randint(low=0, high=nwalkers, size=10)
    for j in range(nwalkers):
        color = 'gray'
        alpha=0.3
        z_order=-1
        if j in nrand:
            color = trend_cols[pick_counter]
            alpha=1
            z_order=10
            pick_counter+=1
        plt.plot(samples[:, j, i], alpha=alpha, color=color,zorder=z_order)

    plt.xlim(0, len(samples))
    plt.ylabel(labels[i])

    plt.savefig('7pPlots/Trends/Trend_plot_{}.png'.format(labels[i]))
    plt.close()




import corner


flat_samples = sampler.get_chain(discard=0, thin=1, flat=True) #we do want to discard burnin for corner/posteriors
truths = []
#truths plot will highlight D23 values as blue lines
for i in range(n_p):
    truths.append(mass_input[i]* stellar_mass*1/(Mearth2Msun))
    truths.append(period_input[i])
    truths.append(e_input[i]**0.5*np.sin(argument_input[i]/180*np.pi))
    truths.append(e_input[i]**0.5*np.cos(argument_input[i]/180*np.pi))
    truths.append(i_input[i])
    truths.append(M_ref_input[i])#mean anomaly

#pick parameters to highlight in corner plot
#This model has way too many parameters to highlight everything, it would be unreadable
sub = flat_samples[:,::6]
sub = sub[:,:7]
sub = np.append(sub,flat_samples[:,37:41], axis=1)

#we grab for labels too
sublabels = labels[::6]
sublabels = sublabels[:7]
sublabels = np.append(sublabels,labels[37:41])
subtruths = truths[::6]
subtruths = subtruths[:7]
subtruths = np.append(subtruths,truths[37:41])

#convert mass ratio to earth mass
conv = np.repeat(stellar_mass * 1/Mearth2Msun, len(sublabels))
conv[-4:] = 1

subtruths[-5:] = -1000000 #don't want the lines for these parameters

good_labels = np.array(['Mass b',
          'Mass c',
          'Mass d',
          'Mass e',
          'Mass f',
          'Mass g',
          'Mass h',
          'Period h',
          r'$\sqrt{e}\sin\omega_{h}$',
          r'$\sqrt{e}\cos\omega_{h}$',
          'Inclination h'])


fig = corner.corner(
    sub[:,:]*conv, labels=good_labels, truths=subtruths
);

fig.savefig('7pPlots/7p_transit_RV_corner.png')
plt.close()


#do pair plots for eccentricities of each planet
pl_colors = ['red', 'blue', 'green', 'purple', 'gold', 'black', 'teal']
pl_letters = ['b', 'c', 'd', 'e', 'f', 'g', 'h']


for n in ttv_fit[:6]:

    ecc_sub1 = flat_samples[:,[n*6+2,n*6+3]]
    ecc_sub2 = flat_samples[:,[(n+1)*6+2,(n+1)*6+3]]

    limits=[(-0.4,0.4),(-0.4,0.4)]
    if n == 5:
        limits=[(-0.4, 0.4),(-0.4,0.6)]
    fig1 = corner.corner(ecc_sub1, color=pl_colors[n])

    fig2 = corner.corner(
        ecc_sub2, labels=[r'$\sqrt{e}\cos\omega$', r'$\sqrt{e}\sin\omega$'],
        fig=fig1, range=limits, color=pl_colors[n+1], truths=[0,0]
    );

    plt.legend([None,pl_letters[n],pl_letters[n+1]], fontsize=14)

    plt.savefig('7pPlots/ecc_plot_{}.png'.format(n))
    plt.close()





indx = np.argmax(sampler.get_log_prob(flat=True))
other_samples = sampler.get_chain(flat=True)#a flattened chain object for grabbing the maximium likelihood
chain_samples = sampler.get_chain(discard=totlength-100, thin=1, flat=False) #this is the last 100 steps of the chain for plotting blue lines on the plot
chain_end = chain_samples[-1,:,:] #we grab the last step for these last chains
final = other_samples[indx] #grab the parameter values at the maximum likelihood


plot_results(final,version ='ttvfast_mcmc_median',samples=chain_end,nindx=50,overplot=False)


#now we grab median values for the RV plot
final_med = []
final_sd = []
for i in range(len(flat_samples[0,:])):
    final_med.append(np.median(flat_samples[:,i]))
    final_sd.append(np.std(flat_samples[:,i]))

final_med = np.array(final_med)
final_sd = np.array(final_sd)

plot_rv_results(final_med, t, version='post_mcmc', error=final_sd)



Mearth2Msun = (ac.M_earth/ac.M_sun).value
Msun2Mearth = 1/(Mearth2Msun)
#start writing posteriors
text_file = open("7p_Transit_RV_K1_posteriors.txt", "w")

for p in range(len(mcmc_name)):
    name = mcmc_name[p]

    if 'mass' in name:
        #convert to Earth Mass
        med = float(np.percentile(flat_samples, [50],axis=0)[0][p]) #grab median percentile
        med *= stellar_mass #right now it's a mass ratio, this converts to solar mass
        med *= Msun2Mearth #convert back to Earth
        upper = float(np.percentile(flat_samples, [84],axis=0)[0][p]*(stellar_mass*Msun2Mearth)) - med #upper limit 1 sigma
        lower = med - float(np.percentile(flat_samples, [16],axis=0)[0][p])*(stellar_mass*Msun2Mearth) #lower limit 1 sigma
    else:
        med = np.percentile(flat_samples, [50],axis=0)[0][p] #median
        upper = np.percentile(flat_samples, [84],axis=0)[0][p] - med #1 sigma upper
        lower = med - np.percentile(flat_samples, [16],axis=0)[0][p] #1 sigma lower
    if p % 6 == 0:
        text_file.write('-----------\n') #for readability
    text_file.write('{}: {} + {} - {}\n'.format(name, med, upper, lower))
text_file.close()



#3 sigma upper limit for h
h3sig = np.percentile(flat_samples, [99.7],axis=0)[0][36]
print('3 sigma upper limit for planet h {}'.format(h3sig))

#calculate the Gelman-Rubin statistic
def Calculate_GR(samples):


    L = len(samples[0]) #number of steps in each chain
    J = len(samples[1]) #number of chains

    chain_means = []
    #first calculate the mean of each chain
    for j in range(J):
        xj = np.mean(samples[:,j,:],axis=0)
        chain_means.append(xj)

    chain_means = np.array(chain_means)

    grand_mean = np.mean(chain_means,axis=0)

    B = np.var(chain_means,axis=0) #interchain variance

    intrachain_var = [] #intrachain variance
    for j in range(J):
        intrachain_var.append(np.var(samples[:,j,:],axis=0))

    intrachain_var = np.array(intrachain_var)

    W = np.mean(intrachain_var,axis=0)

    #finally, calculate the Gelman-Rubin statistic

    GR = ((L-1)/(L)*W + (1/L)*B)/(W)

    return GR


GR = Calculate_GR(samples)

for i in range(len(mcmc_name)):
    print('{}: {}'.format(mcmc_name[i], GR[i]))


#print the acceptance fraction
print('Acceptance Fraction')
print(sampler.acceptance_fraction)
print('Mean acceptance: {}'.format(np.mean(sampler.acceptance_fraction)))
#print the other acor
print('Autocorrelation other way')
print(sampler.acor)



#autocorrelation estimates
ac_dict = {}
for j in range(len(para_name)):
    ac = []
    for i in range(nwalkers):
        thing = samples[:, i, j]
        ac.append(emcee.autocorr.integrated_time(thing, quiet=True)[0])

    ac_dict[para_name[j]] = np.mean(ac)

print(ac_dict)

for i in range(len(ac_dict)):
    print(ac_dict[para_name[i]]*50)

med_ac = np.median(list(ac_dict.values()))

try:
    ac_data = pd.read_csv('ac_7p.csv',sep=',')
    ac_data[totlength] = [med_ac]
    ac_data.to_csv('ac_7p.csv',sep=',',index=False)
except FileNotFoundError:
    ac_data = pd.DataFrame()
    ac_data[totlength] = [med_ac]
    print(ac_data)
    ac_data.to_csv('ac_7p.csv',sep=',',index=False)

#make a plot of log likelihoods over time
lks = np.median(sampler.get_log_prob(), axis=1)

plt.figure(figsize=(14,8))
plt.plot(lks, color='black')

plt.xlabel('MCMC Steps', fontsize=20)
plt.ylabel('Log Likelihood')

plt.savefig('7pPlots/Like_Plot.png',)
