#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 11:31:00 2025

@author: ebotica
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, time, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit


### From this line until polmod line, it would need to be changed for each analysis
sample_ID = '042_10_pulsed_cropped'
save_dir = '/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/Irradiations/20250911/'
os.makedirs(save_dir+sample_ID, exist_ok=True)
read_file = pd.read_csv(r'/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/TGS/Data/PI3/042/Beam_pulsed_042_10_RT_lecroy/2025-09-11/Beam_pulsed_042_10_RT_lecroy-2025-09-11-03.40um-spot00-_postprocessing.txt', sep=',', on_bad_lines='skip', header=None)
read_file2 = pd.read_excel(save_dir+'Irradiations_conditions_beam_checked.xlsx', sheet_name=0)
t0  = '11:12:14' #Write initial time from one of the TGS raw data files in 24h-format
t00 = '11:20:15' #Write initial time from TGS irradiated measurements in 24h-format
tf  = '12:31:06' #Write final time from TGS irradiated measurements in 24h-format
tfff= '12:39:07' #final measurement

polmod=0 #by default 1 to see if it needs background subtraction, =0 if it doesnt need it

tirr = 208
FMT = '%H:%M:%S'
t000 = datetime.strptime(t00, FMT) - datetime.strptime(t0, FMT)
tff = datetime.strptime(tf, FMT) - datetime.strptime(t0, FMT)
tir = t000.total_seconds() - tirr

############## Species related data ################
Barea = 4.44288E-6  # nm2
charge = 3
atoms_cc=8.45e+27
e_charge=1.602e-19
K_SRIM = 0.85
#dpa_cte = (10 * K_SRIM) / (charge * 85 * (1.602e-10) * Barea * 1e14)
dpa_cte = K_SRIM/(Barea * charge * e_charge * atoms_cc)
####################################################

tgs_file = np.array(read_file)
current_file = np.array(read_file2)
l = len(tgs_file)
c = len(current_file)
tsum=np.zeros(l)
ti=np.zeros(l)
ts=0
tc=0
o=0
cum_dpa=0
DPA=np.zeros((l,2))
diff=np.zeros(l)
diff_err=np.zeros(l)
sawf=np.zeros(l)
sawf_err=np.zeros(l)
saw_speed=np.zeros(l)
current=np.zeros((l,2))
curr=[[],[],[],[]]
temp=[[],[]]
temp_plot=[[],[],[]]
tgs_data=[[],[],[],[],[],[],[]]



# Default font styles for all plots
plt.rcParams['font.family'] = 'serif'  # Default font family
plt.rcParams['font.size'] = 12              # Default font size for text
plt.rcParams['axes.titlesize'] = 14         # Font size for titles
plt.rcParams['axes.labelsize'] = 14         # Font size for axis labels
plt.rcParams['xtick.labelsize'] = 12        # Font size for x-axis ticks
plt.rcParams['ytick.labelsize'] = 12        # Font size for y-axis ticks
plt.rcParams['legend.fontsize'] = 14        # Font size for legend
plt.rcParams['figure.titlesize'] = 16       # Font size for figure title (if used)


for i in range(1,l):
    t1=tgs_file[i][0].split()[21]
    tinc = datetime.strptime(t1, FMT) - datetime.strptime(t0, FMT)
    ti[i]=float(tinc.total_seconds())
    ts+=tinc.total_seconds()
    tsum[i]=float(ts)
    # t0=tgs_file[i][0].split()[21]
    diff[i]=float(tgs_file[i][0].split()[5])
    diff_err[i]=float(tgs_file[i][0].split()[6])
    sawf[i]=float(tgs_file[i][0].split()[2])
    sawf_err[i]=float(tgs_file[i][0].split()[3])
    saw_speed[i]=float(tgs_file[i][0].split()[4])
    tgs_data[0].append(tgs_file[i][0].split()[21])
    tgs_data[1].append(float(tgs_file[i][0].split()[2]))
    tgs_data[2].append(float(tgs_file[i][0].split()[3]))
    tgs_data[3].append(float(tgs_file[i][0].split()[4]))
    tgs_data[4].append(float(tgs_file[i][0].split()[5]))
    tgs_data[5].append(float(tgs_file[i][0].split()[6]))
    tgs_data[6].append(float((datetime.combine(datetime.today(), datetime.strptime(tgs_file[i][0].split()[21], FMT).time()) - datetime.combine(datetime.today(), datetime.strptime(tgs_file[1][0].split()[21], FMT).time())).total_seconds()))
    
   

### Current extraction ###
e=datetime.strptime(t00, FMT).time() #Initial irradiation time
d=datetime.strptime(tf, FMT).time() #Final irradiation time
cum_dpa=0

for kk in range(0,len(current_file[0])-1):
    if current_file[kk,0]<=e and current_file[kk+1,0]>e:
        curr_ti=kk
    if current_file[kk,0]<=d and current_file[kk+1,0]>d:
        curr_tf=kk
for k in range(0,c):
    curr[0].append(current_file[k,0])  #time
    curr[1].append(float(current_file[k,1]))  #current
    curr[2].append(float((datetime.combine(datetime.today(), current_file[k,0]) - datetime.combine(datetime.today(), current_file[0,0])).total_seconds()))  #accumulated time
    if k == 0:
        # For first point, use a small time interval or assume initial damage is 0
        time_interval = 1.0  # 1 second default
        dpa = float(current_file[k,1]) * dpa_cte * time_interval
    else:
        # For subsequent points, calculate actual time interval
        time_interval = float((datetime.combine(datetime.today(), current_file[k,0]) - 
                             datetime.combine(datetime.today(), current_file[k-1,0])).total_seconds())
        dpa = float(current_file[k,1]) * dpa_cte * time_interval
        
    
    # dpa=float(current_file[k,1])*(dpa_cte)*(float((datetime.combine(datetime.today(), current_file[k,0]) - datetime.combine(datetime.today(), current_file[k-1,0])).total_seconds()))
    cum_dpa=cum_dpa+dpa
    curr[3].append(cum_dpa)
    if float((datetime.combine(datetime.today(), current_file[k,0]) - datetime.combine(datetime.today(), current_file[1,0])).total_seconds()) >= tgs_data[6][len(tgs_data[6])-1]:
        break
    
    
### DPA value match with data ###

dpa_int=np.interp(tgs_data[6],curr[2],curr[3])



######################## Until this point regular TGS code

# Fixed initialization
y0 = min(tgs_data[1]) if tgs_data[1] else 0  # Find minimum SAW frequency

y1_start = []  # start times when current becomes zero
y1_end = []    # end times when current becomes non-zero again
y1_start.append(curr[2][0])
curr_end = []
curr_start = []


for i in range(1, len(curr[1])):
    if curr[1][i] == 0 and curr[1][i-1] > 0:
        y1_start.append(curr[2][i])  # use seconds, not datetime objects
        curr_start.append(curr[1][i-1])

    
    if curr[1][i] > 0 and curr[1][i-1] == 0 and i > 1:
        y1_end.append(curr[2][i-1])  # use seconds
        curr_end.append(curr[1][i])

# y1_end.pop(0)
y1_end.append(curr[2][len(curr[2])-1])

sawfit0_time = []    # time values during current=0
sawfit0_sawf = []    # SAW frequency values during current=0

for j in range(len(tgs_data[6])):
    current_time = tgs_data[6][j]
    is_in_zero_interval = False
    # Check if this time falls within ANY current=0 interval
    for k in range(min(len(y1_start), len(y1_end))):
        if y1_start[k] <= current_time <= y1_end[k]:
            is_in_zero_interval = True
            break
    # Add to sawfit0 only if NOT in any current=0 interval
    if not is_in_zero_interval:
        sawfit0_time.append(current_time)
        sawfit0_sawf.append(tgs_data[1][j])

sawfit0_time = np.array(sawfit0_time)
sawfit0_sawf = np.array(sawfit0_sawf)
av_curr=np.mean(curr_start)
##Plot for the selected data
plt.figure(figsize=(12, 6))
plt.scatter(tgs_data[6], tgs_data[1], alpha=0.5, color='tab:cyan',label='Beam off', s=10)
plt.scatter(sawfit0_time, sawfit0_sawf, color='#FFA500', label=f'Beam on = {av_curr:.0f}nA', s=20)

# Mark the current=0 intervals
for k in range(0, len(y1_end)):
    plt.axvspan(y1_start[k], y1_end[k], alpha=0.2, color='gray', label='Current = 0nA' if k == 0 else "")

plt.xlabel('Time [s]')
plt.ylabel('SAW Frequency [Hz]')
plt.title('Data Selection')
plt.ticklabel_format(style='plain', axis='x') 
plt.legend()
# plt.ylim((5.5e8,5.62e8))
plt.xlim((000,1250))
plt.grid(True, alpha=0.3)
plt.savefig(save_dir + sample_ID +'/'+sample_ID +'_Data_selection.png', dpi=600)
plt.show()



if len(sawfit0_time) > 0:
    if polmod==1:
        # Fit a polynomial to model the hardening trend (simpler than symbolic regression)
        # Use a low-order polynomial to capture the general trend
        poly_coeffs = np.polyfit(sawfit0_time, sawfit0_sawf, 2)  # quadratic fit
        poly_func = np.poly1d(poly_coeffs)
        hardening_trend = poly_func(tgs_data[6]) # Predict the hardening trend for all time points
        sawfit1_sawf = y0 + np.array(tgs_data[1]) - hardening_trend # Subtract hardening effect and add offset
        plt.figure(figsize=(12, 6))
        plt.scatter(sawfit0_time, sawfit0_sawf, label='SAWf during current=0', alpha=0.5, color='#008080')
        plt.plot(tgs_data[6], hardening_trend, 'r-', label='Secondary trend fit')
        plt.scatter(tgs_data[6], sawfit1_sawf, label='Normalized SAW Frequency', alpha=0.6, color='tab:orange')
        plt.xlabel('Time [s]')
        plt.ylabel('SAW Frequency [Hz]')
        plt.title('SAW Frequency Normalization')
        plt.ticklabel_format(style='plain', axis='x') 
        # plt.ylim((5.5e8,5.6e8))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_dir + sample_ID +'/'+sample_ID +'_Fit+Correction.png', dpi=600)
        plt.show()
    else:
        sawfit1_sawf = np.array(tgs_data[1])
    
else:
    print("No current=0 periods found for hardening analysis")
    sawfit1_sawf = np.array(tgs_data[1])  # use original data


vac_results=[[],[],[],[],[],[],[]]

############     Beam ON fitting      ############

# y1_end=y1_end[0:5]+y1_end[6::]# This just apply to 044_18 as the beam was accidentally off
for hh in range(len(y1_end)):  # for each current start time   
    current_start_time = y1_end[hh] # Use the exact time when current becomes non-zero
    fit_start_time = current_start_time - 20  # Include points BEFORE current starts to get proper initial value
    cumulative_dpa = np.interp(current_start_time, curr[2], curr[3])
    # for log DPA experiments use this window option
    # if cumulative_dpa<0.3:
    #     window=40
    # elif cumulative_dpa<1.2:
    #     window=100
    # else:
    #     window=120
    # For pulsed beam experiments use this window option
    window = 120
    
    # Find data points including the transition (before AND after current starts)
    mask = (np.array(tgs_data[6]) >= fit_start_time) & (np.array(tgs_data[6]) <= current_start_time + window)
    if np.sum(mask) > 3:
        x_data = np.array(tgs_data[6])[mask] - current_start_time  # Time since current start
        y_data = sawfit1_sawf[mask]
        
        print(f"\n=== Fitting for current start at {current_start_time}s ===")
        print(f"Data points: {len(x_data)}")
        print(f"x range: {x_data.min():.1f} to {x_data.max():.1f}s")
        print(f"y range: {y_data.min():.3e} to {y_data.max():.3e}Hz")
        # pr = Poisson's ratio value for Cu
        pr = 0.34
        # c = a, parameter in the formula of the Elastic modulus of a porous material
        c = (3*(9+5*pr)*(1-pr))/(2*(7-5*pr))
        
        # Model function
        def model_function(x, a, b):
            result = np.zeros_like(x)
            mask_before = x <= 0
            mask_after = x > 0
            
            # Before current start: constant value
            result[mask_before] = b
            
            # After current start: apply the radiation model
            x_after = x[mask_after]
            inner_term = 1 - c * np.sqrt(a * x_after)
            inner_term = np.maximum(inner_term, 1e-10)
            result[mask_after] = b * np.sqrt(inner_term)
            
            return result

        x_fit = x_data
        y_fit = y_data
        try:
            # FORCE b to be the average of pre-current points
            pre_current_mask = x_fit <= 0
            if np.sum(pre_current_mask) > 0:
                b_fixed = np.mean(y_fit[pre_current_mask])
                print(f"FORCED b to be average of pre-current points: {b_fixed:.3e}Hz")
                
                # Fit parameter 'a' since b is fixed
                def constrained_model(x, a):
                    result = np.zeros_like(x)
                    mask_before = x <= 0
                    mask_after = x > 0
                    
                    # Before current start: constant value (the fixed b)
                    result[mask_before] = b_fixed
                    
                    # After current start: apply the radiation model with fixed b
                    x_after = x[mask_after]
                    inner_term = 1 - c * np.sqrt(a * x_after)
                    inner_term = np.maximum(inner_term, 1e-10)
                    result[mask_after] = b_fixed * np.sqrt(inner_term)
                    
                    return result

                def get_a_initial_guess(x_fit, y_fit, c, b_fixed):
                    early_post_mask = (x_fit > 0) & (x_fit < 30) # Use early post-current points to estimate a
                    if np.sum(early_post_mask) >= 2:
                        x_post = x_fit[early_post_mask]
                        y_post = y_fit[early_post_mask]
                        
                        radiation_ratio = (y_post / b_fixed)**2
                        # Solve for a: = 1 - c*sqrt(a*x)
                        if len(radiation_ratio) > 0 and np.all(radiation_ratio < 1):
                            # Use the point with strongest effect
                            idx = np.argmin(radiation_ratio)
                            ratio = radiation_ratio[idx]
                            x_ref = x_post[idx]
                            a_guess = ((1 - ratio) / (c * np.sqrt(x_ref)))**2
                        else:
                            a_guess = 1e-9
                    else:
                        a_guess = 1e-9
                    
                    return max(a_guess, 1e-12)
                
                a_guess = get_a_initial_guess(x_fit, y_fit, c, b_fixed)
                print(f"Initial guess for a: {a_guess:.3e}")
                
                # Fit only parameter 'a'
                try:
                    a_params, a_covariance = curve_fit(constrained_model, x_fit, y_fit, 
                                                     p0=[a_guess], 
                                                     bounds=([1e-15], [5e-5]), 
                                                     maxfev=10000)
                    fitted_a = a_params[0]
                    fitted_b = b_fixed  # b is fixed to the pre-current average
                    
                    print(f"Constrained fit successful: a={fitted_a:.3e}, b={fitted_b:.3e} (fixed)")
                    
                except Exception as e:
                    print(f"Constrained fit failed: {e}")
                    print(f"p0: {a_guess}")
                    # Fallback: use initial guesses
                    fitted_a = a_guess
                    fitted_b = b_fixed

            else:
                # No pre-current points available, use regular fitting
                print("No pre-current points found, using regular fitting")
                b_guess = y_fit[0]
                
                def get_regular_initial_guesses(x_fit, y_fit, c):
                    b_guess = y_fit[0]
                    
                    if len(y_fit) > 1 and x_fit[1] > 0:
                        ratio_squared = (y_fit[1] / y_fit[0])**2
                        if ratio_squared < 1:
                            a_guess = ((1 - ratio_squared) / (c * np.sqrt(x_fit[1])))**2
                        else:
                            a_guess = 1e-8
                    else:
                        a_guess = 1e-8
                    
                    return max(a_guess, 1e-12), b_guess
                
                a_guess, b_guess = get_regular_initial_guesses(x_fit, y_fit, c)
                
                try:
                    params, covariance = curve_fit(model_function, x_fit, y_fit, 
                                                  p0=[a_guess, b_guess], 
                                                  bounds=([1e-15, 0.8 * y_fit[0]], 
                                                         [1e-5,  1.2 * y_fit[0]]), 
                                                  maxfev=10000)
                    fitted_a, fitted_b = params
                except Exception as e:
                    print(f"Regular fit failed: {e}")
                    fitted_a, fitted_b = a_guess, b_guess
            
            # Calculate predictions and R²
            y_pred = model_function(x_fit, fitted_a, fitted_b)
            residuals = y_fit - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"Final results: a={fitted_a:.3e}, b={fitted_b:.3e}, R²={r_squared:.3f}")

            # if r_squared > 0.1:
            #     # Use only positive times for physical calculations
            #     positive_mask = x_fit > 0
            #     if np.sum(positive_mask) > 0:
            #         last_time = np.max(x_fit[positive_mask])
            #     else:
            #         last_time = x_fit[-1]
                    
            #     K0 = dpa_cte * curr_end[hh]
            #     cs = fitted_a / (0.9 * K0)
            #     cv = (0.9 * K0 * last_time * cs)**0.5
            #     absolute_end_time = current_start_time + last_time
            #     cumulative_dpa = np.interp(absolute_end_time, curr[2], curr[3])
            # else:
            #     positive_mask = x_fit > 0
            #     if np.sum(positive_mask) > 0:
            #         last_time = np.max(x_fit[positive_mask])
            #     else:
            #         last_time = x_fit[-1]
            #     K0 = dpa_cte * curr_end[hh]
            #     cs = np.nan
            #     cv = np.nan
            #     absolute_end_time = current_start_time + last_time
            #     cumulative_dpa = np.interp(absolute_end_time, curr[2], curr[3])
                
            positive_mask = x_fit > 0
            if np.sum(positive_mask) > 0:
                last_time = np.max(x_fit[positive_mask])
            else:
                last_time = x_fit[-1]   
            K0 = dpa_cte * curr_end[hh]
            cs = fitted_a / (0.9 * K0)
            cv = (0.9 * K0 * last_time * cs)**0.5
            absolute_end_time = current_start_time + last_time
            cumulative_dpa = np.interp(absolute_end_time, curr[2], curr[3])
            
            vac_results[0].append(fitted_a)
            vac_results[1].append(r_squared)
            vac_results[2].append(last_time)
            vac_results[3].append(K0)
            vac_results[4].append(cumulative_dpa)
            vac_results[5].append(cs)
            vac_results[6].append(cv)


            np.save(save_dir + sample_ID +'/'+sample_ID +'_vac1.npy', vac_results)
            np.save('/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/Codes/'+sample_ID+'_vac1.npy', vac_results)
            
            plt.figure(figsize=(10, 6))
            # Plot data with different colors for before/after current
            before_mask = x_data < 0
            after_mask = x_data >= 0
            
            plt.scatter(x_data[before_mask], y_data[before_mask], 
                        label='Beam off data', alpha=0.6, color='gray', s=40)
            plt.scatter(x_data[after_mask], y_data[after_mask], 
                       label='Beam on data (damaged)', alpha=0.6, color='#008080')
            # Plot vertical line at current start
            # plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, 
            #            label='Current start', linewidth=2)
            
            # Plot the fit, just from values greater than zero current
            x_plot = np.linspace(0, x_fit.max(), 200)
            y_plot = model_function(x_plot, fitted_a, fitted_b)
            fit_label = (
                r"$f_{\mathrm{SAW}} = f_{\mathrm{SAW_0}}\sqrt{1 - a\sqrt{x t}}$" "\n"
                fr"$f_{{\mathrm{{SAW_0}}}} = {fitted_b:.3e}\ \mathrm{{Hz}}$" "\n"
                fr"$x = {fitted_a:.3e}\ \mathrm{{s}}^{{-1}}$ " "\n"
                fr"$R^2 = {r_squared:.3f}$")
            plt.plot(x_plot, y_plot, color='#FF4500', linewidth=2.5, label=fit_label)
            plt.xlabel('Time since current start [s]', fontsize=14)
            plt.ylabel('SAW Frequency [Hz]', fontsize=14) 
            plt.title(f'Vacancy creation fit - Beam ON at {current_start_time:.1f}s', fontsize=16)
            plt.legend()
            plt.ticklabel_format(style='plain', axis='x') 
            plt.grid(True, alpha=0.3)
            plt.savefig(save_dir + sample_ID +'/'+sample_ID +f'_Drop_fit_t_{current_start_time:.1f}.png', dpi=600)
            plt.show()
            
        except Exception as e:
            print(f"Fit processing failed: {e}")

    else:
        print(f"Not enough data points around current start at {current_start_time}s")



# Create a formatter for exponential notation
from matplotlib.ticker import ScalarFormatter
# formatter = ScalarFormatter(useMathText=True)
# formatter.set_scientific(True)
# formatter.set_powerlimits((-2, 2))  # Adjust these limits as needed

plt.figure(figsize=(12,5))

ax1 = plt.subplot(1, 2, 1)
scatter1 = ax1.scatter(vac_results[4], vac_results[5], c=vac_results[1], 
                      cmap='rainbow_r', alpha=0.8, s=60, vmin=0, vmax=1)
plt.colorbar(scatter1, label='R² value')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Use ax1 instead of plt
ax1.set_xlabel('DPA')
ax1.set_ylabel(r'$\mathregular{C_s}$', rotation=0, labelpad=15)
ax1.set_title(r'$\mathregular{C_s}$ vs. DPA')
# plt.yscale('log') 
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(1, 2, 2)
scatter2 = ax2.scatter(vac_results[4], vac_results[6], c=vac_results[1], 
                      cmap='rainbow_r', alpha=0.8, s=60, vmin=0, vmax=1)
plt.colorbar(scatter2, label='R² value')
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  # Use ax2 instead of plt
ax2.set_xlabel('DPA')
ax2.set_ylabel(r'$\mathregular{C_v}$', rotation=0, labelpad=20)
ax2.set_title(r'$\mathregular{C_v}$ created vs. DPA')
# plt.yscale('log') 
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir + sample_ID +'/'+sample_ID +'_Cs_Cv_color_R2.png', dpi=600)
plt.show()


cum_cv=0
############     Beam OFF fitting      ############
vac_results2=[[],[],[],[],[],[],[], []]
# y1_start=y1_start[0:5]+y1_start[6::]# This just apply to 044_18 as the beam was accidentally off
for ll in range(1,len(y1_start)):  # for each current start time
    current_off_time = y1_start[ll] # Use the exact time when current becomes zero 
    fit_start_time = current_off_time - 20  # Include points BEFORE current goes to zero to get proper initial value
    cumulative_dpa = np.interp(current_off_time, curr[2], curr[3])

    window = 120
    
    # Find data points including the transition (before AND after current goes to zero)
    mask = (np.array(tgs_data[6]) >= fit_start_time) & (np.array(tgs_data[6]) <= current_off_time + window)
    
    if np.sum(mask) > 3:
        x_data = np.array(tgs_data[6])[mask] - current_off_time # Time since current goes to zero 
        y_data = sawfit1_sawf[mask]
        
        print(f"\n=== Fitting exponential recovery for current OFF at {current_off_time}s ===")
        print(f"Data points: {len(x_data)}")
        print(f"x range: {x_data.min():.1f} to {x_data.max():.1f}s")
        print(f"y range: {y_data.min():.3e} to {y_data.max():.3e}Hz")
        
        pr = 0.34  # Poisson's ratio value for Cu
        c = (3*(9+5*pr)*(1-pr))/(2*(7-5*pr))
        
        # Calculate the average of pre-current data
        pre_current_mask = x_data <= 0
        post_current_mask = x_data > 0
        if np.sum(pre_current_mask) > 0 and np.abs(y_data[pre_current_mask][-1]-y_data[pre_current_mask][-2])<np.abs(y_data[pre_current_mask][-1]-y_data[post_current_mask][0]):
            pre_current_avg = np.mean(y_data[pre_current_mask])
            print(f"Pre-current average SAWf (t<=0): {pre_current_avg:.4e}Hz")
        else:
            pre_current_avg = y_data[pre_current_mask][-1] if len(y_data) > 0 else np.mean(y_data)
            print(f"Using first point as pre-current average: {pre_current_avg:.4e}Hz")
        
        # Model function with forced first value
        def model_function(x, a, b):
            """
            x: time since beam off [s]
            a: Kvs * Cs [s⁻¹] 
            b: undamaged SAW frequency [Hz]
            c0 is calculated from the first value condition
            """
            # Calculation of c0 from the condition: pre_current_avg = b * sqrt(1 - c * c0)
            c0 = (1 - (pre_current_avg / b)**2) / c
            c0 = max(c0, 1e-12)  # Ensure positive
            result = np.zeros_like(x)
            mask_before = x <= 0
            mask_after = x > 0
            # Before current off: constant at pre_current_avg
            result[mask_before] = pre_current_avg
            # After current off: exponential recovery starting from pre_current_avg
            x_after = x[mask_after]
            vacancy_conc = c0 * np.exp(-a * x_after)
            inner_term = 1 - c * vacancy_conc
            inner_term = np.maximum(inner_term, 1e-10)
            result[mask_after] = b * np.sqrt(inner_term)
            
            return result

        x_fit = x_data
        y_fit = y_data
        
        try:
            # Get maximum observed value for plotting bounds
            max_observed = np.max(y_fit)
            print(f"Maximum observed SAWf: {max_observed:.4e}Hz")

            b_guess = max(max_observed * 1.01, pre_current_avg * 1.02)
            a_guess = 0.01  # Reasonable starting point
            print(f"Initial b guess: {b_guess:.4e}Hz")
            print(f"Initial a guess: {a_guess:.3e}")
            
            # Fit ONLY a and b parameters (c0 is determined by first value condition)
            try:
                params, covariance = curve_fit(model_function, x_fit, y_fit, 
                                             p0=[a_guess, b_guess], 
                                             bounds=([1e-10, pre_current_avg], 
                                                    [10, max_observed * 1.2]), 
                                             maxfev=10000)
                fitted_a, fitted_b = params
                
                # Calculate fitted_c0 from the first value condition
                fitted_c0 = (1 - (pre_current_avg / fitted_b)**2) / c
                fitted_c0 = max(fitted_c0, 1e-12)
                
                print(f"Fit successful: a={fitted_a:.3e}, b={fitted_b:.3e}, c0={fitted_c0:.3e}")
                
            except Exception as e:
                print(f"Fit failed: {e}, using initial guesses")
                fitted_a, fitted_b = a_guess, b_guess
                fitted_c0 = (1 - (pre_current_avg / fitted_b)**2) / c
            
            # Calculate predictions and R²
            y_pred = model_function(x_fit, fitted_a, fitted_b)
            residuals = y_fit - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            print(f"Final results: a={fitted_a:.3e}, c0={fitted_c0:.3e}, b={fitted_b:.3e}, R²={r_squared:.3f}")
            
            # Verify first value condition
            first_value = model_function(np.array([0.0]), fitted_a, fitted_b)[0]  # Value at t=0+
            print(f"First value verification: target={pre_current_avg:.4e}, model={first_value:.4e}, error={abs(pre_current_avg - first_value):.4e}")            
            print(ll)         
            print(ll-1)
            print(fitted_c0)
            print(vac_results[6][ll-1])
            
            if vac_results[1][ll-1] < 0.1:
                dcv = 0
                pre_cs = vac_results[5][ll-2]
                pre_cv = fitted_c0
                print("fitted_c0 used for pre_cv as beam-on fitting R^2<0.1")
            else:
                dcv = vac_results[6][ll-1] - cv
                pre_cs = vac_results[5][ll-1]
                pre_cv = vac_results[6][ll-1]          
            print(pre_cv)
            Kvs = fitted_a / pre_cs
            cv = pre_cv * np.exp( -fitted_a * last_time)
            cum_cv = cum_cv + dcv
            absolute_end_time = current_off_time + last_time
            cumulative_dpa = np.interp(absolute_end_time, curr[2], curr[3])
            
            vac_results2[0].append(fitted_a)
            vac_results2[1].append(r_squared)
            vac_results2[2].append(last_time)
            vac_results2[3].append(Kvs)
            vac_results2[4].append(cumulative_dpa)
            vac_results2[5].append(cv)
            vac_results2[6].append(dcv)
            vac_results2[7].append(cum_cv)
            
            
            np.save(save_dir + sample_ID +'/'+sample_ID +'_vac2.npy', vac_results2)
            np.save('/Users/ebotica/Library/CloudStorage/OneDrive-MassachusettsInstituteofTechnology/Desktop/MIT/Short Lab/Codes/'+sample_ID+'_vac2.npy', vac_results2)

            plt.figure(figsize=(10, 6))
            
            # Plot data with different colors for before/after current off
            before_mask = x_data < 0
            after_mask = x_data >= 0
            
            plt.scatter(x_data[before_mask], y_data[before_mask], 
                        label='Beam on data', alpha=0.6, color='gray', s=50)
            plt.scatter(x_data[after_mask], y_data[after_mask], 
                       label='Beam off data (recovery)', alpha=0.6, color='#008080', s=50)

            # # Plot vertical line at current off
            # plt.axvline(x=0, color='black', linestyle='--', alpha=0.7, 
            #            label='Beam off starts', linewidth=2)
            
            # # Mark the first value point
            # plt.plot(0, pre_current_avg, 'o', markersize=8, color='green', 
            #        label=f'First value: {pre_current_avg:.4e} Hz')
            
            # Plot the fit
            x_plot_fine = np.linspace(0, x_fit.max(), 300)
            y_plot_fine = model_function(x_plot_fine, fitted_a, fitted_b)
            fit_label = (
                r"$f_{\mathrm{SAW}} = f_{\mathrm{SAW}_0}\,\sqrt{1 - a\,(C_0 e^{-x t})}$" "\n"
                fr"$f_{{\mathrm{{SAW}}_0}} = {fitted_b:.3e}\ \mathrm{{Hz}}$" "\n"
                fr"$C_0 = {fitted_c0:.3e}$" "\n"
                fr"$x= {fitted_a:.3e}\ \mathrm{{s}}^{{-1}}$" "\n"
                fr"$R^2 = {r_squared:.3f}$")
            plt.plot(x_plot_fine, y_plot_fine,color='#FF4500', linewidth=3,label=fit_label)
            plt.xlabel('Time since beam off [s]', fontsize=14)
            plt.ylabel('SAW Frequency [Hz]', fontsize=14)
            plt.legend()
            plt.ticklabel_format(style='plain', axis='x') 
            plt.title(f'Vacancy annihilation fit - Beam OFF at {current_off_time:.1f}s', fontsize=16)
            plt.grid(True, alpha=0.3)
            
            
            plt.tight_layout()
            plt.savefig(save_dir + sample_ID +'/'+sample_ID +f'_Recovery_fit_t_{current_off_time:.1f}.png', dpi=600)
            plt.show()
            
        except Exception as e:
            print(f"Fit processing failed: {e}")

    else:
        print(f"Not enough data points around current off at {current_off_time}s")
        



plt.figure(figsize=(17,5))

ax1 = plt.subplot(1, 3, 1)
scatter1 = ax1.scatter(vac_results2[4], vac_results2[5], c=vac_results2[1], 
                      cmap='rainbow_r', alpha=0.8, s=60, vmin=0, vmax=1)
plt.colorbar(scatter1, label='R² value')
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.set_xlabel('DPA')
ax1.set_ylabel(r'$\mathregular{C_v}$', rotation=0, labelpad=15)
# plt.yscale('log') 
ax1.set_title(r'$\mathregular{C_v}$ annihilated vs. DPA')
# ax1.set_ylim(0.00001,0.001)
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(1, 3, 2)
scatter2 = ax2.scatter(vac_results2[4], vac_results2[6], color='#008080', alpha=0.6, s=50)
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.set_xlabel('DPA')
ax2.set_ylabel(r'$\mathregular{ΔC_v}$', rotation=0, labelpad=20)
ax2.set_title(r'$\mathregular{ΔC_v}$ vs. DPA')
# ax2.set_ylim(0,None)
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(1, 3, 3)
scatter3 = ax3.scatter(vac_results2[4], vac_results2[7], color='#008080', alpha=0.6, s=50)
ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax3.set_xlabel('DPA')
ax3.set_ylabel(r'$\mathregular{Cumulative ΔC_v}$', labelpad=20)
ax3.set_title(r'$\mathregular{Cumulative ΔC_v}$ vs. DPA')
# ax3.set_ylim(0,None)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir + sample_ID +'/'+sample_ID +'_CumulativeCv_DPA.png', dpi=600)
plt.show()


# plt.figure(figsize=(17,5))

# plt.subplot(1, 3, 1)
# scatter1 = plt.scatter(vac_results2[4], vac_results2[5], c=vac_results2[1], 
#                       cmap='rainbow_r', alpha=0.8, s=60, vmin=0, vmax=1)
# plt.colorbar(scatter1, label='R² value')
# plt.xlabel('DPA')
# plt.ylabel(r'$\mathregular{C_v}$', rotation=0, labelpad=15)
# plt.title(r'$\mathregular{C_v}$ annihilated vs. DPA')
# plt.grid(True, alpha=0.3)
# # plt.gca().xaxis.set_major_formatter(formatter)
# plt.yscale('log') 

# plt.subplot(1, 3, 2)
# scatter2 = plt.scatter(vac_results2[4], vac_results2[6], color='#008080', alpha=0.6, s=50)
# plt.xlabel('DPA')
# plt.ylabel(r'$\mathregular{ΔC_v}$', rotation=0, labelpad=20)
# plt.title(r'$\mathregular{ΔC_v}$ vs. DPA')
# plt.grid(True, alpha=0.3)
# # plt.gca().xaxis.set_major_formatter(formatter)
# plt.yscale('log')  

# plt.subplot(1, 3, 3)
# scatter2 = plt.scatter(vac_results2[4], vac_results2[8], color='#008080', alpha=0.6, s=50)
# plt.xlabel('DPA')
# plt.ylabel(r'$\mathregular{Cumulative ΔC_v}$', labelpad=20)
# plt.title(r'$\mathregular{Cumulative ΔC_v}$ vs. DPA')
# plt.grid(True, alpha=0.3)
# # plt.gca().xaxis.set_major_formatter(formatter)
# plt.yscale('log')  

# plt.tight_layout()
# plt.savefig(save_dir + sample_ID +'/'+sample_ID +'_CumulativeCv_DPA_log.png', dpi=600)
# plt.show()


