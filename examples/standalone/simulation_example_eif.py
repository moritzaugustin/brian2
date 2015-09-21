'''
Created on May 13, 2014
Last updated on February 02, 2015

@author: Moritz Augustin
'''

from brian2 import *
from matplotlib.pyplot import *
from time import time

# simulation parameters
standalone = True # toggle
N_neurons = 2000 # more neurons will make the histogram smoother
runtime = 200*ms
timestep = 0.01*ms
integrator = 'euler' #'euler' # 'milstein'

# input params
mu = 0*mV/msecond
sigma = 6.5*mV/msecond**.5

# neuron parameters
V_r = -70*mV # reset
V_s = -40*mV # spike threshold
C = 200*pF # membrane capacity
E_L = -65*mV # leak potential
g_L = 10*nS # leak conductance
delta_T = 1.5*mV # sodium steepness
V_T = -50*mV # sodium threshold
tau_ref = 2*ms 
lower_bound = True # if True: neurons which fluctuate/drive below the lb get reflected
V_lb = -100*mV # lower bound

# synaptic parameters
# TODO

# analysis parameters
N_raster = min(500, N_neurons)
N_traces = 3
V_plot0 = V_lb #V_r-10*mV # where to put the lower plotting limit
binsize = 1*ms # for the spike rate histogram

# extra title for point wise simulations
extra_title = 'method={integrator}, dt={timestep}, mu={mu} mV/ms, sigma={sigma} mV/ms**2'.format(
                        integrator=integrator, timestep=timestep,
                        mu=mu/(mV/msecond), sigma=sigma/(mV/msecond**.5))
print(extra_title)


warnbelow_dt = .01*ms
if timestep > warnbelow_dt:
    print('attention: please use timesteps <= {warnbelow_dt} since stochastic eq.'
          .format(**globals()))

if standalone:
    set_device('cuda_standalone')
    build_dir = 'eif_cuda'

defaultclock.dt = timestep

G = NeuronGroup(N_neurons, 
    '''dV/dt = 1/C * (g_L*(E_L-V) + g_L*delta_T*exp((V-V_T)/delta_T))  
               + mu + sigma*xi : volt (unless refractory)''', 
    threshold='V>V_s', reset='V=V_r', refractory='tau_ref', method=integrator)

Gstat = NeuronGroup(1, '''
    V_mean : volt
    V_var : volt**2
    ''')

Sstat = Synapses(G, Gstat, '''
    V_mean_post = V_pre/N_pre : volt (summed)
    V_var_post = (V_pre - V_mean)**2 / N_pre : volt**2 (summed) 
    ''')
Sstat.connect(True)

# initialization (of neurongroup and 'statistics' neurongroup)
G.V = 'E_L + (V_T-V_r)/2*rand()' # reset membrane voltage initialization

SpMon = SpikeMonitor(G[:N_raster], record=True)
StMon = StateMonitor(G, ['V'], record=range(N_traces))
StMonStat = StateMonitor(Gstat, ['V_mean', 'V_var'], record=True)
PRMon = PopulationRateMonitor(G)

netlist = [G, SpMon, StMon, PRMon, Gstat, Sstat, StMonStat]

# implement lower bound to ensure v is lower bounded by V_lb
if lower_bound:
    # version 1: reflect values of v below V_lb
    OperationLB = G.custom_operation('V = V + 2*(clip(V, V_lb, inf) - V)', when='resets')
    
    # version 2: set of v below V_lb to V_lb independent of their distance
#     OperationLB = G.custom_operation('v = clip(v, V_lb, inf)', when='resets')
    
    netlist.append(OperationLB)
    
#   # version3: Brian1 implementation (or for Python version of Brian2):
#         @network_operation
#         def if_lower_bound():
#             G.v[G.v < V_lb] = V_lb
    

Net = Network(netlist)

t_start = time()

Net.run(runtime, report="text")

#run(runtime)

if standalone:
    device.build(directory=build_dir, compile=True, run=True, debug=False)

t_duration = time()-t_start
comp_str = ''
if standalone:
    comp_str = ' & compilation'
print('time for simulation{comp_str}: {t_duration}'
      .format(**globals()))

# plotting

# if plot_state_hist:
#     figure()
#     bins = 20
#     hist(StMon.v[:,[0,-1]]/mV, bins=bins, normed=True, 
#          label=['initialization (t=0)', 
#                 'simulation end (t={tmax}ms)'.format(tmax=runtime/ms)])
#     legend()
#     xlim(0,1)
#     title('distribution of trial states')
#     xlabel('membrane voltage x (mV)')
#     ylabel('membrane voltage density (1/mV)')

# show()
# exit()

figure()
suptitle(extra_title)
subplot(311)
lb_string = '' if not lower_bound else ' (with lower bound {V_lb})'.format(**globals())
title('membrane voltage dynamics ({N_traces} neurons + population)'.format(**globals())
      +lb_string)

for k in range(N_traces):
    spkTimes_k = SpMon.it[1][SpMon.it[0] == k]
    plot(spkTimes_k/ms, (V_s/mV)*ones_like(spkTimes_k), 'o', color='gray', markersize=6)
    plot(StMon.t/ms, StMon.V[k]/mV)

V_mean_mV = StMonStat.V_mean[0]/mV
V_std_mV = np.sqrt(StMonStat.V_var[0]/(mV**2))
plot(StMonStat.t/ms, V_mean_mV, '-', color='gray')
plot(StMonStat.t/ms, V_mean_mV + V_std_mV, '--', color='gray')
plot(StMonStat.t/ms, V_mean_mV - V_std_mV, '--', color='gray')
xlabel('time (ms)')
xlim(0, runtime/ms)
ylabel('membrane voltage V (mV)')
ylim((V_plot0/mV, V_s/mV))
# inset try:
# a = axes([0.5, 0.5, 0.2, .3], axisbg='y')
# hist(StMon.v[:,0]/mV, bins=bins)
# setp(a)

subplot(312)
title('spike rasterplot of the first {N_raster} neurons'.format(**globals()))
i, t = SpMon.it
plot(t/ms, i, 'o', markersize=2)
xlabel('time (ms)')
xlim(0, runtime/ms)
ylabel('trial')

subplot(313)
# plot(PRMon.t/ms, PRMon.rate/Hz)
bins = int(binsize/defaultclock.dt)
title('trial-averaged spike rate (binsize={bs} ms == {bins} timesteps)'
      .format(bs=binsize/ms, bins=bins))
t = (PRMon.t/ms)[::bins]
r = np.array([(PRMon.rate/Hz)[k:min(k+bins,len(PRMon.rate/Hz))].sum()/float(bins) 
              for k in range(0, len(PRMon.rate/Hz), bins)])
# plot(PRMon.t/ms, PRMon.rate/Hz)
bar(t, r, color='black', edgecolor='black')
plot(t, 5*ones_like(t), color='gray')
xlabel('time (ms)')
xlim(0, runtime/ms)
ylabel('spike rate (Hz)')

show()
