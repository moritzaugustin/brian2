'''
Created on Jun 20, 2014

@author: augustin
'''

from brian2 import *
from pylab import *
import time


standalone = True
build_dir = 'adaptation_oscillations_standalone_cuda'

N_neurons = 4000
sparsity = 0.05 # each neuron receives approx. N_neurons*sparsity connections => 0: uncoupled network
runtime = 1000*ms


if standalone:
    set_device('cpp_standalone')

# model parameters (excitatory sparsely coupled adaptive leaky integrate-and-fire neurons)
tau_v = 10 * ms # membrane time const
tau_w = 200 * ms # adaptation time const
v_t = 1 * mV # threshold voltage
v_r = 0 * mV # reset voltage
dw = 0.1 * mV # spike-triggered adaptation increment
Tref = 2.5 * ms # refractory period
if sparsity>0:
    syn_weight = 1.03/(N_neurons*sparsity) * mV # currently: constant synaptic weights
    syn_delay = 2 * ms  # currently: constant synaptic delays
# input noise:
input_mean = 0.14 * mV/ms
input_std = 0.07 * mV/ms**.5

# brian neuron model specification
eqs_neurons = '''
dv/dt = (-v-w)/tau_v + input_mean + input_std*xi : volt (unless refractory)
dw/dt = -w/tau_w : volt (unless refractory)
'''
reset_neurons = '''
v = v_r
w = w+dw
'''

neurons = NeuronGroup(N_neurons, 
                      eqs_neurons, 
                      reset=reset_neurons,
                      threshold='v > v_t', 
                      refractory='Tref')

# random initialization of neuron state values
neurons.v = 'rand()*v_t' 
neurons.w = 'rand()*10*dw'

# brian synaptic model specification
if sparsity>0:
    synapses = Synapses(neurons, neurons, 'c: volt', pre='v += c')
    synapses.connect('i!=j', p=sparsity)
    synapses.c[:] = 'syn_weight' 
    synapses.delay[:] = 'syn_delay' 
    # BUG in brian2?: 
    # distributed delays do not work + rand()*0.1*ms'# len(synapses))*defaultclock.dt*1   #'syn_delay'
#     print(defaultclock.dt)
#     print(synapses.delay[:].mean())
#     print(synapses.delay[:].std())
#     print(synapses.delay[:]/ms)
    syn_str = ' with {nsyn} synapses!'.format(nsyn=len(synapses)) if not standalone else ''
    print('simulating recurrent network'+syn_str)
else:
    print('simulating feed forward network (no recurrency)!')

# monitors
stateMon = StateMonitor(neurons, ['v', 'w'], [0]) #record=True) #[0])
spikeMon = SpikeMonitor(neurons)
rateMon = PopulationRateMonitor(neurons)

t_start = time.time()

#run(runtime)

print('run() took {s}s'.format(s=time.time()-t_start))

if standalone:
    #print('starting standalone build and simulation')
    #device.build(project_dir=build_dir, 
    #             compile_project=True, run_project=True, debug=True)
    
    def result_array(arrname, dtype='float64'):
        result_dir = build_dir + '/results'
        return fromfile(result_dir+'/'+arrname, dtype=dtype)

    rateMon_t = result_array('_dynamic_array_ratemonitor_t') * second
    rateMon_rate = result_array('_dynamic_array_ratemonitor_rate') * Hz
    spikeMon_i = result_array('_dynamic_array_spikemonitor_i', dtype='int32')
    spikeMon_t = result_array('_dynamic_array_spikemonitor_t') * second
    stateMon_t = result_array('_dynamic_array_statemonitor_t') * second
    stateMon_v = result_array('_dynamic_array_statemonitor__recorded_v') * volt
    stateMon_w = result_array('_dynamic_array_statemonitor__recorded_w') * volt   
else:
    rateMon_t = rateMon.t
    rateMon_rate = rateMon.rate
    spikeMon_i, spikeMon_t = spikeMon.it
    stateMon_t = stateMon.t
    stateMon_v = stateMon[0].v.T
    stateMon_w = stateMon[0].w.T

# plotting

matplotlib.rcParams['legend.fontsize'] = 'small'
matplotlib.rcParams['axes.titlesize'] = 'small'
matplotlib.rcParams['axes.labelsize'] = 'small'
matplotlib.rcParams['xtick.labelsize'] = 'small'
matplotlib.rcParams['ytick.labelsize'] = 'small'

figure()
subplot(411)
binsize = 10
title('network-averaged spike rate (binsize={bs} ms)'.format(bs=binsize*defaultclock.dt/ms))
rateHist_t = (rateMon_t/ms)[::binsize]
rateHist_rate = np.array([(rateMon_rate/Hz)[k:k+binsize].sum()/binsize for k in range(0, len(rateMon_rate/Hz), 
                                                                          binsize)])
bar(rateHist_t, rateHist_rate, color='black', edgecolor='black')
# xlabel('time (ms)')
xlim(0, runtime/ms)
ylabel('spike rate (Hz)')

subplot(412)
title('spike rasterplot of the network'.format(N=N_neurons))
plot(spikeMon_t/ms, spikeMon_i, 'o', markersize=0.2)
# xlabel('time (ms)')
xlim(0, runtime/ms)
ylabel('neuron')

subplot(413)
title('membrane voltage dynamics of the first neuron')
plot(stateMon_t/ms, stateMon_v/mV)
# xlabel('time (ms)')
xlim(0, runtime/ms)
ylabel('membrane voltage v (mV)')
ylim((-0.1, 1.1))

subplot(414)
title('adaptation  dynamics of the first neuron')
plot(stateMon_t/ms, stateMon_w/mV)
xlabel('time (ms)')
xlim(0, runtime/ms)
ylabel('membrane voltage w (mV)')
# POPULATION AVERAGED ADAPTATION:
# plot(stateMon.t/ms, (stateMon.w/mV).mean(axis=0))

show()
