#!/usr/bin/env python
# coding: latin-1
"""
NMDA model from synapses/nonlinear.py

incorporated as small additional excitation in the example COBAHH
"""

from brian2 import *

set_device('cpp_standalone')

# Parameters
N = 400 #8000
use_nmda = True
only_syndynamics = True # removes the pre spike dependency from the NMDA synapses and replaces them by a random process

area = 20000*umetre**2
Cm = (1*ufarad*cm**-2) * area
gl = (5e-5*siemens*cm**-2) * area

El = -60*mV
EK = -90*mV
ENa = 50*mV
g_na = (100*msiemens*cm**-2) * area
g_kd = (30*msiemens*cm**-2) * area
VT = -63*mV
# Time constants
taue = 5*ms
taui = 10*ms
# Reversal potentials
Ee = 0*mV
Ei = -80*mV
we = 6*nS  # excitatory synaptic weight
wi = 67*nS  # inhibitory synaptic weight
# NMDA params   
a = 1 / (10*ms)
b = 1 / (10*ms)
c = 1 / (10*ms)
wNMDA = 0.1*nS
gNMDAsingle_max = 0.5*nS
# only for only_syndynamics
syn_noise_spikes_per_second = 20
syn_noise = syn_noise_spikes_per_second*wNMDA/second**0.5

# The model
eqs = Equations('''
dv/dt = (gl*(El-v)+(ge+gnmda)*(Ee-v)+gi*(Ei-v)-
         g_na*(m*m*m)*h*(v-ENa)-
         g_kd*(n*n*n*n)*(v-EK))/Cm : volt
dm/dt = alpha_m*(1-m)-beta_m*m : 1
dn/dt = alpha_n*(1-n)-beta_n*n : 1
dh/dt = alpha_h*(1-h)-beta_h*h : 1
dge/dt = -ge*(1./taue) : siemens
dgi/dt = -gi*(1./taui) : siemens
gnmda : siemens
alpha_m = 0.32*(mV**-1)*(13*mV-v+VT)/
         (exp((13*mV-v+VT)/(4*mV))-1.)/ms : Hz
beta_m = 0.28*(mV**-1)*(v-VT-40*mV)/
        (exp((v-VT-40*mV)/(5*mV))-1)/ms : Hz
alpha_h = 0.128*exp((17*mV-v+VT)/(18*mV))/ms : Hz
beta_h = 4./(1+exp((40*mV-v+VT)/(5*mV)))/ms : Hz
alpha_n = 0.032*(mV**-1)*(15*mV-v+VT)/
         (exp((15*mV-v+VT)/(5*mV))-1.)/ms : Hz
beta_n = .5*exp((10*mV-v+VT)/(40*mV))/ms : Hz
''')

P = NeuronGroup(N, model=eqs, threshold='v>-20*mV', refractory=3*ms,
                method='exponential_euler')
N_exc = int(0.8*N)
Pe = P[:N_exc]
Pi = P[N_exc:]
Ce = Synapses(Pe, P, pre='ge+=we', connect='rand()<0.02')
Ci = Synapses(Pi, P, pre='gi+=wi', connect='rand()<0.02')

if use_nmda:
    print('warning: maybe wrong assumption of NMDA having the same reversal potential as AMPA')
    print('warning: the g_syn dynamics was maybe wrongly changed in order tohave physical units...')
    if not only_syndynamics:
        Cnmda = Synapses(Pe, P,
                '''# This variable could also be called g_syn to avoid confusion
                    dg_syn/dt = -a*g_syn+b*x*(1-g_syn/gNMDAsingle_max) : siemens
                    gnmda_post = g_syn : siemens (summed)
                    dx/dt=-c*x : siemens
                    w : siemens # synaptic weight
                ''', pre='x += wNMDA', 
                connect='rand()<0.02') # NMDA synapses
    else:
        Cnmda = Synapses(Pe, P,
        '''# This variable could also be called g_syn to avoid confusion
            dg_syn/dt = -a*g_syn+b*x*(1-g_syn/gNMDAsingle_max) : siemens
            gnmda_post = g_syn : siemens (summed)
            dx/dt=-c*x + syn_noise*xi : siemens
            w : siemens # synaptic weight
        ''', connect='rand()<0.02') # NMDA synapses

# Initialization
P.v = 'El + (randn() * 5 - 5)*mV'
P.ge = '(randn() * 1.5 + 4) * 10.*nS'
P.gi = '(randn() * 12 + 20) * 10.*nS'
P.gnmda = '0 * nS'

# Record a few traces
trace = StateMonitor(P, 'v', record=[1, 10, 100])
s_mon = SpikeMonitor(P)
run(1 * second, report='text')

device.build(directory='COBAHH_NMDA_cpp', compile=True, run=True)

figure()
subplot(2,1,1)
plot(trace.t/ms, trace[1].v/mV)
plot(trace.t/ms, trace[10].v/mV)
plot(trace.t/ms, trace[100].v/mV)
xlabel('t (ms)')
ylabel('v (mV)')
subplot(2,1,2)
plot(s_mon.t/ms, s_mon.i, '.k')
ylabel('neuron id')
xlabel('t (ms)')
show()
