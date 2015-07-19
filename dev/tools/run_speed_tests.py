from brian2 import *
from brian2.tests.features import *
from brian2.tests.features.base import *
from brian2.tests.features.speed import *

# Full testing
'''
res = run_speed_tests()
res.plot_all_tests()
show()
'''

# Quick testing
for i in range(0, 10):
    res = run_speed_tests(configurations=[#DefaultConfiguration,
                                          #WeaveConfiguration,
                                          #CythonConfiguration,
                                          CPPStandaloneConfiguration,
                                          CPPStandaloneConfigurationOpenMP,
                                          CUDAStandaloneConfiguration,
                                          CUDAStandaloneConfigurationDoubleSMs,
                                          CUDAStandaloneConfigurationFourSMs
                                          ],
                          speed_tests= [
                                       HHNeuronsOnly,
                                       CUBAFixedConnectivity,
                                       COBAHH,
                                       BrunelHakimModel,
                                       BrunelHakimModelWithDelay,
                                       STDPEventDriven,
                                       STDPNotEventDriven,
                                       #Vogels,
                                       #VogelsWithSynapticDynamic,
                                       AdaptationOscillation,
                                       VerySparseMediumRateSynapsesOnly,
                                       SparseMediumRateSynapsesOnly,
                                       DenseMediumRateSynapsesOnly,
                                       SparseLowRateSynapsesOnly,
                                       SparseHighRateSynapsesOnly,
                                       ],
                          n_slice=slice(i, i+1),
                          #n_slice=slice(None, -1),
                          run_twice=False,
                          verbose=False
                          )
    #res.plot_all_tests()

# Debug
# c = GeNNConfiguration()
# c.before_run()
# f = VerySparseMediumRateSynapsesOnly(1000000)
# f.run()
# c.after_run()
