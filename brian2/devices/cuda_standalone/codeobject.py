'''
Module implementing the CUDA "standalone" `CodeObject`
'''
from brian2.codegen.codeobject import CodeObject
from brian2.codegen.targets import codegen_targets
from brian2.codegen.templates import Templater
from brian2.codegen.generators.cuda_generator import (CUDACodeGenerator,
                                                     c_data_type)
from brian2.devices.cpp_standalone import CPPStandaloneCodeObject
from brian2.devices.device import get_device
from brian2.core.preferences import prefs

__all__ = ['CUDAStandaloneCodeObject']


class CUDAStandaloneCodeObject(CPPStandaloneCodeObject):
    '''
    CUDA standalone code object
    
    The ``code`` should be a `~brian2.codegen.templates.MultiTemplate`
    object with two macros defined, ``main`` (for the main loop code) and
    ``support_code`` for any support code (e.g. function definitions).
    '''
    templater = Templater('brian2.devices.cuda_standalone',
                          env_globals={'c_data_type': c_data_type})
    generator_class = CUDACodeGenerator
    serializing_form = "syn"
    runs_every_tick = True  #default True, set False in generate_main_source
    rand_calls = 0
    randn_calls = 0

    def __call__(self, **kwds):
        return self.run()

    def run(self):
        get_device().main_queue.append(('run_code_object', (self,)))

codegen_targets.add(CUDAStandaloneCodeObject)
