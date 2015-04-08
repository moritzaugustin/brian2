'''
Module implementing the CUDA "standalone" device.
'''
import os
import shutil
import inspect
from collections import defaultdict

import numpy as np

from brian2.codegen.cpp_prefs import get_compiler_and_args
from brian2.core.clocks import defaultclock
from brian2.core.network import Network
from brian2.core.variables import *
from brian2.devices.device import all_devices
from brian2.synapses.synapses import Synapses
from brian2.utils.filetools import copy_directory, ensure_directory
from brian2.codegen.generators.cuda_generator import c_data_type
from brian2.utils.logger import get_logger
from brian2.units import second

from .codeobject import CUDAStandaloneCodeObject
from brian2.devices.cpp_standalone.device import CPPWriter, CPPStandaloneDevice, freeze
from brian2.monitors.statemonitor import StateMonitor


__all__ = []

logger = get_logger(__name__)

class CUDAWriter(CPPWriter):
    def __init__(self, project_dir):
        self.project_dir = project_dir
        self.source_files = []
        self.header_files = []
        
    def write(self, filename, contents):
        logger.debug('Writing file %s:\n%s' % (filename, contents))
        if filename.lower().endswith('.cu'):
            self.source_files.append(filename)
        if filename.lower().endswith('.cpp'):
            self.source_files.append(filename)
        elif filename.lower().endswith('.h'):
            self.header_files.append(filename)
        elif filename.endswith('.*'):
            self.write(filename[:-1]+'cu', contents.cu_file)
            self.write(filename[:-1]+'h', contents.h_file)
            return
        fullfilename = os.path.join(self.project_dir, filename)
        if os.path.exists(fullfilename):
            if open(fullfilename, 'r').read()==contents:
                return
        open(fullfilename, 'w').write(contents)


class CUDAStandaloneDevice(CPPStandaloneDevice):
    '''
    The `Device` used for CUDA standalone simulations.
    '''
    def code_object_class(self, codeobj_class=None):
        # Ignore the requested codeobj_class
        return CUDAStandaloneCodeObject

    def code_object(self, owner, name, abstract_code, variables, template_name,
                    variable_indices, codeobj_class=None, template_kwds=None,
                    override_conditional_write=None):
        codeobj = super(CUDAStandaloneDevice, self).code_object(owner, name, abstract_code, variables,
                                                               template_name, variable_indices,
                                                               codeobj_class=codeobj_class,
                                                               template_kwds=template_kwds,
                                                               override_conditional_write=override_conditional_write,
                                                               )
        return codeobj
    
    def check_OPENMP_compatible(self, nb_threads):
        if nb_threads > 0:
            raise NotImplementedError("Using OpenMP in an CUDA standalone project is not supported")
        
    def generate_objects_source(self, writer, arange_arrays, synapses, static_array_specs, networks):
        codeobj_with_rand = [co for co in self.code_objects.values() if co.runs_every_tick == True and co.rand_calls > 0]
        codeobj_with_randn = [co for co in self.code_objects.values() if co.runs_every_tick == True and co.randn_calls > 0]
        arr_tmp = CUDAStandaloneCodeObject.templater.objects(
                        None, None,
                        array_specs=self.arrays,
                        dynamic_array_specs=self.dynamic_arrays,
                        dynamic_array_2d_specs=self.dynamic_arrays_2d,
                        zero_arrays=self.zero_arrays,
                        arange_arrays=arange_arrays,
                        synapses=synapses,
                        clocks=self.clocks,
                        static_array_specs=static_array_specs,
                        networks=networks,
                        code_objects=self.code_objects.values(),
                        codeobj_with_rand=codeobj_with_rand,
                        codeobj_with_randn=codeobj_with_randn)
        writer.write('objects.*', arr_tmp)

    def generate_main_source(self, writer, main_includes):
        main_lines = []
        procedures = [('', main_lines)]
        runfuncs = {}
        for func, args in self.main_queue:
            if func=='run_code_object':
                codeobj, = args
                codeobj.runs_every_tick = False
                main_lines.append('_run_%s();' % codeobj.name)
            elif func=='run_network':
                net, netcode = args
                main_lines.extend(netcode)
            elif func=='set_by_array':
                arrayname, staticarrayname = args
                code = '''
                for(int i=0; i<_num_{staticarrayname}; i++)
                {{
                    {arrayname}[i] = {staticarrayname}[i];
                }}
                cudaMemcpy(dev{arrayname}, {arrayname}, sizeof({arrayname}[0])*_num_{arrayname}, cudaMemcpyHostToDevice);
                '''.format(arrayname=arrayname, staticarrayname=staticarrayname)
                main_lines.extend(code.split('\n'))
            elif func=='set_array_by_array':
                arrayname, staticarrayname_index, staticarrayname_value = args
                code = '''
                for(int i=0; i<_num_{staticarrayname_index}; i++)
                {{
                    {arrayname}[{staticarrayname_index}[i]] = {staticarrayname_value}[i];
                }}
                cudaMemcpy(dev{arrayname}, {arrayname}, sizeof({arrayname}[0])*_num_{arrayname}, cudaMemcpyHostToDevice);
                '''.format(arrayname=arrayname, staticarrayname_index=staticarrayname_index,
                           staticarrayname_value=staticarrayname_value)
                main_lines.extend(code.split('\n'))
            elif func=='insert_code':
                main_lines.append(args)
            elif func=='start_run_func':
                name, include_in_parent = args
                if include_in_parent:
                    main_lines.append('%s();' % name)
                main_lines = []
                procedures.append((name, main_lines))
            elif func=='end_run_func':
                name, include_in_parent = args
                name, main_lines = procedures.pop(-1)
                runfuncs[name] = main_lines
                name, main_lines = procedures[-1]
            else:
                raise NotImplementedError("Unknown main queue function type "+func)

        # generate the finalisations
        for codeobj in self.code_objects.itervalues():
            if hasattr(codeobj.code, 'main_finalise'):
                main_lines.append(codeobj.code.main_finalise)
                
        main_tmp = CUDAStandaloneCodeObject.templater.main(None, None,
                                                          main_lines=main_lines,
                                                          code_objects=self.code_objects.values(),
                                                          report_func=self.report_func,
                                                          dt=float(defaultclock.dt),
                                                          additional_headers=main_includes,
                                                          )
        writer.write('main.cu', main_tmp)
        
    def generate_codeobj_source(self, writer):
        #check how many random numbers are needed per step
        for code_object in self.code_objects.itervalues():
            num_occurences_rand = code_object.code.cu_file.count("_rand(")
            num_occurences_randn = code_object.code.cu_file.count("_randn(")
            if num_occurences_rand > 0:
                #first one is alway the definition, so subtract 1
                code_object.rand_calls = num_occurences_rand - 1
                for i in range(0, code_object.rand_calls):
                    if code_object.owner.N <> 0:
                        code_object.code.cu_file = code_object.code.cu_file.replace("_rand(_vectorisation_idx)", "_rand(_vectorisation_idx + " + str(i) + " * " + str(code_object.owner.N) + ")", 1)
                    else:
                        code_object.code.cu_file = code_object.code.cu_file.replace("_rand(_vectorisation_idx)", "_rand(_vectorisation_idx + " + str(i) + " * N)", 1)
            if num_occurences_randn > 0:
                #first one is alway the definition, so subtract 1
                code_object.randn_calls = num_occurences_randn - 1
                for i in range(0, code_object.randn_calls):
                    if code_object.owner.N <> 0:
                        code_object.code.cu_file = code_object.code.cu_file.replace("_randn(_vectorisation_idx)", "_randn(_vectorisation_idx + " + str(i) + " * " + str(code_object.owner.N) + ")", 1)
                    else:
                        code_object.code.cu_file = code_object.code.cu_file.replace("_randn(_vectorisation_idx)", "_randn(_vectorisation_idx + " + str(i) + " *N)", 1)

        code_object_defs = defaultdict(list)
        host_parameters = defaultdict(list)
        device_parameters = defaultdict(list)
        kernel_variables = defaultdict(list)
        # Generate data for non-constant values
        for codeobj in self.code_objects.itervalues():
            lines = []
            lines_1 = []
            lines_2 = []
            lines_3 = []
            additional_code = []
            number_elements = ""
            if hasattr(codeobj, 'owner') and hasattr(codeobj.owner, '_N') and codeobj.owner._N <> 0:
                number_elements = str(codeobj.owner._N)
            else:
                number_elements = "N"
            for k, v in codeobj.variables.iteritems():
                if k == "_python_randn" and codeobj.runs_every_tick == False and codeobj.template_name <> "synapses_create":
                    additional_code.append('''
                        //genenerate an array of random numbers on the device
                        float* dev_array_randn;
                        cudaMalloc((void**)&dev_array_randn, sizeof(float)*''' + number_elements + ''' * ''' + str(codeobj.randn_calls) + ''');
                        if(!dev_array_randn)
                        {
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof(float)*''' + number_elements + '''*''' + str(codeobj.randn_calls) + ''');
                        }
                        curandGenerator_t gen;
                        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
                        curandSetPseudoRandomGeneratorSeed(gen, time(0));
                        curandGenerateNormal(gen, dev_array_randn, ''' + number_elements + '''*''' + str(codeobj.randn_calls) + ''', 0, 1);''')
                    line = "float* _array_{name}_randn".format(name=codeobj.name)
                    lines_2.append(line)
                    lines_1.append("dev_array_randn")
                elif k == "_python_rand" and codeobj.runs_every_tick == False and codeobj.template_name <> "synapses_create":
                    additional_code.append('''
                        //genenerate an array of random numbers on the device
                        float* dev_array_rand;
                        cudaMalloc((void**)&dev_array_rand, sizeof(float)*''' + number_elements + '''*''' + str(codeobj.rand_calls) + ''');
                        if(!dev_array_rand)
                        {
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof(float)*''' + number_elements + '''*''' + str(codeobj.rand_calls) + ''');
                        }
                        curandGenerator_t gen;
                        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
                        curandSetPseudoRandomGeneratorSeed(gen, time(0));
                        curandGenerateUniform(gen, dev_array_rand, ''' + number_elements + '''*''' + str(codeobj.rand_calls) + ''');''')
                    line = "float* _array_{name}_rand".format(name=codeobj.name)
                    lines_2.append(line)
                    lines_1.append("dev_array_rand")
                elif isinstance(v, AttributeVariable):
                    # We assume all attributes are implemented as property-like methods
                    line = 'const {c_type} {varname} = {objname}.{attrname}();'
                    lines.append(line.format(c_type=c_data_type(v.dtype), varname=k, objname=v.obj.name,
                                             attrname=v.attribute)) 
                    lines_1.append(k)
                    line = "{c_type} par_{varname}"
                    lines_2.append(line.format(c_type=c_data_type(v.dtype), varname=k))
                    line = 'const {c_type} {varname} = par_{varname};'
                    lines_3.append(line.format(c_type=c_data_type(v.dtype), varname=k))
                elif isinstance(v, ArrayVariable):
                    try:
                        if isinstance(v, DynamicArrayVariable):
                            if v.dimensions == 1:
                                dyn_array_name = self.dynamic_arrays[v]
                                array_name = self.arrays[v]
                                line = '{c_type}* const {array_name} = thrust::raw_pointer_cast(&dev{dyn_array_name}[0]);'
                                line = line.format(c_type=c_data_type(v.dtype), array_name=array_name,
                                                   dyn_array_name=dyn_array_name)
                                lines.append(line)
                                line = 'const int _num{k} = dev{dyn_array_name}.size();'
                                line = line.format(k=k, dyn_array_name=dyn_array_name)
                                lines.append(line)
                                
                                lines_1.append(array_name)
                                lines_1.append("_num" + k)
                                
                                line = "{c_type}* par_{array_name}"
                                lines_2.append(line.format(c_type=c_data_type(v.dtype), array_name=array_name))
                                line = "int par_num_{array_name}"
                                lines_2.append(line.format(array_name=array_name))
                                
                                line = "{c_type}* _ptr{array_name} = par_{array_name};"
                                lines_3.append(line.format(c_type=c_data_type(v.dtype), array_name=array_name))
                                line = "const int _num{array_name} = par_num_{array_name};"
                                lines_3.append(line.format(array_name=array_name))
                        else:
                            lines_1.append("dev"+self.get_array_name(v))
                            lines_2.append("%s* par_%s" % (c_data_type(v.dtype), self.get_array_name(v)))
                            lines_3.append("%s* _ptr%s = par_%s;" % (c_data_type(v.dtype),  self.get_array_name(v), self.get_array_name(v)))

                            lines.append('const int _num%s = %s;' % (k, v.size))
                            lines_3.append('const int _num%s = %s;' % (k, v.size))
                    except TypeError:
                        pass
            for line in lines:
                # Sometimes an array is referred to by to different keys in our
                # dictionary -- make sure to never add a line twice
                if not line in code_object_defs[codeobj.name]:
                    code_object_defs[codeobj.name].append(line)
            for line in lines_1:
                if not line in host_parameters[codeobj.name]:
                    host_parameters[codeobj.name].append(line)
            for line in lines_2:
                if not line in device_parameters[codeobj.name]:
                    device_parameters[codeobj.name].append(line)
            for line in lines_3:
                if not line in kernel_variables[codeobj.name]:
                    kernel_variables[codeobj.name].append(line)
                    
            for line in additional_code:
                code_object_defs[codeobj.name].append(line)
        
        # Generate the code objects
        for codeobj in self.code_objects.itervalues():
            ns = codeobj.variables
            # TODO: fix these freeze/CONSTANTS hacks somehow - they work but not elegant.
            code = freeze(codeobj.code.cu_file, ns)
            if isinstance(codeobj.owner, StateMonitor):
                for varname, var in codeobj.owner.recorded_variables.iteritems():
                    record_var = codeobj.owner.variables[varname]
                    _data = self.get_array_name(record_var, access_data=False)
                    if record_var in self.dynamic_arrays:
                        code = code.replace('%DATA_ARR%', 'thrust::raw_pointer_cast(&dev%s[0])' % (_data), 1)
                    else:
                        code = code.replace('%DATA_ARR%', 'dev%s' % (_data), 1)
            code = code.replace('%CONSTANTS%', '\n\t\t'.join(code_object_defs[codeobj.name]))
            code = code.replace('%HOST_PARAMETERS%', ',\n\t\t\t'.join(host_parameters[codeobj.name]))
            code = code.replace('%DEVICE_PARAMETERS%', ',\n\t'.join(device_parameters[codeobj.name]))
            code = code.replace('%KERNEL_VARIABLES%', '\n\t'.join(kernel_variables[codeobj.name]))
            code = code.replace('%CODEOBJ_NAME%', codeobj.name)
            code = '#include "objects.h"\n'+code
            
            writer.write('code_objects/'+codeobj.name+'.cu', code)
            writer.write('code_objects/'+codeobj.name+'.h', codeobj.code.h_file)
            
    def generate_rand_source(self, writer):
        codeobj_with_rand = [co for co in self.code_objects.values() if co.runs_every_tick == True and co.rand_calls > 0]
        codeobj_with_randn = [co for co in self.code_objects.values() if co.runs_every_tick == True and co.randn_calls > 0]
        rand_tmp = CUDAStandaloneCodeObject.templater.rand(None, None,
                                                           code_objects=self.code_objects.values(),
                                                           codeobj_with_rand=codeobj_with_rand,
                                                           codeobj_with_randn=codeobj_with_randn)
        writer.write('rand.*', rand_tmp)
    
    def copy_source_files(self, writer, directory):
        # Copy the brianlibdirectory
        brianlib_dir = os.path.join(os.path.split(inspect.getsourcefile(CUDAStandaloneCodeObject))[0],
                                    'brianlib')
        brianlib_files = copy_directory(brianlib_dir, os.path.join(directory, 'brianlib'))
        for file in brianlib_files:
            if file.lower().endswith('.cpp'):
                writer.source_files.append('brianlib/'+file)
            if file.lower().endswith('.cu'):
                writer.source_files.append('brianlib/'+file)
            elif file.lower().endswith('.h'):
                writer.header_files.append('brianlib/'+file)
        shutil.copy2(os.path.join(os.path.split(inspect.getsourcefile(Synapses))[0], 'stdint_compat.h'),
                     os.path.join(directory, 'brianlib', 'stdint_compat.h'))

    def generate_network_source(self, writer, compiler):
        if compiler=='msvc':
            std_move = 'std::move'
        else:
            std_move = ''
        network_tmp = CUDAStandaloneCodeObject.templater.network(None, None,
                                                             std_move=std_move)
        writer.write('network.*', network_tmp)
        
    def generate_synapses_classes_source(self, writer):
        synapses_classes_tmp = CUDAStandaloneCodeObject.templater.synapses_classes(None, None)
        writer.write('synapses_classes.*', synapses_classes_tmp)
        
    def generate_run_source(self, writer, run_includes):
        run_tmp = CUDAStandaloneCodeObject.templater.run(None, None, run_funcs=self.runfuncs,
                                                        code_objects=self.code_objects.values(),
                                                        additional_headers=run_includes,
                                                        )
        writer.write('run.*', run_tmp)
        
    def generate_makefile(self, writer, compiler, native, compiler_flags, nb_threads):
        if compiler=='msvc':
            if native:
                arch_flag = ''
                try:
                    from cpuinfo import cpuinfo
                    res = cpuinfo.get_cpu_info()
                    if 'sse' in res['flags']:
                        arch_flag = '/arch:SSE'
                    if 'sse2' in res['flags']:
                        arch_flag = '/arch:SSE2'
                except ImportError:
                    logger.warn('Native flag for MSVC compiler requires installation of the py-cpuinfo module')
                compiler_flags += ' '+arch_flag
            
            if nb_threads>1:
                openmp_flag = '/openmp'
            else:
                openmp_flag = ''
            # Generate the visual studio makefile
            source_bases = [fname.replace('.cpp', '').replace('/', '\\') for fname in writer.source_files]
            win_makefile_tmp = CUDAStandaloneCodeObject.templater.win_makefile(
                None, None,
                source_bases=source_bases,
                compiler_flags=compiler_flags,
                openmp_flag=openmp_flag,
                )
            writer.write('win_makefile', win_makefile_tmp)
        else:
            # Generate the makefile
            if os.name=='nt':
                rm_cmd = 'del *.o /s\n\tdel main.exe $(DEPS)'
            else:
                rm_cmd = 'rm $(OBJS) $(PROGRAM) $(DEPS)'
            makefile_tmp = CUDAStandaloneCodeObject.templater.makefile(None, None,
                source_files=' '.join(writer.source_files),
                header_files=' '.join(writer.header_files),
                compiler_flags=compiler_flags,
                rm_cmd=rm_cmd)
            writer.write('makefile', makefile_tmp)


    def build(self, directory='output',
              compile=True, run=True, debug=False, clean=True,
              with_output=True, native=True,
              additional_source_files=None, additional_header_files=None,
              main_includes=None, run_includes=None,
              run_args=None, **kwds):
        '''
        Build the project
        
        TODO: more details
        
        Parameters
        ----------
        directory : str
            The output directory to write the project to, any existing files will be overwritten.
        compile : bool
            Whether or not to attempt to compile the project
        run : bool
            Whether or not to attempt to run the built project if it successfully builds.
        debug : bool
            Whether to compile in debug mode.
        with_output : bool
            Whether or not to show the ``stdout`` of the built program when run.
        native : bool
            Whether or not to compile for the current machine's architecture (best for speed, but not portable)
        clean : bool
            Whether or not to clean the project before building
        additional_source_files : list of str
            A list of additional ``.cpp`` files to include in the build.
        additional_header_files : list of str
            A list of additional ``.h`` files to include in the build.
        main_includes : list of str
            A list of additional header files to include in ``main.cpp``.
        run_includes : list of str
            A list of additional header files to include in ``run.cpp``.
        '''
        renames = {'project_dir': 'directory',
                   'compile_project': 'compile',
                   'run_project': 'run'}
        if len(kwds):
            msg = ''
            for kwd in kwds:
                if kwd in renames:
                    msg += ("Keyword argument '%s' has been renamed to "
                            "'%s'. ") % (kwd, renames[kwd])
                else:
                    msg += "Unknown keyword argument '%s'. " % kwd
            raise TypeError(msg)

        if additional_source_files is None:
            additional_source_files = []
        if additional_header_files is None:
            additional_header_files = []
        if main_includes is None:
            main_includes = []
        if run_includes is None:
            run_includes = []
        if run_args is None:
            run_args = []

        compiler, extra_compile_args = get_compiler_and_args()
        compiler_flags = ' '.join(extra_compile_args)
        self.project_dir = directory
        ensure_directory(directory)
        
        for d in ['code_objects', 'results', 'static_arrays']:
            ensure_directory(os.path.join(directory, d))
            
        writer = CUDAWriter(directory)
        
        logger.debug("Writing CUDA standalone project to directory "+os.path.normpath(directory))
        arange_arrays = sorted([(var, start)
                                for var, start in self.arange_arrays.iteritems()],
                               key=lambda (var, start): var.name)

        self.write_static_arrays(directory)
        self.find_synapses()

        # Not sure what the best place is to call Network.after_run -- at the
        # moment the only important thing it does is to clear the objects stored
        # in magic_network. If this is not done, this might lead to problems
        # for repeated runs of standalone (e.g. in the test suite).
        for net in self.networks:
            net.after_run()
            
        self.generate_main_source(writer, main_includes)
        self.generate_codeobj_source(writer)        
        self.generate_objects_source(writer, arange_arrays, self.net_synapses, self.static_array_specs, self.networks)
        self.generate_network_source(writer, compiler)
        self.generate_synapses_classes_source(writer)
        self.generate_run_source(writer, run_includes)
        self.generate_rand_source(writer)
        self.copy_source_files(writer, directory)
        
        writer.source_files.extend(additional_source_files)
        writer.header_files.extend(additional_header_files)
        
        self.generate_makefile(writer, compiler, native, compiler_flags, nb_threads=0)
        
        if compile:
            self.compile_source(directory, compiler, debug, clean, native)
            if run:
                self.run(directory, with_output, run_args)
                
    def network_run(self, net, duration, report=None, report_period=10*second,
                    namespace=None, profile=True, level=0, **kwds):
        CPPStandaloneDevice.network_run(self, net, duration, report, report_period, namespace, profile, level+1)
        clock = 0
        for func, args in self.main_queue:
            if func=='run_network':
                net, netcode = args
                clock = net._clocks[0]
                run_action = netcode.pop()
                netcode.append('{net.name}.add(&{clock.name}, _run_random_number_generation);'.format(clock=clock, net=net));
                netcode.append(run_action)


                        
cuda_standalone_device = CUDAStandaloneDevice()

all_devices['cuda_standalone'] = cuda_standalone_device
