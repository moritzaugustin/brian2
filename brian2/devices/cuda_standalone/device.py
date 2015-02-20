'''
Module implementing the CUDA "standalone" device.
'''
import os
import shutil
import subprocess
import inspect
from collections import defaultdict

import numpy as np

from brian2.core.clocks import defaultclock
from brian2.core.network import Network
from brian2.devices.device import Device, all_devices
from brian2.core.variables import *
from brian2.synapses.synapses import Synapses
from brian2.core.preferences import prefs, BrianPreference
from brian2.utils.filetools import copy_directory, ensure_directory, in_directory
from brian2.utils.stringtools import word_substitute
from brian2.codegen.generators.cuda_generator import c_data_type
from brian2.units.fundamentalunits import Quantity, have_same_dimensions
from brian2.units import second
from brian2.utils.logger import get_logger

from .codeobject import CUDAStandaloneCodeObject
from brian2.devices.cpp_standalone.device import CPPWriter, CPPStandaloneDevice, freeze, invert_dict


__all__ = []

logger = get_logger(__name__)


# Preferences
prefs.register_preferences(
    'devices.cuda_standalone',
    'CUDA standalone preferences ',
    optimisation_flags = BrianPreference(
        default='-O3',
        docs='''
        Optimisation flags to pass to the compiler
        '''
        ),
    )

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
        self.code_objects[codeobj.name] = codeobj
        return codeobj

    def build(self, directory='output', compile=True, run=False, debug=True,
              optimisations='-O3 -ffast-math',
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
            Whether or not to attempt to compile the project using GNU make.
        run : bool
            Whether or not to attempt to run the built project if it successfully builds.
        debug : bool
            Whether to compile in debug mode.
        with_output : bool
            Whether or not to show the ``stdout`` of the built program when run.
        native : bool
            Whether or not to compile natively using the ``--march=native`` gcc option.
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
        host_parameters = defaultdict(list)
        device_parameters = defaultdict(list)
        kernel_variables = defaultdict(list)
        self.project_dir = directory
        ensure_directory(directory)
        
        for d in ['code_objects', 'results', 'static_arrays']:
            ensure_directory(os.path.join(directory, d))
            
        writer = CUDAWriter(directory)
        

        logger.debug("Writing CUDA standalone project to directory "+os.path.normpath(directory))
        arange_arrays = sorted([(var, start)
                                for var, start in self.arange_arrays.iteritems()],
                               key=lambda (var, start): var.name)

        # # Find np arrays in the namespaces and convert them into static
        # # arrays. Hopefully they are correctly used in the code: For example,
        # # this works for the namespaces for functions with C++ (e.g. TimedArray
        # # treats it as a C array) but does not work in places that are
        # # implicitly vectorized (state updaters, resets, etc.). But arrays
        # # shouldn't be used there anyway.
        for code_object in self.code_objects.itervalues():
            for name, value in code_object.variables.iteritems():
                if isinstance(value, np.ndarray):
                    self.static_arrays[name] = value

        # write the static arrays
        logger.debug("static arrays: "+str(sorted(self.static_arrays.keys())))
        static_array_specs = []
        for name, arr in sorted(self.static_arrays.items()):
            arr.tofile(os.path.join(directory, 'static_arrays', name))
            static_array_specs.append((name, c_data_type(arr.dtype), arr.size, name))

        # Write the global objects
        networks = [net() for net in Network.__instances__()
                    if net().name != '_fake_network']
        synapses = []
        for net in networks:
            synapses.extend(s for s in net.objects if isinstance(s, Synapses))

        # Not sure what the best place is to call Network.after_run -- at the
        # moment the only important thing it does is to clear the objects stored
        # in magic_network. If this is not done, this might lead to problems
        # for repeated runs of standalone (e.g. in the test suite).
        for net in networks:
            net.after_run()
            
        #check how many random numbers are needed per step
        num_rand_normal = 0
        num_rand_uniform = 0
        for code_object in self.code_objects.itervalues():
            if code_object.runs_every_tick:
                for name, value in code_object.variables.iteritems():
                    if name == "_python_rand":
                        code_object.rand_start = num_rand_uniform
                        num_rand_uniform += code_object.owner.N
                    elif name == "_python_randn":
                        code_object.rand_start_uniform = num_rand_normal
                        num_rand_normal += code_object.owner.N
                        
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
                        networks=networks)
        arr_tmp.cu_file = arr_tmp.cu_file.replace('%RANDOM_NUMBER_NORMAL%', str(num_rand_normal))
        arr_tmp.cu_file = arr_tmp.cu_file.replace('%RANDOM_NUMBER_UNIFORM%', str(num_rand_uniform))
        writer.write('objects.*', arr_tmp)

        main_lines = []
        procedures = [('', main_lines)]
        runfuncs = {}
        for func, args in self.main_queue:
            if func=='run_code_object':
                codeobj, = args
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

        # Generate data for non-constant values
        code_object_defs = defaultdict(list)
        for codeobj in self.code_objects.itervalues():
            lines = []
            additional_code = []
            for k, v in codeobj.variables.iteritems():
                if k == "_python_rand" and codeobj.runs_every_tick == False and codeobj.template_name <> "synapses_create":
                    additional_code.append('''
                        //genenerate an arry of random numbers on the device
                        float* dev_array_randn;
                        cudaMalloc((void**)&dev_array_randn, sizeof(float)*N);
                        if(!dev_array_randn)
                        {
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof(float)*N);
                        }
                        curandGenerator_t gen;
                        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
                        curandSetPseudoRandomGeneratorSeed(gen, time(0));
                        curandGenerateNormal(gen, dev_array_randn, N, 0, 1);''')
                    line = "float* _array_randn"
                    device_parameters[codeobj.name].append(line)
                    host_parameters[codeobj.name].append("dev_array_randn")
                elif k == "_python_rand" and codeobj.runs_every_tick == False and codeobj.template_name <> "synapses_create":
                    additional_code.append('''
                        //genenerate an arry of random numbers on the device
                        float* dev_array_rand;
                        cudaMalloc((void**)&dev_array_rand, sizeof(float)*N);
                        if(!dev_array_rand)
                        {
                            printf("ERROR while allocating device memory with size %ld\\n", sizeof(float)*N);
                        }
                        curandGenerator_t gen;
                        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
                        curandSetPseudoRandomGeneratorSeed(gen, time(0));
                        curandGenerateUniform(gen, dev_array_rand, N);''')
                    line = "float* _array_rand"
                    device_parameters[codeobj.name].append(line)
                    host_parameters[codeobj.name].append("dev_array_rand")
                elif isinstance(v, AttributeVariable):
                    # We assume all attributes are implemented as property-like methods
                    line = 'const {c_type} {varname} = {objname}.{attrname}();'
                    lines.append(line.format(c_type=c_data_type(v.dtype), varname=k, objname=v.obj.name,
                                             attrname=v.attribute)) 
                    host_parameters[codeobj.name].append(k)
                    line = "{c_type} par_{varname}"
                    device_parameters[codeobj.name].append(line.format(c_type=c_data_type(v.dtype), varname=k))
                    line = 'const {c_type} {varname} = par_{varname};'
                    kernel_variables[codeobj.name].append(line.format(c_type=c_data_type(v.dtype), varname=k))
                elif isinstance(v, ArrayVariable):
                    try:
                        if isinstance(v, DynamicArrayVariable):
                            if v.dimensions == 1:
                                dyn_array_name = self.dynamic_arrays[v]
                                array_name = self.arrays[v]
                                line = '{c_type}* const {array_name} = &{dyn_array_name}[0];'
                                line = line.format(c_type=c_data_type(v.dtype), array_name=array_name,
                                                   dyn_array_name=dyn_array_name)
                                lines.append(line)
                                line = 'const int _num{k} = {dyn_array_name}.size();'
                                line = line.format(k=k, dyn_array_name=dyn_array_name)
                                lines.append(line)
                                
                                host_parameters[codeobj.name].append(array_name)
                                host_parameters[codeobj.name].append("_num" + k)
                                
                                line = "{c_type}* par_{array_name}"
                                device_parameters[codeobj.name].append(line.format(c_type=c_data_type(v.dtype), array_name=array_name))
                                line = "int par_num{array_name}"
                                device_parameters[codeobj.name].append(line.format(array_name=array_name))
                                
                                line = "{c_type}* _ptr{array_name} = par_{array_name};"
                                kernel_variables[codeobj.name].append(line.format(c_type=c_data_type(v.dtype), array_name=array_name))
                                line = "const int _num{array_name} = par_num{array_name};"
                                kernel_variables[codeobj.name].append(line.format(array_name=array_name))
                        else:
                            host_parameters[codeobj.name].append("dev"+self.get_array_name(v))
                            device_parameters[codeobj.name].append("%s* par_%s" % (c_data_type(v.dtype), self.get_array_name(v)))
                            kernel_variables[codeobj.name].append("%s* _ptr%s = par_%s;" % (c_data_type(v.dtype),  self.get_array_name(v), self.get_array_name(v)))

                            code_object_defs[codeobj.name].append('const int _num%s = %s;' % (k, v.size))
                            kernel_variables[codeobj.name].append('const int _num%s = %s;' % (k, v.size))
                    except TypeError:
                        pass
            for line in lines:
                # Sometimes an array is referred to by to different keys in our
                # dictionary -- make sure to never add a line twice
                if not line in code_object_defs[codeobj.name]:
                    code_object_defs[codeobj.name].append(line)
            for line in additional_code:
                code_object_defs[codeobj.name].append(line)

        # Generate the code objects
        for codeobj in self.code_objects.itervalues():
            ns = codeobj.variables
            # TODO: fix these freeze/CONSTANTS hacks somehow - they work but not elegant.
            code = freeze(codeobj.code.cu_file, ns)
            code = code.replace('%CONSTANTS%', '\n\t\t'.join(code_object_defs[codeobj.name]))
            code = code.replace('%HOST_PARAMETERS%', ',\n\t\t\t'.join(host_parameters[codeobj.name]))
            code = code.replace('%DEVICE_PARAMETERS%', ',\n\t'.join(device_parameters[codeobj.name]))
            code = code.replace('%KERNEL_VARIABLES%', '\n\t'.join(kernel_variables[codeobj.name]))
            code = code.replace('%RAND_NORMAL_START%', str(codeobj.rand_start_normal))
            code = code.replace('%RAND_UNIFORM_START%', str(codeobj.rand_start_uniform))
            code = '#include "objects.h"\n'+code
            
            writer.write('code_objects/'+codeobj.name+'.cu', code)
            writer.write('code_objects/'+codeobj.name+'.h', codeobj.code.h_file)
                    
        # The code_objects are passed in the right order to run them because they were
        # sorted by the Network object. To support multiple clocks we'll need to be
        # smarter about that.
        main_tmp = CUDAStandaloneCodeObject.templater.main(None, None,
                                                          main_lines=main_lines,
                                                          code_objects=self.code_objects.values(),
                                                          report_func=self.report_func,
                                                          dt=float(defaultclock.dt),
                                                          additional_headers=main_includes,
                                                          )
        writer.write('main.cu', main_tmp)

        main_tmp = CUDAStandaloneCodeObject.templater.network(None, None)
        writer.write('network.*', main_tmp)

        main_tmp = CUDAStandaloneCodeObject.templater.synapses_classes(None, None)
        writer.write('synapses_classes.*', main_tmp)
        
        # Generate the run functions
        run_tmp = CUDAStandaloneCodeObject.templater.run(None, None, run_funcs=runfuncs,
                                                        code_objects=self.code_objects.values(),
                                                        additional_headers=run_includes,
                                                        )
        writer.write('run.*', run_tmp)
        
        rand_tmp = CUDAStandaloneCodeObject.templater.rand(None, None)
        writer.write('rand.*', rand_tmp)

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

        writer.source_files.extend(additional_source_files)
        writer.header_files.extend(additional_header_files)

        # Generate the makefile
        if os.name=='nt':
            rm_cmd = 'del'
        else:
            rm_cmd = 'rm'
        makefile_tmp = CUDAStandaloneCodeObject.templater.makefile(None, None,
            source_files=' '.join(writer.source_files),
            header_files=' '.join(writer.header_files),
            optimisations=prefs['devices.cuda_standalone.optimisation_flags'],
            rm_cmd=rm_cmd)
        writer.write('makefile', makefile_tmp)

        # build the project
        if compile:
            with in_directory(directory):
                if debug:
                    x = os.system('make debug')
                elif native:
                    x = os.system('make native')
                else:
                    x = os.system('make')
                if x==0:
                    if run:
                        if not with_output:
                            stdout = open(os.devnull, 'w')
                        else:
                            stdout = None
                        if os.name=='nt':
                            x = subprocess.call(['main'] + run_args, stdout=stdout)
                        else:
                            x = subprocess.call(['./main'] + run_args, stdout=stdout)
                        if x:
                            raise RuntimeError("Project run failed")
                        self.has_been_run = True
                else:
                    raise RuntimeError("Project compilation failed")
                
    def network_run(self, net, duration, report=None, report_period=10*second,
                        namespace=None, level=0):
        net._clocks = [obj.clock for obj in net.objects]
        # We have to use +2 for the level argument here, since this function is
        # called through the device_override mechanism
        net.before_run(namespace, level=level+2)
            
        self.clocks.update(net._clocks)

        # We run a simplified "update loop" that only advances the clocks
        # This can be useful because some Python code might use the t attribute
        # of the Network or a NeuronGroup etc.
        t_end = net.t+duration
        for clock in net._clocks:
            clock.set_interval(net.t, t_end)
            # manually set the clock to the end, no need to run Clock.tick() in a loop
            clock._i = clock._i_end
        net.t_ = float(t_end)

        # TODO: remove this horrible hack
        for clock in self.clocks:
            if clock.name=='clock':
                clock._name = '_clock'
            
        # Extract all the CodeObjects
        # Note that since we ran the Network object, these CodeObjects will be sorted into the right
        # running order, assuming that there is only one clock
        code_objects = []
        for obj in net.objects:
            for codeobj in obj._code_objects:
                codeobj.runs_every_tick = True
                code_objects.append((obj.clock, codeobj))

        # Code for a progress reporting function
        standard_code = '''
        void report_progress(const double elapsed, const double completed, const double duration)
        {
            if (completed == 0.0)
            {
                %STREAMNAME% << "Starting simulation for duration " << duration << " s";
            } else
            {
                %STREAMNAME% << completed*duration << " s (" << (int)(completed*100.) << "%) simulated in " << elapsed << " s";
                if (completed < 1.0)
                {
                    const int remaining = (int)((1-completed)/completed*elapsed+0.5);
                    %STREAMNAME% << ", estimated " << remaining << " s remaining.";
                }
            }

            %STREAMNAME% << std::endl << std::flush;
        }
        '''
        if report is None:
            self.report_func = ''
        elif report == 'text' or report == 'stdout':
            self.report_func = standard_code.replace('%STREAMNAME%', 'std::cout')
        elif report == 'stderr':
            self.report_func = standard_code.replace('%STREAMNAME%', 'std::cerr')
        elif isinstance(report, basestring):
            self.report_func = '''
            void report_progress(const double elapsed, const double completed, const double duration)
            {
            %REPORT%
            }
            '''.replace('%REPORT%', report)
        else:
            raise TypeError(('report argument has to be either "text", '
                             '"stdout", "stderr", or the code for a report '
                             'function'))

        if report is not None:
            report_call = 'report_progress'
        else:
            report_call = 'NULL'

        # Generate the updaters
        run_lines = ['{net.name}.clear();'.format(net=net)]
        run_lines.append('{net.name}.add(&{clock.name}, _run_random_number_generation);'.format(clock=clock, net=net));
        for clock, codeobj in code_objects:
            run_lines.append('{net.name}.add(&{clock.name}, _run_{codeobj.name});'.format(clock=clock, net=net,
                                                                                               codeobj=codeobj))
        run_lines.append('{net.name}.run({duration}, {report_call}, {report_period});'.format(net=net,
                                                                                              duration=float(duration),
                                                                                              report_call=report_call,
                                                                                              report_period=float(report_period)))
        self.main_queue.append(('run_network', (net, run_lines)))



class RunFunctionContext(object):
    def __enter__(self):
        cuda_standalone_device.main_queue.append(('start_run_func', (self.name, self.include_in_parent)))
    def __exit__(self, type, value, traceback):
        cuda_standalone_device.main_queue.append(('end_run_func', (self.name, self.include_in_parent)))


cuda_standalone_device = CUDAStandaloneDevice()

all_devices['cuda_standalone'] = cuda_standalone_device
