PROGRAM = main
SRCS = objects.cu code_objects/synapses_pre_push_spikes.cu code_objects/synapses_stateupdater_codeobject.cu code_objects/spikemonitor_codeobject.cu code_objects/neurongroup_resetter_codeobject.cu code_objects/ratemonitor_codeobject.cu code_objects/synapses_post_initialise_queue.cu code_objects/poissongroup_group_variable_set_conditional_codeobject.cu code_objects/synapses_post_codeobject.cu code_objects/statemonitor_codeobject.cu code_objects/synapses_pre_codeobject.cu code_objects/poissongroup_thresholder_codeobject.cu code_objects/synapses_group_variable_set_conditional_codeobject_1.cu code_objects/neurongroup_thresholder_codeobject.cu code_objects/synapses_post_push_spikes.cu code_objects/synapses_synapses_create_codeobject.cu code_objects/neurongroup_stateupdater_codeobject.cu code_objects/synapses_pre_initialise_queue.cu code_objects/synapses_group_variable_set_conditional_codeobject.cu main.cu run.cu brianlib/network.cpp random.cu 
H_SRCS = objects.h code_objects/synapses_pre_push_spikes.h code_objects/synapses_stateupdater_codeobject.h code_objects/spikemonitor_codeobject.h code_objects/neurongroup_resetter_codeobject.h code_objects/ratemonitor_codeobject.h code_objects/synapses_post_initialise_queue.h code_objects/poissongroup_group_variable_set_conditional_codeobject.h code_objects/synapses_post_codeobject.h code_objects/statemonitor_codeobject.h code_objects/synapses_pre_codeobject.h code_objects/poissongroup_thresholder_codeobject.h code_objects/synapses_group_variable_set_conditional_codeobject_1.h code_objects/neurongroup_thresholder_codeobject.h code_objects/synapses_post_push_spikes.h code_objects/synapses_synapses_create_codeobject.h code_objects/neurongroup_stateupdater_codeobject.h code_objects/synapses_pre_initialise_queue.h code_objects/synapses_group_variable_set_conditional_codeobject.h run.h brianlib/network.h brianlib/common_math.h brianlib/synapses.h brianlib/dynamic_array.h brianlib/clocks.h brianlib/CudaVector.h random.h
SRCS_CPP = $(wildcard *.cpp */*.cpp)
SRCS_CU  = $(wildcard *.cu */*.cu)
OBJS_CPP = ${SRCS_CPP:.cpp=.o}
OBJS_CU  = ${SRCS_CU:.cu=.o}
CC = @nvcc
DEBUG = -g
OPTIMISATIONS = --use_fast_math -O3 -arch=sm_20
CFLAGS = -c $(OPTIMISATIONS) -I. 
LFLAGS = -lcurand

all: executable

# Adds debug flags
debug: CFLAGS += $(DEBUG)
debug: LFLAGS += $(DEBUG)
debug: executable

# Adds -march=native to optimisations, which only works for recent gcc versions
native: OPTIMISATIONS += -march=native
native: executable

.PHONY: all debug native executable clean

executable: $(OBJS_CPP) $(OBJS_CU) $(DEPS)
	$(CC) $(LFLAGS) $(OBJS_CPP) $(OBJS_CU) -o $(PROGRAM)

clean:
	rm $(OBJS_CPP) $(OBJS_CU) $(PROGRAM) $(DEPS) results/*

%.o : %.cu
	$(CC) $(CFLAGS) $< -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@
