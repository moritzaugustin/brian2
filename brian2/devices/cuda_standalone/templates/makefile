PROGRAM = main
SRCS_CPP = $(wildcard *.cpp */*.cpp)
SRCS_CU  = $(wildcard *.cu */*.cu)
OBJS_CPP = ${SRCS_CPP:.cpp=.o}
OBJS_CU  = ${SRCS_CU:.cu=.o}
CC = @nvcc
DEBUG = -g -G -DTHRUST_DEBUG --ptxas-options=-v 
OPTIMISATIONS = --use_fast_math {{ compiler_flags }}
CFLAGS = -c $(OPTIMISATIONS) -I.
LFLAGS = -lcurand

all: executable

# Adds debug flags
debug: CFLAGS += $(DEBUG)
debug: LFLAGS += $(DEBUG)
debug: executable

.PHONY: all debug executable clean native

native: executable

#we need SM >= 2.0 for the -dc flag (see below)
# attention: changing the arch version (and maybe others) requires at least 
#   adaptation of the threadcount of the _combine kernel in spatialstateupdate.cu template
#   since there a static --ptxas-options=-v based register consumption is used
executable: $(OBJS_CPP) $(OBJS_CU)
	$(CC) $(LFLAGS) -arch=sm_20 $(OBJS_CPP) $(OBJS_CU) -o $(PROGRAM)

clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(PROGRAM) results/*

#we need the "-dc" flag for using the same device variable in multiple files
%.o : %.cu
	$(CC) $(CFLAGS) -dc $< -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@

