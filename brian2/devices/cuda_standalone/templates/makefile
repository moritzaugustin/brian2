PROGRAM = main
SRCS_CPP = $(wildcard *.cpp */*.cpp)
SRCS_CU  = $(wildcard *.cu */*.cu)
OBJS_CPP = ${SRCS_CPP:.cpp=.o}
OBJS_CU  = ${SRCS_CU:.cu=.o}
CC = @nvcc
DEBUG = -g -G -DTHRUST_DEBUG --ptxas-options=-v
#we need SM >= 2.0 for the -dc flag (see below) 
OPTIMISATIONS = --use_fast_math -arch=sm_35 {{ compiler_flags }}
CFLAGS = -c $(OPTIMISATIONS) -I.
LFLAGS = -lcurand

all: executable

# Adds debug flags
debug: CFLAGS += $(DEBUG)
debug: LFLAGS += $(DEBUG)
debug: executable

.PHONY: all debug executable clean native

native: executable

executable: $(OBJS_CPP) $(OBJS_CU)
	$(CC) $(LFLAGS) -arch=sm_20 $(OBJS_CPP) $(OBJS_CU) -o $(PROGRAM)

clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(PROGRAM) results/*

#we need the "-dc" flag for using the same device variable in multiple files
%.o : %.cu
	$(CC) $(CFLAGS) -dc $< -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@

