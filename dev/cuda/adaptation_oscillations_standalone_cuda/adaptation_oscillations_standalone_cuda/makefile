PROGRAM = main
SRCS_CPP = $(wildcard *.cpp */*.cpp)
SRCS_CU  = $(wildcard *.cu */*.cu)
OBJS_CPP = ${SRCS_CPP:.cpp=.o}
OBJS_CU  = ${SRCS_CU:.cu=.o}
CC = @nvcc
DEBUG = -g -G
OPTIMISATIONS = --use_fast_math -arch=sm_20
CFLAGS = -c $(OPTIMISATIONS) -I.
LFLAGS = -lcurand

all: executable

# Adds debug flags
debug: CFLAGS += $(DEBUG)
debug: LFLAGS += $(DEBUG)
debug: executable

.PHONY: all debug executable clean

executable: $(OBJS_CPP) $(OBJS_CU) brianlib/spikequeue.h
	$(CC) $(LFLAGS) -arch=sm_20 $(OBJS_CPP) $(OBJS_CU) -o $(PROGRAM)

clean:
	rm -f $(OBJS_CPP) $(OBJS_CU) $(PROGRAM) $(DEPS) results/*

%.o : %.cu
	$(CC) $(CFLAGS) -dc $< -o $@

%.o : %.cpp
	$(CC) $(CFLAGS) $< -o $@
