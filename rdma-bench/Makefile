PROJECT := $(notdir $(shell pwd))
OBJECTS	:= main.o log.o errstr.o local.o remote.o util.o dma.o

CUDA_HOME := /usr/local/cuda

CC	:= gcc
NVCC	:= gcc
CFLAGS	:= -Wall -Wextra -D_REENTRANT -g
INCLUDE	:= /opt/DIS/include /opt/DIS/include/dis
LIBS	:= -lsisci -lrt -lpthread

ifneq ($(wildcard $(CUDA_HOME)/*),)
	NVCC 	:= $(CUDA_HOME)/bin/nvcc
	INCLUDE += $(CUDA_HOME)/include
	LIBS	+= -lcuda
else
	CFLAGS += -DNO_CUDA
endif


.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(NVCC) -o $@ $^ -L/opt/DIS/lib64 $(LIBS)

%.o: %.c
	$(CC) -x c -std=gnu99 $(CFLAGS) $(addprefix -I,$(INCLUDE)) -o $@ $< -c 

#%.o: %.cu
#	$(NVCC) -x cu -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c 
