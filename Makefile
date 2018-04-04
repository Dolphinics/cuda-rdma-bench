PROJECT := rdma-bench
SOURCES := $(wildcard src/*.c) $(wildcard src/*.cu)
OBJECTS	:= $(SOURCES:%=%.o) 
CUDA_DIR := /usr/local/cuda

CC      := gcc
NVCC    := $(CUDA_DIR)/bin/nvcc
CFLAGS  := -Wall -Wextra -D_REENTRANT -g

INCLUDE	:= $(CUDA_DIR)/include /opt/DIS/include /opt/DIS/include/dis

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(NVCC) -o $@ $^ -L/opt/DIS/lib64 -lsisci -lcuda -lrt -lpthread

src/%.c.o: src/%.c
	$(CC) -x c -std=gnu99 $(CFLAGS) $(addprefix -I,$(INCLUDE)) -o $@ $< -c 

src/%.cu.o: src/%.cu
	$(NVCC) -x cu -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c 
