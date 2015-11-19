# Variables
CC	:= gcc
CFLAGS	:= -Wall -Wextra
NVCC	:= /usr/local/cuda/bin/nvcc

INCLUDE	:= 			\
	/opt/DIS/include	\
	/opt/DIS/include/dis 	\
	/usr/local/cuda/include

# Targets
.PHONY: shared_cuda shared_sisci all clean

all: shared_cuda shared_sisci

clean:
	-$(RM) cuda_shared.o sci_shared.o

shared_cuda: cuda_shared.o
	$(CC) -shared -fPIC -o $@ $^

shared_sisci: sci_shared.o
	$(CC) -shared -fPIC -o $@ $^

# How to compile
%.o: %.cu
	$(NVCC) -std=c++11 --compiler-options "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c

%.o: %.c
	$(CC) -x c -std=gnu11 $(CFLAGS) $(addprefix -I,$(INCLUDE)) -o $@ $< -c
