# Variables
CC	:= gcc
CFLAGS	:= -Wall -Wextra
NVCC	:= /usr/local/cuda/bin/nvcc

INCLUDE	:= 			\
	/opt/DIS/include	\
	/opt/DIS/include/dis 	\
	/usr/local/cuda/include

# Targets
.PHONY: all clean libshcuda.so libshsci.so

all: libshcuda.so libshsci.so

clean:
	-$(RM) libsh*.so sh*.o

libshcuda.so: shcuda.o
	$(CC) -shared -o $@ $^

libshsci.so: shsci.o
	$(CC) -shared -o $@ $^

# How to compile
%.o: %.cu
	$(NVCC) -std=c++11 --compiler-options "-fPIC $(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c

%.o: %.c
	$(CC) -x c -std=gnu11 -fPIC $(CFLAGS) $(addprefix -I,$(INCLUDE)) -o $@ $< -c
