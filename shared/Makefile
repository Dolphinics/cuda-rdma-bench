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

all: libshcuda.a libshsci.a

clean:
	-$(RM) libsh*.a libsh*.so sh*.o

libshcuda.a: shcuda.o
	$(NVCC) -shared -o $@ $^

libshsci.a: shsci.o
	$(CC) -shared -o $@ $^

# How to compile
%.o: %.cu
	$(NVCC) -x cu -Xcompiler "-fPIC $(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c

%.o: %.c
	$(CC) -x c -std=gnu99 -fPIC $(CFLAGS) $(addprefix -I,$(INCLUDE)) -o $@ $< -c
