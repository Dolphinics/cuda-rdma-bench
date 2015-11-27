PROJECT := $(notdir $(shell pwd))
OBJECTS	:= main.o local.o remote.o dma.o

CC		:= gcc
NVCC 	:= /usr/local/cuda/bin/nvcc
CFLAGS	:= -Wall -Wextra

INCLUDE	:= /usr/local/cuda/include /opt/DIS/include /opt/DIS/include/dis ../shared

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS) ../shared/shsci.o ../shared/shcuda.o
	$(NVCC) -o $@ $^ -L/opt/DIS/lib64 -lsisci -lcuda -lrt

%.o: %.c
	$(CC) -x c -std=gnu99 $(CFLAGS) $(addprefix -I,$(INCLUDE)) -o $@ $< -c

%.o: %.cu
	$(NVCC) -x cu -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c
