PROJECT := $(notdir $(shell pwd))
OBJECTS	:= main.o reporting.o local.o server.o bench.o

CC	:= gcc
NVCC 	:= /usr/local/cuda/bin/nvcc
CFLAGS	:= -Wall -Wextra -D_REENTRANT

INCLUDE	:= /usr/local/cuda/include /opt/DIS/include /opt/DIS/include/dis

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(NVCC) -o $@ $^ -L/opt/DIS/lib64 -lsisci -lcuda -lrt

%.o: %.c
	$(CC) -x c -std=gnu99 $(CFLAGS) $(addprefix -I,$(INCLUDE)) -o $@ $< -c 

%.o: %.cu
	$(NVCC) -x cu -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c 
