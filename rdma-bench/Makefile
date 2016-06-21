PROJECT := $(notdir $(shell pwd))
OBJECTS	:= main.o log.o errstr.o local.o remote.o util.o dma.o

CC	:= gcc
NVCC 	:= /usr/local/cuda/bin/nvcc
CFLAGS	:= -Wall -Wextra -D_REENTRANT -g

INCLUDE	:= /usr/local/cuda/include /opt/DIS/include /opt/DIS/include/dis

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(NVCC) -o $@ $^ -L/opt/DIS/lib64 -lsisci -lcuda -lrt -lpthread

%.o: %.c
	$(CC) -x c -std=gnu99 $(CFLAGS) $(addprefix -I,$(INCLUDE)) -o $@ $< -c 

%.o: %.cu
	$(NVCC) -x cu -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c 
