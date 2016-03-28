PROJECT := $(notdir $(shell pwd))
OBJECTS	:= main.o devbuf.o hostbuf.o bench.o stream.o
DEPS	:= devbuf.h hostbuf.h bench.h stream.h

CC      := gcc
NVCC    := /usr/local/cuda/bin/nvcc
CFLAGS  := -Wall -Wextra -Wno-gnu-designator -g

INCLUDE	:= /usr/local/cuda/include 

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(NVCC) -o $@ $^ 

%.o: %.cu $(DEPS)
	$(NVCC) -x cu -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c 
