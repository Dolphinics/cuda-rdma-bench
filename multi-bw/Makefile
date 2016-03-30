PROJECT := $(notdir $(shell pwd))
OBJECTS	:= main.o devbuf.o hostbuf.o bench.o stream.o event.o
DEPS	:= devbuf.h hostbuf.h bench.h stream.h event.h
CFLAGS  := -Wall -Wextra -g
NVCC    := /usr/local/cuda/bin/nvcc

ifeq ($(shell uname -s),Darwin)
CCDIR	:= /Library/Developer/CommandLineTools/usr/bin/
CFLAGS  += -Wno-gnu-designator
else
CCDIR   := /usr/bin/g++
endif

INCLUDE	:= /usr/local/cuda/include 

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(NVCC) -ccbin $(CCDIR) -o $@ $^ 

%.o: %.cu $(DEPS)
	$(NVCC) -std=c++11 -x cu -ccbin $(CCDIR) -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c 
