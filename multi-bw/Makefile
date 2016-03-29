PROJECT := $(notdir $(shell pwd))
OBJECTS	:= main.o devbuf.o hostbuf.o bench.o stream.o
DEPS	:= devbuf.h hostbuf.h bench.h stream.h

ifeq ($(shell uname -s),Darwin)
CCDIR	:= /Library/Developer/CommandLineTools/usr/bin/
else
CCDIR   := /usr/bin/
endif
NVCC    := /usr/local/cuda/bin/nvcc
CFLAGS  := -Wall -Wextra -Wno-gnu-designator -g

INCLUDE	:= /usr/local/cuda/include 

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(NVCC) -ccbin $(CCDIR) -o $@ $^ 

%.o: %.cu $(DEPS)
	$(NVCC) -x cu -ccbin $(CCDIR) -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c 
