PROJECT := lgpu2lgpu-p2p
OBJECTS	:= p2pmemcpy.o

CFLAGS	:= -Wall -Wextra
CC 	:= /usr/local/cuda/bin/nvcc

INCLUDE	:= /usr/local/cuda/include ../shared
LD_PATH := ../shared

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(CC) -o $@ $^ $(addprefix -L,$(LD_PATH)) -lshcuda

%.o: %.cu
	$(CC) -std=c++11 --compiler-options "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c
