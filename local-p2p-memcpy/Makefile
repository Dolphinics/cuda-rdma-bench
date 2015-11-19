PROJECT := local-p2p-memcpy
OBJECTS	:= local-p2p-memcpy.o

CC	:= gcc
CFLAGS	:= -Wall -Wextra
NVCC 	:= /usr/local/cuda/bin/nvcc

INCLUDE	:= /usr/local/cuda/include

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS)
	$(CC) -o $@ $^

%.o: %.cu
	$(NVCC) -std=c++11 --compiler-options "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c
