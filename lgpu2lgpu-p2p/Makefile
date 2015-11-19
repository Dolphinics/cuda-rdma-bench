PROJECT := lgpu2lgpu-p2p
OBJECTS	:= p2pmemcpy.o

CC 	:= /usr/local/cuda/bin/nvcc
CFLAGS	:= -Wall -Wextra

INCLUDE	:= /usr/local/cuda/include ../shared

.PHONY: all clean $(PROJECT)

all: $(PROJECT)

clean:
	-$(RM) $(PROJECT) $(OBJECTS)

$(PROJECT): $(OBJECTS) ../shared/libshcuda.a
	$(CC) -o $@ $^ 

%.o: %.cu
	$(CC) -Xcompiler "$(CFLAGS)" $(addprefix -I,$(INCLUDE)) -o $@ $< -c
