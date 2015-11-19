include ../Makefile

name := $(notdir $(shell pwd))
objs := memcpy.o

.PHONY: $(name) all clean

all: $(name)

$(name): $(eval $(use_cuda))
$(name): $(objs)
	$(NVCC) -o $@ $^ $(libs)

clean:
	-$(RM) $(name) $(objs)
