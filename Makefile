PROJECT	:= gd

# Necessary binaries (C/C++ compiler and CUDA compiler)
CC      := gcc
ifneq ($(shell which colorgcc),)
	CC := colorgcc
endif
CFLAGS  := -Wall -Wextra -g
NVCC    := /usr/local/cuda/bin/nvcc

# Include paths for SISCI and CUDA
INCLUDE	:= /opt/DIS/include /opt/DIS/include/dis /usr/local/cuda/include

# Link paths
LD_PATH := /opt/DIS/lib64

# Source files
SOURCES := $(wildcard src/*.c) $(wildcard src/*.cpp) $(wildcard src/*.cu)
HEADERS	:= $(wildcard src/*.h) $(wildcard src/*.hpp)


# Makefile targets
.PHONY: all clean $(PROJECT)-bench $(PROJECT)-query

all: $(PROJECT)-query $(PROJECT)-bench

clean:
	-$(RM) $(QUERY_OBJS)
	-$(RM) $(BENCH_OBJS)
	-$(RM) $(PROJECT)-query $(PROJECT)-bench

# Target template
define target_template
obj/$(2).%.cu.o: src/%.cu
	-@mkdir -p $$(@D)
	$$(NVCC) -std=c++11 --compiler-options "$$(CFLAGS)" $$(addprefix -I,$$(INCLUDE)) -o $$@ $$< -c -D$(1)

obj/$(2).%.cpp.o: src/%.cpp
	-@mkdir -p $$(@D)
	$$(CC) -x c++ -std=gnu++11 $$(CFLAGS) $$(addprefix -I,$$(INCLUDE)) -o $$@ $$< -c -D$(1)

obj/$(2).%.c.o: src/%.c
	-@mkdir -p $$(@D)
	$$(CC) -x c -std=gnu11 $$(CFLAGS) $$(addprefix -I,$$(INCLUDE)) -o $$@ $$< -c -D$(1)

$(1)_OBJS = $$(SOURCES:src/%=obj/$(2).%.o)
endef


$(eval $(call target_template,QUERY,query))
$(eval $(call target_template,BENCH,bench))

$(PROJECT)-query: $(QUERY_OBJS)
	$(NVCC) -o $@ $^ -lsisci $(addprefix -L,$(LD_PATH)) -lcuda

$(PROJECT)-bench: $(BENCH_OBJS)
	$(NVCC) -o $@ $^ -lsisci $(addprefix -L,$(LD_PATH)) -lcuda

