# Source code directories
DIRS 	:= local-memcpy

# Common libraries
LD_LIBS := rt
LD_PATH	:=

# Necessary binaries (C/C++ compiler and CUDA compiler)
CC     	:= gcc
CFLAGS	:= -Wall -Wextra 
NVCC  	:= /usr/local/cuda/bin/nvcc
NVFLAGS	:=

# Include paths for SISCI and CUDA
INC_DIS	:= /opt/DIS/include /opt/DIS/include/dis 
INC_NV	:= /usr/local/cuda/include

# Link paths
LD_DIS	:= /opt/DIS/lib64
LD_NV	:=

# Libraries
LIB_DIS	:= sisci
LIB_NV	:= cuda


libs += $(addprefix -L,$(LD_PATH)) $(addprefix -l,$(LD_LIBS))

define use_cuda
incs += $(addprefix -I,$(INC_NV))
libs += $(addprefix -L,$(LD_NV)) $(addprefix -l,$(LIB_NV))
defs += -DCUDA_ENABLED
endef

define use_sisci
incs += $(addprefix -I,$(INC_DIS))
libs += $(addprefix -L,$(LD_NV)) $(addprefix -l,$(LIB_NV))
defs += -DSISCI_ENABLED
endef


# Makefile targets
ifeq ($(filter $(notdir $(shell pwd)),$(DIRS)),)
.PHONY: all clean $(DIRS)

all: $(DIRS)

$(DIRS):
	$(MAKE) -C $@ 

clean: $(addprefix clean-,$(DIRS))

clean-%: %
	$(MAKE) -C $< clean

endif


# How to compile different source code files
%.o: %.cu
		$(NVCC) -std=c++11 --compiler-options "$(CFLAGS)" $(incs) -o $@ $< -c $(defs)

%.o: %.cpp
		$(CC) -x c++ -std=gnu++11 $(CFLAGS) $(incs) -o $@ $< -c $(defs)

%.o: %.c
		$(CC) -x c -std=gnu11 $(CFLAGS) $(incs) -o $@ $< -c $(defs)

