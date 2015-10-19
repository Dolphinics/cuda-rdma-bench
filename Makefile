OBJECTS := main.o 
TARGET  := gpudirect
NVCC    := nvcc
CC      := gcc
CFLAGS  := -Wall -Wextra
IPATH	:= /opt/DIS/include /opt/DIS/include/dis $(CUDA_HOME)/targets/x86_64-linux/include
LPATH   := /opt/DIS/lib64

.PHONY: $(TARGET) all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) -o $@ $^ -lsisci -L$(LPATH) -lcuda

clean:
	-$(RM) $(OBJECTS) $(TARGET)

%.o: %.cu
	$(NVCC) -std=c++11 --compiler-options "$(CFLAGS)" $(addprefix -I,$(IPATH)) -o $@ $< -c

%.o: %.cpp
	$(CC) -x c++ -std=c++11 $(CFLAGS) $(addprefix -I,$(IPATH)) -o $@ $< -c

