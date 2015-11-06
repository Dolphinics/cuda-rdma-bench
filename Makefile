PINGOBJ := ping.o common.o
PONGOBJ := pong.o common.o
NVCC    := /usr/local/cuda/bin/nvcc
CC      := gcc
CFLAGS  := -Wall -Wextra -g
IPATH	:= /opt/DIS/include /opt/DIS/include/dis /usr/local/cuda/include
LPATH   := /opt/DIS/lib64

.PHONY: all clean ping pong

all: ping pong

ping: $(PINGOBJ) ping_main.o
	$(NVCC) -o $@ $^ -lsisci -L$(LPATH) -lcuda

pong: $(PONGOBJ) pong_main.o
	$(NVCC) -o $@ $^ -lsisci -L$(LPATH) -lcuda

clean:
	-$(RM) ping ping_main.o $(PINGOBJ)
	-$(RM) pong pong_main.o $(PONGOBJ)

%.o: %.cu
	$(NVCC) -std=c++11 --compiler-options "$(CFLAGS)" $(addprefix -I,$(IPATH)) -o $@ $< -c

%.o: %.cpp
	$(CC) -x c++ -std=gnu++11 $(CFLAGS) $(addprefix -I,$(IPATH)) -o $@ $< -c

%.o: %.c
	$(CC) -x c -std=gnu11 $(CFLAGS) $(addprefix -I,$(IPATH)) -o $@ $< -c

# TODO: Do this more elegantly
ping_main.o: main.c
	$(CC) -x c -std=gnu11 $(CFLAGS) $(addprefix -I,$(IPATH)) -o $@ $< -c -DPING
	
pong_main.o: main.c
	$(CC) -x c -std=gnu11 $(CFLAGS) $(addprefix -I,$(IPATH)) -o $@ $< -c -DPONG
