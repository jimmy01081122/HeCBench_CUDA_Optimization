CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_70
LAUNCHER  =

program = main
source = main.cu shmem_kernels.cu
obj = main.o shmem_kernels.o

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wno-deprecated-gpu-targets -Xcompiler -Wall -arch=$(ARCH)
LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cu shmem_kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

shmem_kernels.o: shmem_kernels.cu shmem_kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1000
