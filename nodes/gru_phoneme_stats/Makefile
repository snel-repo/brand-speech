PROJECT=gru_phoneme_stats

ROOT ?=../..
include $(ROOT)/setenv.mk

# manually set path for conda environment
CONDA_ENV_EXISTS := $(shell if [ -d $(shell conda info --base)/envs/brand-speech ]; then echo 1; else echo 0; fi)

# Set CONDA_PREFIX based on the existence of the conda environment
ifeq ($(CONDA_ENV_EXISTS),1)
    CONDA_PREFIX := $(shell conda info --base)/envs/brand-speech
else
    CONDA_PREFIX := /home/envs/brand-speech
endif

PYTHON_VERSION=3.9 # This works for brand-speech env
PYTHON_LIB=python$(PYTHON_VERSION)

LIBPYTHON=$(CONDA_PREFIX)/lib/
INCPYTHON=$(CONDA_PREFIX)/include/$(PYTHON_LIB)

TARGET=$(PROJECT).bin
CYTHON_TARGET=$(GENERATED_PATH)/$(PROJECT).c

all:
	cp $(PROJECT).py $(PROJECT).pyx
	cython --embed $(PROJECT).pyx -o $(CYTHON_TARGET)
	gcc $(CYTHON_TARGET) -o $(TARGET) -I $(INCPYTHON) -L $(LIBPYTHON)  -Wl,-rpath=$(LIBPYTHON) -l$(PYTHON_LIB) -lpthread -lm -lutil -ldl
	$(RM) $(PROJECT).pyx
clean:
	$(RM) $(CYTHON_TARGET) $(PROJECT).pyx
