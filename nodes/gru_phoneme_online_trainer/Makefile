PROJECT=gru_phoneme_online_trainer

ROOT ?=../..
include $(ROOT)/setenv.mk

# manually set path for conda environment
CONDA_PREFIX=$(shell conda info --base)/envs/brand-speech

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
