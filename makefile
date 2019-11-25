#################
# Paths and Flags
#################
SHELL = /bin/bash
CC = g++ 
NVCC = nvcc -I $(CUDA_HOME)/include
TARGET_PATH = ./bin
TARGET_NAME_OPT = cueso
TARGET_NAME_DBG = dbg$(TARGET_NAME_OPT)
OBJ_PATH = ./obj
CFLAGS = -std=c++11
#CUDAFLAGS = --gpu-architecture=compute_35 --gpu-code=sm_35
CUDAFLAGS = 
DBG_CFLAGS = $(CFLAGS) -g -DDEBUG
OPT_CFLAGS = $(CFLAGS) -O3
CU_DBG_CFLAGS = $(CFLAGS) $(CUDAFLAGS) -g -DDEBUG
CU_OPT_CFLAGS = $(CFLAGS) $(CUDAFLAGS) -O3
LFLAGS = -lm

#################
# Source
#################

CPP_FILES = $(wildcard	src/phase_field/*.cpp \
						src/utils/*.cpp \
						src/base/*.cpp \
						)

CU_FILES = $(wildcard	src/phase_field/*.cu \
						src/utils/*.cu \
						src/base/*.cu \
						)

#################
# Lists
#################

TEMP_LIST_OPT = $(CPP_FILES:%=$(OBJ_PATH)/opt/%)
TEMP_LIST_DBG = $(CPP_FILES:%=$(OBJ_PATH)/dbg/%)
OBJECTS_OPT = $(TEMP_LIST_OPT:%.cpp=%.o)
OBJECTS_DBG = $(TEMP_LIST_DBG:%.cpp=%.o)
DEPS_OPT = $(TEMP_LIST_OPT:%.cpp=%.d)
DEPS_DBG = $(TEMP_LIST_DBG:%.cpp=%.d)
	
TEMP_LIST_CU_OPT = $(CU_FILES:%=$(OBJ_PATH)/opt/%)
TEMP_LIST_CU_DBG = $(CU_FILES:%=$(OBJ_PATH)/dbg/%)
OBJECTS_CU_OPT = $(TEMP_LIST_CU_OPT:%.cu=%.o)
OBJECTS_CU_DBG = $(TEMP_LIST_CU_DBG:%.cu=%.o)
DEPS_CU_OPT = $(TEMP_LIST_CU_OPT:%.cu=%.d)
DEPS_CU_DBG = $(TEMP_LIST_CU_DBG:%.cu=%.d)

.DELETE_ON_ERROR:

# ---------------------------------------------------------------------------------------------
#  TARGET RULES
# ---------------------------------------------------------------------------------------------

usage:
	@echo ""
	@echo "Usage:"
	@echo "  make usage   (to see this info)"
	@echo "  make clean   (to delete all the .o files)"
	@echo "  make dbg     (to build a debug version)"
	@echo "  make opt     (to build an optimized version)"
	@echo ""

dbg : $(TARGET_PATH)/$(TARGET_NAME_DBG)

opt : $(TARGET_PATH)/$(TARGET_NAME_OPT)

# This rule makes the optimized binary by using mpicxx with the optimized ".o" files
$(TARGET_PATH)/$(TARGET_NAME_OPT) : partialcleanopt $(OBJECTS_OPT) $(OBJECTS_CU_OPT)
	$(NVCC) $(CUDAFLAGS) -o $(TARGET_PATH)/$(TARGET_NAME_OPT) $(OBJECTS_OPT) $(OBJECTS_CU_OPT) $(LFLAGS)

# This rule makes the debug binary by using mpicxx with the debug ".o" files
$(TARGET_PATH)/$(TARGET_NAME_DBG) : partialcleandbg $(OBJECTS_DBG) $(OBJECTS_CU_DBG)
	$(NVCC) $(CUDAFLAGS) -o $(TARGET_PATH)/$(TARGET_NAME_DBG) $(OBJECTS_DBG) $(OBJECTS_CU_DBG) $(LFLAGS)

partialcleandbg :
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	@rm -f $(TARGET_PATH)/$(TARGET_NAME_DBG)

partialcleanopt :
	@if [ ! -d "$(TARGET_PATH)" ]; then mkdir -p "$(TARGET_PATH)"; fi
	@rm -f $(TARGET_PATH)/$(TARGET_NAME_OPT)

clean : partialcleandbg partialcleanopt
	rm -f $(OBJECTS_OPT)
	rm -f $(OBJECTS_DBG)
	rm -f $(OBJECTS_CU_OPT)
	rm -f $(OBJECTS_CU_DBG)
	rm -f $(DEPS_OPT)
	rm -f $(DEPS_DBG)
	rm -f $(DEPS_CU_OPT)
	rm -f $(DEPS_CU_DBG)

# ---------------------------------------------------------------------------------------------
#  DEPENDENCIES FOR .CPP FILES
# ---------------------------------------------------------------------------------------------

# This includes all of the ".d" files. Each ".d" file contains a
# generated rule that tells it how to make .o files. (The reason these are generated is so that
# dependencies for these rules can be generated.)
-include $(DEPS_OPT)

-include $(DEPS_DBG)

# This rule makes the optimized ".d" files by using "mpicxx -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make an optimized ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_OPT) : $(OBJ_PATH)/opt/%.d : %.cpp
	@echo -e "Computing opt dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -en "$$(dirname $@)/" > $@
	@$(CC) $(OPT_CFLAGS) -MM $< >> $@
	@echo -e "	$(CC) $(OPT_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

# This rule makes the debug ".d" files by using "mpicxx -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a debug ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_DBG) : $(OBJ_PATH)/dbg/%.d : %.cpp
	@echo -e "Computing dbg dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -en "$$(dirname $@)/" > $@
	@$(CC) $(DBG_CFLAGS) -MM $< >> $@
	@echo -e "	$(CC) $(DBG_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

# ---------------------------------------------------------------------------------------------
#  DEPENDENCIES FOR .CU FILES
# ---------------------------------------------------------------------------------------------

# This includes all of the ".d" files. Each ".d" file contains a
# generated rule that tells it how to make .o files. (The reason these are generated is so that
# dependencies for these rules can be generated.)
-include $(DEPS_CU_OPT)

-include $(DEPS_CU_DBG)

# This rule makes the optimized ".d" files by using "mpicxx -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make an optimized ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_CU_OPT) : $(OBJ_PATH)/opt/%.d : %.cu
	@echo -e "Computing opt dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -en "$$(dirname $@)/" > $@
	@$(NVCC) $(CU_OPT_CFLAGS) -M $< >> $@
	@echo -e "	$(NVCC) $(CU_OPT_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@

# This rule makes the debug ".d" files by using "mpicxx -MM" with the corresponding ".cpp" file
# The ".d" file will contain a rule that says how to make a debug ".o" file.
# "$<" refers to the ".cpp" file, and "$@" refers to the ".d" file
$(DEPS_CU_DBG) : $(OBJ_PATH)/dbg/%.d : %.cu
	@echo -e "Computing dbg dependencies for $<"
	@-rm -f $$(dirname $@)/$$(basename $@ .d).o
	@if [ ! -d "$$(dirname $@)" ]; then mkdir -p "$$(dirname $@)"; fi
	@echo -en "$$(dirname $@)/" > $@
	@$(NVCC) $(CU_DBG_CFLAGS) -M $< >> $@
	@echo -e "	$(NVCC) $(CU_DBG_CFLAGS) -c $< -o $$(dirname $@)/$$(basename $@ .d).o" >> $@
