TARGETS = \
P04_S13_NSCBC\

ALL: $(TARGETS)
clean::
	-@$(RM) $(TARGETS)

CFLAGS    = -fopenmp -O3#-Wno-unused-variable -Wunused-but-set-variable
FFLAGS    = -fopenmp -O3#-g3 -Wall -Wextra -fcheck=all
CPPFLAGS  = -fopenmp -O3#-Wno-unused-variable
FPPFLAGS  =
LOCDIR    = BubbleDissolved/
EXAMPLESC =
EXAMPLESF =
MANSEC    = IGA

topdir = /home/hu450/PetIGA
PETIGA_DIR ?= $(topdir)
include $(PETIGA_DIR)/lib/petiga/conf/variables
include $(PETIGA_DIR)/lib/petiga/conf/rules

P04_S13_NSCBC: P04_S13_NSCBC.o P04_S13_MatlabExport.o P04_S13_MatrixOperation.o P04_S13_Utility.o P04_S13_IC.o P04_S13_Residual.o P04_S13_Jacobian.o P04_S13_TIMFunc.o
	$(CLINKER) -o $@ P04_S13_NSCBC.o P04_S13_MatlabExport.o P04_S13_MatrixOperation.o P04_S13_Utility.o P04_S13_IC.o P04_S13_Residual.o P04_S13_Jacobian.o P04_S13_TIMFunc.o $(PETIGA_LIB) 
	$(RM) -f P04_S13_NSCBC.o P04_S13_MatlabExport.o P04_S13_MatrixOperation.o P04_S13_Utility.o P04_S13_IC.o P04_S13_Residual.o P04_S13_Jacobian.o P04_S13_TIMFunc.o

include $(PETIGA_DIR)/lib/petiga/conf/test
