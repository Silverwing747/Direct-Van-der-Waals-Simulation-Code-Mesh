# -*- mode: makefile-gmake -*-

TARGETS = \
CahnHilliardPrimal

ALL: $(TARGETS)

topdir := $(shell cd ../.. && pwd)
PETIGA_DIR ?= $(topdir)
include $(PETIGA_DIR)/lib/petiga/conf/variables
include $(PETIGA_DIR)/lib/petiga/conf/rules

makefile.ad = makefile.minifad

$(shell $(MAKE) -f $(makefile.ad) bootstrap)
include $(makefile.ad)
PETIGA_INCLUDE += $(AD_INCLUDE)

OPTFLAGS = -Ofast
CFLAGS   = $(OPTFLAGS)
CXXFLAGS = $(OPTFLAGS)
FFLAGS   = $(OPTFLAGS)

CahnHilliardPrimal: CahnHilliardPrimal.o CahnHilliardPrimalFAD.o | chkopts
	$(CLINKER) -o $@ $^ $(PETIGA_LIB)
	$(RM) -f $^
