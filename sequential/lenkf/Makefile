CC = cc
PKG_CONFIG = pkg-config

UNAME := $(shell uname)

# PATH_MDB_MATRIX = $(HOME)/src/libmdb_matrix
PATH_MDB_MATRIX = ../libmdb_matrix/

# BLAS = veclib
BLAS = openblas

INCLUDE_DIR += -I$(PATH_MDB_MATRIX)
LDFLAGS += -L$(PATH_MDB_MATRIX)

LIBS += -lm -lgsl -lconfuse

# export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig"

ifeq ($(BLAS),openblas)
   DEFINES += -DOPENBLAS
   LDFLAGS += -Wl,-rpath,$(PATH_MDB_MATRIX)

   INCLUDE_DIR += $(shell $(PKG_CONFIG) --cflags openblas)
   LIBS += $(shell $(PKG_CONFIG) --libs openblas)
else ifeq ($(BLAS),veclib)
   DEFINES += -DVECLIB
   LIBS += -framework Accelerate
   INCLUDE_DIR += -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/Current/Frameworks/vecLib.framework/Headers
else
   $(error $(BLAS) not supported!)
endif


ifeq ($(UNAME),Linux)
   DEFINES += -DLINUX
else ifeq ($(UNAME),Darwin)
   DEFINES += -DOSX
else
$(error $(UNAME) not supported!)
endif

ELEM = double

MILD_WARNINGS = -Wall -Wstrict-prototypes
# The strict warning setup below is used by the FreeBSD people
STRICT_WARNINGS = -W -Wall -ansi -pedantic -Wbad-function-cast -Wcast-align \
                  -Wcast-qual -Wchar-subscripts -Winline \
                  -Wmissing-prototypes -Wnested-externs -Wpointer-arith \
                  -Wredundant-decls -Wshadow -Wstrict-prototypes -Wwrite-strings
CPPFLAGS = $(MILD_WARNINGS) $(INCLUDE_DIR)
CFLAGS = $(CPPFLAGS) -O3 -g

ifeq ($(ELEM),float)
   DEFINES += -DLENKF_FLOAT_ELEM
   LIBS += -lfftw3f -lmdb_matrix_s
else
ifeq ($(ELEM),double)
   DEFINES += -DLENKF_DOUBLE_ELEM
   LIBS += -lfftw3 -lmdb_matrix_d
else
endif
endif

# Uncomment the following to enamble profiling via gprof
#CFLAGS += -pg
#LDFLAGS += -pg

#DEFINES += $(DEFINES) -DNDEBUG

#DEFINES += -DHASH_FUNCTION=HASH_FNV -DHASH_EMIT_KEYS=3


################################################################################


BIN = lenkf ensemble_test randn_test blas_test randn_conv_test_new convmtx \
      compute_P_HT


all: $(BIN)

ensemble_test: ensemble.o ensemble_test.o randn.o util.o

randn_test: randn_test.o randn.o ensemble.o util.o

randn_conv_test_new: randn_conv_test_new.o randn.o ensemble.o

lenkf: lenkf.o lenkf_main.o ensemble.o randn.o util.o \
       lenkf_config.o arg_bundle.o edot_table.o

blas_test: blas_test.o randn.o util.o

convmtx: convmtx.o

compute_P_HT: compute_P_HT.o randn.o ensemble.o lenkf.o \
              arg_bundle.o edot_table.o util.o


%.o : %.c
	$(CC) -c $(CFLAGS) $(DEFINES) $< -o $@

% : %.o
	$(CC) $(LDFLAGS) $^ -o $@ $(LIBS)


################################################################################
# Adapted from Section 4.14 Generating Prerequisites Automatically of
# the reference manual for GNU make

DEPEND = $(patsubst %.c,%.d,$(wildcard *.c))

%.d: %.c
	@set -e; rm -f $@; \
        $(CC) -MM $(CPPFLAGS) $(DEFINES) $< > $@.$$$$; \
        sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
        rm -f $@.$$$$

include $(DEPEND)

################################################################################


clean:
	rm -f *.o *.d $(BIN)
