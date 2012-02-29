SUBDIRS=src

all clean:
	@for dir in $(SUBDIRS); do (cd $$dir && ${MAKE} $@); done

distclean:
	@for dir in $(SUBDIRS); do (cd $$dir && ${MAKE} $@); done
	${RM} Makefile config/common.mk config/targets.mk config/config.h

