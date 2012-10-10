.PHONY: all

SHOC_ROOT=../../..
-include $(SHOC_ROOT)/config/config.mk

SUBDIRS=@MPI_SUBDIRS@


all:
	@for dir in ${SUBDIRS}; do ${MAKE} -C $$dir $@; done

clean:
	@if test -n "$(SUBDIRS)"; then \
	    rev=""; for dir in $(SUBDIRS); do rev="$$dir $$rev"; done; \
	    for dir in $$rev; do ${MAKE} -C $$dir $@; done \
	fi

distclean:
	@if test -n "$(SUBDIRS)"; then \
	    rev=""; for dir in $(SUBDIRS); do rev="$$dir $$rev"; done; \
	    for dir in $$rev; do ${MAKE} -C $$dir $@; done \
	fi
	${RM} Makefile

