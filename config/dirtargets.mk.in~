all opencl cuda:
	@for dir in $(SUBDIRS); do ${MAKE} -C $$dir $@; done

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

