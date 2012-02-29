all opencl cuda:
	@for dir in $(SUBDIRS); do (cd $$dir && $(MAKE) $@) || exit 1; done

clean:
	@if test -n "$(SUBDIRS)"; then \
	  rev=""; for dir in $(SUBDIRS); do rev="$$dir $$rev"; done; \
	  for dir in $$rev; do (cd $$dir && $(MAKE) $@); done \
	fi

distclean:
	@if test -n "$(SUBDIRS)"; then \
	  rev=""; for dir in $(SUBDIRS); do rev="$$dir $$rev"; done; \
	  for dir in $$rev; do (cd $$dir && $(MAKE) $@); done \
	fi
	${RM} Makefile

