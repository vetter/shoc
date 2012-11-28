toolsdir		= $(prefix)/tools
libexec_SCRIPTS	= driver.pl numatest.pl
bin_SCRIPTS		= shocdriver
doc_DATA		= compilerVersion.txt buildFlags.txt

shocdriver:
	cat $(srcdir)/shocdriver.in | sed "s,SHOC_LIBEXEC_DIR,@libexecdir@," | sed "s,SHOC_BIN_DIR,@bindir@," > $@

compilerVersion.txt:
	${CXX} --version > $@

buildFlags.txt:
	cp ${top_builddir}/config/common.mk $@


