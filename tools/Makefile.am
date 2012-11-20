toolsdir		= $(prefix)/tools
libexec_SCRIPTS	= driver.pl numatest.pl
bin_SCRIPTS		= shocdriver
doc_DATA		= compilerVersion.txt buildFlags.txt

shocdriver:
	cat $(srcdir)/shocdriver.in | sed "s,SHOC_INST_DIR,@libexecdir@," > $@

compilerVersion.txt:
	${CXX} --version > $@

buildFlags.txt:
	cp ${top_builddir}/config/common.mk $@


