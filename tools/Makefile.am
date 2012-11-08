toolsdir		= $(prefix)/tools
tools_SCRIPTS	= driver.pl
tools_DATA		= compilerVersion.txt buildFlags.txt

compilerVersion.txt:
	${CXX} --version > $@

buildFlags.txt:
	cp ${top_builddir}/config/common.mk $@

clean:
	${RM} compilerVersion.txt buildFlags.txt

