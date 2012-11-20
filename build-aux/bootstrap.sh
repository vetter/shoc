#!/bin/sh

echo "Bootstrapping SHOC build system."
aclocal || exit 1
automake --foreign --add-missing --copy || exit 1
autoconf || exit 1
echo "Done.  Now configure, make, and make install."

