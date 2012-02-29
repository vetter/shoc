#!/bin/sh

echo "Bootstrapping SHOC build system."
aclocal || exit 1
autoconf || exit 1
echo "Done.  Now configure and make."

