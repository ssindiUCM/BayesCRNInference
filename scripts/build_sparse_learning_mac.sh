#!/bin/bash
# Build Zhang et al.'s sparse_learning program on this Mac without Homebrew.
# Their source is copied unchanged from Dropbox; the libconfig library it needs is downloaded and compiled
# into the program directly. Result: ~/sparse-learning-build/sparse_learning
set -e
SRC="$HOME/Dropbox/WritingProjects/2026_06_08_CRNWork/sparse-learning-CRN/src"
B="$HOME/sparse-learning-build"
command -v clang++ >/dev/null || { echo "No C++ compiler. Run: xcode-select --install   (then run this script again)"; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$B"; mkdir -p "$B"; cd "$B"

echo "1/3 unpacking libconfig 1.7.3"
LOCAL_TGZ="$SCRIPT_DIR/vendor/libconfig-1.7.3.tar.gz"     # copy kept in the repo
if [ -f "$LOCAL_TGZ" ]; then cp "$LOCAL_TGZ" libconfig.tgz
else curl -sSL -o libconfig.tgz https://github.com/hyperrealm/libconfig/archive/refs/tags/v1.7.3.tar.gz; fi
tar xzf libconfig.tgz
L="$B/libconfig-1.7.3/lib"

echo "2/3 compiling libconfig"
for f in grammar.c libconfig.c scanctx.c scanner.c strbuf.c strvec.c util.c wincompat.c; do
  clang -O2 -DLIBCONFIG_STATIC -DHAVE_NEWLOCALE -DHAVE_USELOCALE -DHAVE_FREELOCALE -w -c "$L/$f" -o "lc_${f%.c}.o"
done
clang++ -O2 -DLIBCONFIG_STATIC -DLIBCONFIGXX_STATIC -w -c "$L/libconfigcpp.cc" -o lc_libconfigcpp.o

echo "3/3 compiling sparse_learning (their source, unchanged)"
cp "$SRC"/*.cpp "$SRC"/*.h .
for f in sparse_learning param config initialize utils; do
  clang++ -O2 -DDEBUG -DLIBCONFIG_STATIC -DLIBCONFIGXX_STATIC -I"$L" -w -c "$f.cpp" -o "$f.o"
done
clang++ -O2 sparse_learning.o param.o config.o initialize.o utils.o lc_*.o -o sparse_learning
echo "Built $B/sparse_learning"
