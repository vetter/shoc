# CMake build artifacts
# Add these to your .gitignore if migrating to CMake

# CMake build directories
build/
build-*/
cmake-build-*/

# CMake cache and files
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
CTestTestfile.cmake
install_manifest.txt

# Generated config
config/config.h

# Build outputs (preserve autotools structure)
# bin/ is intentionally not ignored as it's the install location
lib/
*.a
*.so
*.dylib

# IDE files (CMake-generated)
.vscode/
.idea/
*.code-workspace

# macOS
.DS_Store
