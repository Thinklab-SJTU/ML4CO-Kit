# libstdc++fs is only required for std::filesystem on GCC < 9.
# Apple Clang / libc++ and GCC >= 9 do not need a separate library.
set(DREAMPLACE_STDCXXFS_LIBS "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
  set(DREAMPLACE_STDCXXFS_LIBS stdc++fs)
endif()
