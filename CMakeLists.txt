# The portable build: the synthesizer as a library, and votrax-say on top of it.
#
#   cmake -S . -B build/cmake
#   cmake --build build/cmake --config Release
#
# Works with MSVC, GCC and Clang, on Windows and on Unix-likes.  The Windows
# deliverables -- the NVDA add-on's DLLs and the GUI -- keep their own scripts
# (nvda-addon/package.py, gui-native/build.cmd), because both are Windows-only
# and one of them has to produce two architectures from one run.
cmake_minimum_required(VERSION 3.16)
project(votrax_native VERSION 1.2.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(VOTRAX_SOURCES
    src/votrax.c
    src/votrax_core.c
    src/votrax_filters.c
    src/votrax_rom.c
    src/ttv.c
    src/ttv_tables.c
)

# Static, and VOTRAX_STATIC so votrax.h decorates nothing: a program linking
# this carries the whole synthesizer and needs nothing beside it.
add_library(votrax STATIC ${VOTRAX_SOURCES})
target_include_directories(votrax PUBLIC src)
target_compile_definitions(votrax PUBLIC VOTRAX_STATIC)

if(MSVC)
    target_compile_options(votrax PRIVATE /W4)
    target_compile_definitions(votrax PRIVATE _CRT_SECURE_NO_WARNINGS)
    # The static CRT, as the GUI uses: the exe runs on a machine that never
    # installed a redistributable.
    set_property(TARGET votrax PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_compile_options(votrax PRIVATE -Wall -Wextra)
    find_library(MATH_LIBRARY m)
    if(MATH_LIBRARY)
        target_link_libraries(votrax PUBLIC ${MATH_LIBRARY})
    endif()
endif()

add_executable(votrax-say cli/votrax_say.c)
target_link_libraries(votrax-say PRIVATE votrax)
if(MSVC)
    target_compile_options(votrax-say PRIVATE /W4)
    target_compile_definitions(votrax-say PRIVATE _CRT_SECURE_NO_WARNINGS)
    set_property(TARGET votrax-say PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    target_compile_options(votrax-say PRIVATE -Wall -Wextra)
endif()

install(TARGETS votrax-say RUNTIME DESTINATION bin)
