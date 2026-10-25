cmake_minimum_required(VERSION 3.16)
project(vortexpak_engine_loader C)

# Path to the VortexPak native library
set(VORTEXPAK_NATIVE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vortexpak_native")

add_executable(engine_loader engine_loader.c)

target_include_directories(engine_loader PRIVATE ${VORTEXPAK_NATIVE_DIR})

# Link against the shared library
if(WIN32)
    target_link_libraries(engine_loader
        ${VORTEXPAK_NATIVE_DIR}/target/release/vortexpak_native.dll.lib)
    # Copy DLL next to executable for easy running
    add_custom_command(TARGET engine_loader POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${VORTEXPAK_NATIVE_DIR}/target/release/vortexpak_native.dll
        $<TARGET_FILE_DIR:engine_loader>)
else()
    target_link_libraries(engine_loader
        ${VORTEXPAK_NATIVE_DIR}/target/release/libvortexpak_native.so)
    set_target_properties(engine_loader PROPERTIES
        BUILD_RPATH "${VORTEXPAK_NATIVE_DIR}/target/release")
endif()
