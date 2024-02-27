include(ExternalProject)

ExternalProject_Add(
        ext_msgpack-c
        PREFIX msgpack-c
        URL https://github.com/msgpack/msgpack-c/releases/download/cpp-3.3.0/msgpack-3.3.0.tar.gz
        URL_HASH SHA256=6e114d12a5ddb8cb11f669f83f32246e484a8addd0ce93f274996f1941c1f07b
        DOWNLOAD_DIR "${OPEN3D_THIRD_PARTY_DOWNLOAD_DIR}/msgpack"
        # do not configure
        CONFIGURE_COMMAND ""
        # do not build
        BUILD_COMMAND ""
        # do not install
        INSTALL_COMMAND ""
        )
ExternalProject_Get_Property( ext_msgpack-c SOURCE_DIR )
set( MSGPACK_INCLUDE_DIRS "${SOURCE_DIR}/include/" )
