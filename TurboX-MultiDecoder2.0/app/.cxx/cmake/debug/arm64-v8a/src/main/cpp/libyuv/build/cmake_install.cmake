# Install script for directory: /home/yuandekun/Flamingo Demo/App_dev/TurboX-MultiDecoder2.0/app/src/main/cpp/libyuv/libyuv

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE PROGRAM FILES "/home/yuandekun/Flamingo Demo/App_dev/TurboX-MultiDecoder2.0/app/.cxx/cmake/debug/arm64-v8a/yuvconvert")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/yuandekun/Flamingo Demo/App_dev/TurboX-MultiDecoder2.0/app/.cxx/cmake/debug/arm64-v8a/src/main/cpp/libyuv/build/libyuv.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/yuandekun/Flamingo Demo/App_dev/TurboX-MultiDecoder2.0/app/libs/arm64-v8a/libyuv.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libyuv.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libyuv.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/home/yuandekun/Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libyuv.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/home/yuandekun/Flamingo Demo/App_dev/TurboX-MultiDecoder2.0/app/src/main/cpp/libyuv/libyuv/include/")
endif()

