#!/bin/bash

# Prefer explicit ANDROID_NDK; fallback to common env vars if unset.
if [ -z "$ANDROID_NDK" ]; then
  if [ -n "$ANDROID_NDK_HOME" ]; then
    ANDROID_NDK="$ANDROID_NDK_HOME"
  elif [ -n "$ANDROID_NDK_ROOT" ]; then
    ANDROID_NDK="$ANDROID_NDK_ROOT"
  elif [ -n "$ANDROID_SDK_ROOT" ] && [ -d "$ANDROID_SDK_ROOT/ndk-bundle" ]; then
    ANDROID_NDK="$ANDROID_SDK_ROOT/ndk-bundle"
  fi
fi

if [ -z "$ANDROID_NDK" ] || [ ! -d "$ANDROID_NDK" ]; then
  echo "ERROR: ANDROID_NDK is not set or points to an invalid directory." >&2
  echo "Set ANDROID_NDK (or ANDROID_NDK_HOME/ANDROID_NDK_ROOT) to your Android NDK path." >&2
  exit 1
fi

cmake ../../../ \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_static \
-DMNN_USE_LOGCAT=false \
-DMNN_BUILD_BENCHMARK=ON \
-DMNN_USE_SSE=OFF \
-DMNN_BUILD_TEST=ON \
-DANDROID_NATIVE_API_LEVEL=android-21  \
-DMNN_BUILD_FOR_ANDROID_COMMAND=true \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $*

make -j4
