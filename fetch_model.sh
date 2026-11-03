#!/bin/sh
# Fetch the RIFE v4.6 ncnn model (flownet.param/.bin) and, optionally, the rife-ncnn-vulkan binary used for comparison.
set -e
cd "$(dirname "$0")"
mkdir -p ext
if [ ! -f ext/rife-ncnn-vulkan/models/rife-v4.6/flownet.bin ]; then
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/nihui/rife-ncnn-vulkan.git ext/rife-ncnn-vulkan
  git -C ext/rife-ncnn-vulkan sparse-checkout set models/rife-v4.6
fi
echo "model: ext/rife-ncnn-vulkan/models/rife-v4.6"
if [ "$1" = "--ncnn-bin" ]; then
  mkdir -p ext/ncnn-bin && cd ext/ncnn-bin
  curl -sSLO https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-ubuntu.zip
  unzip -qo rife-ncnn-vulkan-20221029-ubuntu.zip
  echo "ncnn binary: ext/ncnn-bin/rife-ncnn-vulkan-20221029-ubuntu"
fi
