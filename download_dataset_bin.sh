#!/usr/bin/env bash

DATASET_FOLDER = "./dataset_bin"

if [[ ! -d "$DATASET_FOLDER" ]]; then
  mkdir dataset_bin
fi

if [[ ! -f "$DATASET_FOLDER/cifar-10-binary.tar.gz" ]]; then
  wget --directory-prefix "$DATASET_FOLDER" https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-binary.tar.gz
fi

tar xf "$DATASET_FOLDER/cifar-10-binary.tar.gz" --directory "$DATASET_FOLDER" --strip-components=1

rm "$DATASET_FOLDER/cifar-10-binary.tar.gz"
