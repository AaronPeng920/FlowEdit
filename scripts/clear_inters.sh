#!/bin/bash

cd ../inters || exit 1

subdirs=$(find "." -mindepth 1 -maxdepth 1 -type d)
for subdir in $subdirs; do
    rm -rf "$subdir"/*
    find "$subdir" -maxdepth 1 -name '.*' -exec rm -rf {} +
done
