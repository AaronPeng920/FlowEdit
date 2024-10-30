#!/bin/bash

cd ../inters

find "." -maxdepth 1 -type f -exec rm -f {} +

find "." -mindepth 1 -type d | while read -r dir; do
    find "$dir" -type f -exec rm -f {} +
done

