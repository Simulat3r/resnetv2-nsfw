#!/bin/bash

scripts_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
base_dir="$(dirname "$scripts_dir")"
raw_data_dir="$base_dir/raw_data"
data_dir="$base_dir/data"

declare -a class_names=(
	"neutral"
	"drawings"
	"sexy"
	"porn"
	"hentai"
	)

train_dir="$data_dir/train"
mkdir -p "$train_dir"

echo "Copying image to the training folder"
for cname in "${class_names[@]}"
do
    classdir="$train_dir/$cname"
    find $classdir -type f  | xargs -n 100 -P 80 python3 $scripts_dir/filter.py
done