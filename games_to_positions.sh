#!/bin/sh

SPLIT=0.99
TRAINING_FOLDER="training"
VALIDATION_FOLDER="validation"

if [ $# -ne 2 ]; then
    echo "Usage: $0 input_folder output_folder"
    exit 1
fi

input_folder=$1
output_folder=$2

if [ ! -d "$output_folder" ]; then
    mkdir "$output_folder"
fi

if [ ! -d "$output_folder"/"$TRAINING_FOLDER" ]; then
    mkdir "$output_folder"/"$TRAINING_FOLDER"
fi

if [ ! -d "$output_folder"/"$VALIDATION_FOLDER" ]; then
    mkdir "$output_folder"/"$VALIDATION_FOLDER"
fi

for file in "$input_folder"/*.pgn; do
    echo "Starting converting games from $file to positions"
    python -u games_to_positions.py "$file" "$output_folder" --split "$SPLIT" &
done