#!/bin/sh

# This script execute the complete workflow for training a model
# It takes as input the folder where the games are stored and the name of the model
# It also takes as input the minimum and maximum elo of the games to be used for training

DATASET_FILTERED="dataset_filtered"
DATASET="dataset"

if [ -z "$folder" ] || [ -z "$name_model" ]; then
    echo "Usage: $0 folder name_model [min_elo] [max_elo]"
    exit 1
fi

if [ ! -d "$folder" ]; then
    echo "Folder $folder does not exist"
    exit 1
fi

folder=$1
name_model=$2

if [ -z "$3" ]; then
    min_elo=0
else
    min_elo=$3
fi

if [ -z "$4" ]; then
    max_elo=3000
else
    max_elo=$4
fi

echo "Filtering games with elo between $min_elo and $max_elo from $folder"

python preprocessing/filter_games.py "$folder" "$DATASET_FILTERED" --min-elo "$min_elo" --max-elo "$max_elo"

echo "Generating dataset from $DATASET_FILTERED"

python preprocessing/games_to_positions.py "$DATASET_FILTERED" "$DATASET"

echo "Training model $name_model"

python training.py "$name_model"



