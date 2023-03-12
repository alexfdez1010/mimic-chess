#!/bin/sh

DATASET_FILTERED="database_filtered"
DOWNLOAD_FOLDER="lichess_database"

INITIAL_MONTH=11
INITIAL_YEAR=2019
FINAL_MONTH=12
FINAL_YEAR=2019

WAITING_CONSTANT=5

download_and_filter() {
  year=$1
  month=$2
  min_elo=$3
  max_elo=$4

  if [ "$month" -lt 10 ]; then
      month="0$month"
  fi

  file="lichess_db_standard_rated_$year-$month.pgn"
  file_compressed="$file.zst"

  if [ -f "$file" ]; then
      echo "File $file already exists. Skipping download"
  else
      echo "No file $file found. Downloading from lichess"
      wget https://database.lichess.org/standard/"$file_compressed" --no-check-certificate
      zstd -d "$file_compressed"
      rm "$file_compressed"
  fi

  echo "Starting filtering games with elo between $min_elo and $max_elo from $file"

  python -u ../filter_games_by_elo.py "$file" "../$DATASET_FILTERED" --min-elo "$min_elo" --max-elo "$max_elo" && rm "$file"

  echo "Finished filtering games with elo between $min_elo and $max_elo from $file"
}

if [ $# -ne 2 ]; then
    echo "Usage: $0 min_elo max_elo"
    exit 1
fi

min_elo=$1
max_elo=$2

echo "Downloading games from $INITIAL_MONTH/$INITIAL_YEAR to $FINAL_MONTH/$FINAL_YEAR from lichess"

if [ ! -d download/"$DOWNLOAD_FOLDER" ]; then
    mkdir download/"$DOWNLOAD_FOLDER"
fi

cd download/"$DOWNLOAD_FOLDER" || exit

for year in $(seq "$INITIAL_YEAR" "$FINAL_YEAR"); do
    for month in $(seq "$INITIAL_MONTH" "$FINAL_MONTH"); do
      download_and_filter "$year" "$month" "$min_elo" "$max_elo" &
      echo "Downloading and filtering games from $month/$year..."
      sleep "$WAITING_CONSTANT" # To avoid errors of too many requests
    done
done

