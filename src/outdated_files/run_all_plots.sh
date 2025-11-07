#!/usr/bin/env bash

for folder in */; do
    # Remove trailing slash
    folder_no_slash="${folder%/}"

    # Check if folder name starts with one of the specified prefixes
    if [[ $folder_no_slash == adult* || \
          $folder_no_slash == banks* || \
          $folder_no_slash == compas* || \
          $folder_no_slash == german* ]]; then

        # Check if folder name contains "grad"
        if [[ $folder_no_slash == *Grad* ]]; then
            echo "Running on $folder_no_slash with --expgrad True"
            python plots.py --path "$folder_no_slash" --expgrad True
        else
            echo "Running on $folder_no_slash (no --expgrad)"
            python plots.py --path "$folder_no_slash"
        fi
    fi
done


