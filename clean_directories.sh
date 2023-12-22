#!/bin/bash

# Directories to delete files in
DIRECTORIES=(
    "/home/users/shreshtha.singh/qxr_ln/shreshtha_notebooks/syntheticxrayproject-modified/numpy_nodules"
    "/home/users/shreshtha.singh/qxr_ln/shreshtha_notebooks/syntheticxrayproject-modified/chestXRays"
    "/home/users/shreshtha.singh/qxr_ln/shreshtha_notebooks/syntheticxrayproject-modified/textCTs"
    "/home/users/shreshtha.singh/qxr_ln/shreshtha_notebooks/syntheticxrayproject-modified/textNodules"
    "/home/users/shreshtha.singh/qxr_ln/shreshtha_notebooks/syntheticxrayproject-modified/textXRays"
)

# Function to clean directories
clean_directories() {
    for dir in "${DIRECTORIES[@]}"; do
        echo "Deleting files in $dir"
        rm -rf "$dir"/*
    done
}

# Main function
main() {
    clean_directories
}

# Run the script
main
