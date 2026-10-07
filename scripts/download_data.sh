#!/bin/bash
set -e

# Configuration
DATA_DIR="./data/speech_commands_v0.02"
ARCHIVE_NAME="speech_commands_v0.02.tar.gz"
URL="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

echo "Creating data directory: $DATA_DIR"
mkdir -p "$DATA_DIR"

if [ -f "$DATA_DIR/$ARCHIVE_NAME" ]; then
    echo "Archive $ARCHIVE_NAME already exists in $DATA_DIR."
else
    echo "Downloading Speech Commands v0.02 dataset..."
    curl -L "$URL" -o "$DATA_DIR/$ARCHIVE_NAME"
fi

echo "Extracting archive..."
tar -xzf "$DATA_DIR/$ARCHIVE_NAME" -C "$DATA_DIR"

echo "Dataset downloaded and extracted."
echo "Dataset Stats:"
echo "Total files: $(find "$DATA_DIR" -type f -name "*.wav" | wc -l)"
