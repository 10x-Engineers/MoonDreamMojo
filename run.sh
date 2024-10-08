#!/bin/bash

# Run the first Mojo file
magic run mojo image_encoder.mojo

# Check if the first command was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to run main.mojo"
  exit 1
fi

# Run the second Mojo file
magic run mojo llm.mojo

# Check if the second command was successful
if [ $? -ne 0 ]; then
  echo "Error: Failed to run llm.mojo"
  exit 1
fi

echo "Both files executed successfully!"
