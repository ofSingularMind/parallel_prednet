#!/bin/bash

# Mount the SSHFS (assuming key-based auth is set up)
sshfs aledbetter@login.delftblue.tudelft.nl:/scratch/aledbetter/thesis_data /home/evalexii/remote_dataset

# Check if mount was successful
if mount | grep -q "/home/evalexii/remote_dataset"; then
    # Run the Python script
    python /home/evalexii/Documents/Thesis/code/parallel_prednet/testing.py
    # Unmount the SSHFS after the script finishes
    fusermount -u /home/evalexii/remote_dataset
else
    echo "Failed to mount SSHFS."
fi
