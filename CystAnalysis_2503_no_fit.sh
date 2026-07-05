#!/bin/bash
echo "* Running on GPU"


# Set the database directory (argument 1)
DATABASE_DIR="$1"
FID="$2"
CUDA_DEVICE="$3"
# Validate required arguments
if [ -z "$DATABASE_DIR" ] || [ -z "$FID" ]; then
    echo -e "\033[1;31m[ERROR] Missing arguments! Usage: $0 [DATABASE_DIR] [FID] [CUDA_DEVICE]\033[0m"
    exit 1
fi

# Set the CUDA device
if [ -z "$CUDA_DEVICE" ]; then
    echo "[WARNING] No CUDA device provided. Defaulting to CPU mode."
    CUDA_DEVICE="CPU"  # Set to "0" if you want to default to GPU 0 instead
fi

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
echo "* CUDA Device Set to: $CUDA_VISIBLE_DEVICES"


# Define the folder path for mask_statistics
MASK_STATS_DIR="$DATABASE_DIR/$FID/mask_statistics"
# Create the mask_statistics directory if it does not exist
if [ ! -d "$MASK_STATS_DIR" ]; then
    mkdir -p "$MASK_STATS_DIR"
fi


# Define Executable File Name (Linux)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINUX_EXEC="$SCRIPT_DIR/HistogramAnalysis_final_2503_no_fit"
# Check if the Linux executable exists and is executable
if [ ! -x "$LINUX_EXEC" ]; then
    echo -e "\033[1;31mError: $LINUX_EXEC not found or not executable!\033[0m"
    exit 1
fi



# Run the HistogramAnalysis.exe (Ensure it's executable)
# * Order of files : imageFile   maskFile   scaledImageFile   thresImageFile   csv_filename
# * Files (corr_image and output files) need to be in same folder

# Check if Required Input Files Exist
for file in "$DATABASE_DIR/$FID/${FID}_t2_corr.nii.gz" \
            "$DATABASE_DIR/$FID/${FID}_pancreas.nii.gz" \
            "$DATABASE_DIR/$FID/${FID}_bduct.nii.gz" \
            "$DATABASE_DIR/$FID/${FID}_pduct.nii.gz"
do
    if [ ! -f "$file" ]; then
        echo -e "\033[1;31m[ERROR] Missing input file: $file\033[0m"
        exit 1
    fi
done


echo "[INFO] Running : $LINUX_EXEC..."
$LINUX_EXEC "$FID" \
            "$DATABASE_DIR/$FID/${FID}_t2_corr.nii.gz" \
            "$DATABASE_DIR/$FID/${FID}_pancreas.nii.gz" \
            "$DATABASE_DIR/$FID/${FID}_bduct.nii.gz" \
            "$DATABASE_DIR/$FID/${FID}_pduct.nii.gz" \
            "$DATABASE_DIR/$FID/${FID}_t2_pancreas_scaled.nii.gz" \
            "$DATABASE_DIR/$FID/${FID}_t2_pancreas_thresholded.nii.gz" \
            "$MASK_STATS_DIR/${FID}_t2_pancreas_histogram.csv"

exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo -e "\033[1;31mError: The process failed with exit code $exit_code\033[0m"
    exit $exit_code
else
    echo -e "\033[1;32mSuccess: Process completed successfully!\033[0m"
    exit 0
fi