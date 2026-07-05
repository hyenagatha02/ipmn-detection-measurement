import os
import sys
import glob
import SimpleITK as sitk
import time
from utils import format_time

'''
bias field correction
- correct the intensity inhomogeneity (bias) often seen in MRI data, which can arise due to imperfections in the MRI scanner's magnetic field. 
- This bias can cause some regions of the image to appear darker or lighter than they should be, independent of the actual tissue properties.
'''

def BiasCorrection(fid, input_fpath, output_fpath):
    """
    Applies N4 Bias Field Correction to an MRI image.
    """
    # logs = []  # print logs for multiprocessing 
    if os.path.exists(output_fpath):
        # print(f"... Already exists : {output_fpath} ...")
        # return
        # logs = [f"[{fid}] Already exists : {output_fpath}"]
        return f"[{fid}] Already exists : {output_fpath}"  # logs
    
    start_time = time.time()
    
    # Load input image
    input_image = sitk.ReadImage(input_fpath)
    # sitk.WriteImage(input_image, output_fpath)
    # Convert to 32-bit float
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)

    # Create mask for bias correction
    mask = input_image > 25
    # Convert mask to 0 or 1
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Initialize N4 Bias Correction filter
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetMaximumNumberOfIterations([50, 50, 50, 50])
    n4.SetConvergenceThreshold(0.001)
    n4.SetBiasFieldFullWidthAtHalfMaximum(0.15)
    n4.SetWienerFilterNoise(0.01)
    n4.SetSplineOrder(3)
    n4.SetNumberOfControlPoints([4, 4, 16])

    # Apply bias correction
    output_image = n4.Execute(input_image, mask)
    # # extracts the bias field that was applied during correction
    # correction_field = n4.GetLogBiasFieldAsImage(input_image)

    # Save the corrected image
    sitk.WriteImage(output_image, output_fpath)
    
    elapsed_time = time.time() - start_time
    # print(f"... Successfully saved : \033[36m{output_fpath}\033[0m ({format_time(elapsed_time)}) ...")
    # logs.append(f"... Successfully saved : \033[36m{output_fpath}\033[0m ({format_time(elapsed_time)}) ...")  # for multiprocessing

    return f"[{fid}] Successfully saved : \033[36m{output_fpath}\033[0m ({format_time(elapsed_time)}) ..." # logs



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bias_correction.py input_fpath output_fpath")
        sys.exit(1)

    input_fpath = sys.argv[1]
    output_fpath = sys.argv[2]
    
    print(f"input_fpath : {input_fpath}")
    print(f"output_fpath : {output_fpath}")
    if not os.path.exists(os.path.dirname(output_fpath)):
        os.makedirs(os.path.dirname(output_fpath))
        print(f"(created output dir : {os.path.dirname(output_fpath)})")

    BiasCorrection(input_fpath, output_fpath)