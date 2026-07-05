import os
import glob
import shutil
import SimpleITK as sitk
from scipy.ndimage import binary_dilation
# import nibabel as nib
import numpy as np

__all__ = ['format_time', 'check_files', 'read_sitk', 'save_sitk', 'split_mask', 'dilate_mask', 'remove_mask']

def format_time(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.2f}s"

# def check_files(fpath):
#     if os.path.isdir(fpath): # folder
#         for f in os.listdir(fpath):
#             if ('@eaDir' in f):
#                 shutil.rmtree(os.path.join(fpath, f))
#                 print(f'(removed {os.path.join(fpath, f)})')
#             if ('.ipynb_checkpoints' in f):
#                 shutil.rmtree(os.path.join(fpath, f))
#                 print(f'(removed {os.path.join(fpath, f)})')
#             if ('Thumbs.db' in f):
#                 os.remove(os.path.join(fpath, f))
#                 print(f'(removed {os.path.join(fpath, f)})')
#     else: # file
#         if ('Thumbs.db' in fpath):
#             os.remove(fpath)
#             print(f'(removed {fpath})')

def check_files(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):  # bottom-up traversal
        # Remove unwanted directories
        for dirname in dirnames:
            dir_full_path = os.path.join(dirpath, dirname)
            if '@eaDir' in dirname or '.ipynb_checkpoints' in dirname:
                shutil.rmtree(dir_full_path)
                print(f'(Removed directory: {dir_full_path})')

        # Remove unwanted files
        for filename in filenames:
            file_full_path = os.path.join(dirpath, filename)
            if filename in ['Thumbs.db']:
                os.remove(file_full_path)
                print(f'(Removed file: {file_full_path})')


def read_sitk(file_path):
    """Read a NIfTI mask and return the image and array."""
    mask = sitk.ReadImage(file_path)
    mask_array = sitk.GetArrayFromImage(mask)
    return mask, mask_array

def save_sitk(mask_array, reference_mask, output_path):
    """Save a mask array as a NIfTI file with identical metadata."""
    mask_image = sitk.GetImageFromArray(mask_array)
    mask_image.CopyInformation(reference_mask)
    sitk.WriteImage(mask_image, output_path)
    print(f'\033[1;36m-- Saved mask to: {output_path} -\033[0m')  # {os.path.basename(output_path)}


def split_mask(fid, fpath, pred_fpath):

    '''Seperate predicted merged mask to bduct / pduct / pancreas / bile_in_pancreas splited mask'''
    save_bduct = os.path.join(f"{fpath}/{fid}_bduct.nii.gz")
    save_pduct = os.path.join(f"{fpath}/{fid}_pduct.nii.gz")
    save_pancreas = os.path.join(f"{fpath}/{fid}_pancreas.nii.gz")
    save_bile_in_pancreas = os.path.join(f"{fpath}/{fid}_bile_in_pancreas.nii.gz")

    if os.path.exists(save_bduct) and os.path.exists(save_pduct) and os.path.exists(save_pancreas) and os.path.exists(save_bile_in_pancreas):
        print("... Already have all splited masks ...")
        _, bduct_arr = read_sitk(save_bduct)
        print(f'[bduct]            {bduct_arr.shape} {np.unique(bduct_arr, return_counts=True)}')
        _, pduct_arr = read_sitk(save_pduct)
        print(f'[pduct]            {pduct_arr.shape} {np.unique(pduct_arr, return_counts=True)}')
        _, pancreas_arr = read_sitk(save_pancreas)
        print(f'[pancreas]         {pancreas_arr.shape} {np.unique(pancreas_arr, return_counts=True)}')
        _, bile_in_pancreas_arr = read_sitk(save_bile_in_pancreas)
        print(f'[bile_in_pancreas] {bile_in_pancreas_arr.shape} {np.unique(bile_in_pancreas_arr, return_counts=True)}')

        return save_bduct, save_pduct, save_pancreas, save_bile_in_pancreas

    assert os.path.exists(pred_fpath), print(f"there's no {pred_fpath}")
    
    pred_img, pred_arr = read_sitk(pred_fpath)
    print(f'[pred]             {pred_arr.shape} {np.unique(pred_arr, return_counts=True)}')

    pure_bduct = (pred_arr == 1)
    pure_bduct = pure_bduct.astype(np.uint8) # array([False,  True]) -> array([0, 1])
    print(f'[bduct]            {pure_bduct.shape} {np.unique(pure_bduct, return_counts=True)}')
    pure_pduct = (pred_arr == 2)
    pure_pduct = pure_pduct.astype(np.uint8) # array([False,  True]) -> array([0, 1])
    print(f'[pduct]            {pure_pduct.shape} {np.unique(pure_pduct, return_counts=True)}')
    pure_pancreas = (pred_arr == 3)
    pure_pancreas = pure_pancreas.astype(np.uint8) # array([False,  True]) -> array([0, 1])
    print(f'[pancreas]         {pure_pancreas.shape} {np.unique(pure_pancreas, return_counts=True)}')
    pure_bile_in_pancreas = (pred_arr == 4)
    pure_bile_in_pancreas = pure_bile_in_pancreas.astype(np.uint8) # array([False,  True]) -> array([0, 1])
    print(f'[bile_in_pancreas] {pure_bile_in_pancreas.shape} {np.unique(pure_bile_in_pancreas, return_counts=True)}')

    save_sitk(pure_bduct, pred_img, save_bduct)
    # print(f"-- Saved pure_bduct to : {save_bduct}")
    save_sitk(pure_pduct, pred_img, save_pduct)
    # print(f"-- Saved pure_bduct to : {save_pduct}")
    save_sitk(pure_pancreas, pred_img, save_pancreas)
    # print(f"-- Saved pure_bduct to : {save_pancreas}")
    save_sitk(pure_bile_in_pancreas, pred_img, save_bile_in_pancreas)
    # print(f"-- Saved pure_bduct to : {save_bile_in_pancreas}")

    # print("... Done split_mask ...")

    return save_bduct, save_pduct, save_pancreas, save_bile_in_pancreas


def dilate_mask(fid, duct_fpath, dilation_radius=(1, 1, 0)):  # 3d_ellipse_r1

    """Apply dilation to a NIfTI mask (SimpleITK)"""
    # sitk.BinaryDilate designed to only support 3d dilation
    # dilation_radius=(1, 1, 0) : 3x3 elliptical kernel (Expands by 1 voxel radius in x and y, none in z)
    
    # save_dilated = duct_fpath.replace('.nii.gz', '_dilated_3d_ellipse_r1.nii.gz')
    save_dilated = duct_fpath.replace('.nii.gz', '_dilated.nii.gz')
    if os.path.exists(save_dilated):
        print(f"... Already has dilated mask of {os.path.basename(duct_fpath)} ...")
        _, org_arr = read_sitk(duct_fpath)
        print(f'[original] {org_arr.shape} {np.unique(org_arr, return_counts=True)}')
        _, dilated_arr = read_sitk(save_dilated)
        print(f'[dilated]  {dilated_arr.shape} {np.unique(dilated_arr, return_counts=True)}')
        return save_dilated
    
    mask_img, mask_arr = read_sitk(duct_fpath)
    print(f'[original] {mask_arr.shape} {np.unique(mask_arr, return_counts=True)}')

    dilated_mask = sitk.BinaryDilate(mask_img, dilation_radius)
    dilated_arr = sitk.GetArrayFromImage(dilated_mask)
    print(f'[dilated]  {dilated_arr.shape} {np.unique(dilated_arr, return_counts=True)}')

    save_sitk(dilated_arr, mask_img, save_dilated)
    # print(f"-- Save dilated {suffix} mask : {save_dilated}")

    # print(f"... Done dilate_mask of {os.path.basename(duct_fpath)} ...")

    return save_dilated
    

def remove_mask(fid, fpath, bduct_fpath, pduct_fpath, bile_in_pancreas_fpath, pancreas_fpath, pancreas_T_fpath):

    def handle_cyst_and_dilation_overlap(panc_T_arr, bduct_arr, pduct_arr, bile_in_pancreas_arr, is_dilated):
        """Handles the overlap between cyst and dilated duct masks, 
        modifies the pancreas mask, and saves the modified masks."""

        # handle overlapped area (cyst & duct)
        cyst_arr = (panc_T_arr == 2) # .astype(np.uint8) # array([False,  True]) -> array([0, 1])

        # Compute overlaps
        bduct_overlap = (bduct_arr == 1) & cyst_arr
        pduct_overlap = (pduct_arr == 1) & cyst_arr
        bile_in_pancreas_overlap = (bile_in_pancreas_arr == 1) & cyst_arr
        total_overlap = bduct_overlap | pduct_overlap | bile_in_pancreas_overlap
        print(f' (bduct_overlap) {np.unique(bduct_overlap, return_counts=True)}')
        print(f' (pduct_overlap) {np.unique(pduct_overlap, return_counts=True)}')
        print(f' (bile_in_pancreas_overlap) {np.unique(bile_in_pancreas_overlap, return_counts=True)}')
        print(f' (total_overlap) {np.unique(total_overlap, return_counts=True)}')
        if not is_dilated:
            # print("Nothing should be overlapped ! ")
            assert np.unique(total_overlap) == np.array([False]), print(np.unique(total_overlap))
        
        # Save panc_T_arr (removed duct)
        panc_T_arr[total_overlap] = 1 # set cyst (2) to pancreas (1)
        print(f' [panc_T_rm_duct_arr]  {np.unique(panc_T_arr, return_counts=True)}')
        # Save cyst_arr (original)
        cyst_arr = cyst_arr.astype(np.uint8) # array([False,  True]) -> array([0, 1])
        print(f' [cyst_arr]            {np.unique(cyst_arr, return_counts=True)}')
        # Save cyst mask (removed duct)
        cyst_arr_rm_duct = (panc_T_arr == 2).astype(np.uint8)
        print(f' [cyst_rm_duct_arr]    {np.unique(cyst_arr_rm_duct, return_counts=True)}')
        if not is_dilated:
            assert np.array(cyst_arr).all() == np.array(cyst_arr_rm_duct).all()
        
        return panc_T_arr, cyst_arr, cyst_arr_rm_duct


    # cyst_folder = os.path.join(fpath, 'cyst')
    # if not os.path.exists(cyst_folder):
    #     os.makedirs(cyst_folder)
    # save_panc_T_rm_duct = os.path.join(fpath, f'{fid}_t2_pancreas_thresholded_{suffix}.nii.gz')  # f'{fid}_t2_pancreas_T_{suffix}.nii.gz'
    # save_cyst = os.path.join(cyst_folder, f'{fid}_t2_cyst.nii.gz')  # f'{fid}_cyst_T.nii.gz'
    # save_cyst_rm_duct = os.path.join(cyst_folder, f'{fid}_t2_cyst_{suffix}.nii.gz')  # f'{fid}_cyst_T_{suffix}.nii.gz'
    if '_dilated' in bduct_fpath:
        assert '_dilated' in pduct_fpath and '_dilated' in bile_in_pancreas_fpath, print(f"check duct mask path (bduct {os.path.basename(bduct_fpath)} | pduct {os.path.basename(pduct_fpath)} | bile_in_pancreas_fpath {os.path.basename(bile_in_pancreas_fpath)})")
        is_dilated = True
        save_cyst = os.path.join(fpath, f'{fid}_t2_cyst.nii.gz')
        save_cyst_rm_duct = os.path.join(fpath, f'{fid}_t2_cyst_rm_duct_dilated.nii.gz')
        save_panc_T_rm_duct = os.path.join(fpath, f'{fid}_t2_pancreas_thresholded_rm_duct_dilated.nii.gz')
        if os.path.exists(save_cyst) and os.path.exists(save_cyst_rm_duct) and os.path.exists(save_panc_T_rm_duct):
            print("... Already have cyst_rm_duct_dilated mask ...")
            print("... Already have pancreas_thresholded_rm_duct_dilated mask ...")
            _, cyst_rm_duct_arr = read_sitk(save_cyst_rm_duct)
            print(f'[cyst_rm_duct_arr]    {cyst_rm_duct_arr.shape} {np.unique(cyst_rm_duct_arr, return_counts=True)}')
            _, panc_T_rm_duct_arr = read_sitk(save_panc_T_rm_duct)
            print(f'[panc_T_rm_duct_arr]  {panc_T_rm_duct_arr.shape} {np.unique(panc_T_rm_duct_arr, return_counts=True)}')

            return save_cyst, save_cyst_rm_duct, save_panc_T_rm_duct
    else:
        assert '_dilated' not in pduct_fpath and '_dilated' not in bile_in_pancreas_fpath, print(f"check duct mask path (bduct {os.path.basename(bduct_fpath)} | pduct {os.path.basename(pduct_fpath)} | bile_in_pancreas_fpath {os.path.basename(bile_in_pancreas_fpath)})")
        is_dilated = False
        save_cyst = os.path.join(fpath, f'{fid}_t2_cyst.nii.gz')
        save_cyst_rm_duct = os.path.join(fpath, f'{fid}_t2_cyst_rm_duct.nii.gz')  # No Need
        save_panc_T_rm_duct = os.path.join(fpath, f'{fid}_t2_pancreas_thresholded_rm_duct.nii.gz')  # No Need
        if os.path.exists(save_cyst):
            print("... Already have cyst mask ...")
            _, cyst_arr = read_sitk(save_cyst)
            print(f'[cyst_arr]            {cyst_arr.shape} {np.unique(cyst_arr, return_counts=True)}')

            return save_cyst, save_cyst_rm_duct, save_panc_T_rm_duct

    _, bduct_arr = read_sitk(bduct_fpath)
    _, pduct_arr = read_sitk(pduct_fpath)
    _, bile_in_pancreas_arr = read_sitk(bile_in_pancreas_fpath)
    _, panc_arr = read_sitk(pancreas_fpath)
    panc_T_mask, panc_T_arr = read_sitk(pancreas_T_fpath)
    print(f'[bduct_arr]             {bduct_arr.shape}  {np.unique(bduct_arr, return_counts=True)}')
    print(f'[pduct_arr]             {pduct_arr.shape}  {np.unique(pduct_arr, return_counts=True)}')
    print(f'[bile_in_pancreas_arr]  {bile_in_pancreas_arr.shape}  {np.unique(bile_in_pancreas_arr, return_counts=True)}')
    print(f'[panc_arr]              {panc_arr.shape}  {np.unique(panc_arr, return_counts=True)}')
    print(f'[panc_T_arr]            {panc_T_arr.shape} {np.unique(panc_T_arr, return_counts=True)}')
    
    if len(np.unique(panc_T_arr)) > 2: # if cyst exists
        panc_T_arr_rm_duct, cyst_arr, cyst_arr_rm_duct = handle_cyst_and_dilation_overlap(
            np.copy(panc_T_arr), bduct_arr, pduct_arr, bile_in_pancreas_arr, is_dilated
        )
        # print(f'[cyst_arr]            {cyst_arr.shape} {np.unique(cyst_arr, return_counts=True)}')
        # print(f'[cyst_rm_duct_arr]    {cyst_rm_duct_arr.shape} {np.unique(cyst_rm_duct_arr, return_counts=True)}')
        # print(f'[panc_T_rm_duct_arr]  {panc_T_rm_duct_arr.shape} {np.unique(panc_T_rm_duct_arr, return_counts=True)}')

        if not is_dilated:
            assert not os.path.exists(save_cyst)
            save_sitk(cyst_arr, panc_T_mask, save_cyst)
        else:
            assert not os.path.exists(save_cyst_rm_duct)
            assert not os.path.exists(save_panc_T_rm_duct)
            save_sitk(panc_T_arr_rm_duct, panc_T_mask, save_panc_T_rm_duct)
            save_sitk(cyst_arr_rm_duct, panc_T_mask, save_cyst_rm_duct)
        
    return save_cyst, save_cyst_rm_duct, save_panc_T_rm_duct