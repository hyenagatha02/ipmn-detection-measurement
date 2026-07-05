import os
import glob
import shutil
import argparse

import time
import gc
import dicom2nifti
from pydicom.tag import Tag
import dicom2nifti.common as common

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import check_files

'''
[docs_dicom2nifti] https://dicom2nifti.readthedocs.io/en/latest/
ITK-SNAP 사용시
- 폴더명 한글 포함 X
- dicom2nifti.convert_directory(input_directory, output_directory, compression=True, reorient=False) compression=True, reorient=False 설정값 필수
'''


def _is_float(value):
    """
    Check if the value is a float, returning False for None.
    """
    try:
        if value is None:  ## added
            return False
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _is_bval_type_a(grouped_dicoms):
    """
    Check if the bvals are stored in the first of 2 currently known ways for single frame DTI.
    """
    bval_tag = Tag(0x2001, 0x1003)  # b-value
    bvec_x_tag = Tag(0x2005, 0x10b0)  # x-direction
    bvec_y_tag = Tag(0x2005, 0x10b1)  # y-direction
    bvec_z_tag = Tag(0x2005, 0x10b2)  # z-direction

    for group in grouped_dicoms:
        # Skip empty groups
        if not group or not group[0]:  ## added
            continue

        # Check if all required tags are present and valid
        if (
            bvec_x_tag in group[0] and _is_float(common.get_fl_value(group[0][bvec_x_tag])) and
            bvec_y_tag in group[0] and _is_float(common.get_fl_value(group[0][bvec_y_tag])) and
            bvec_z_tag in group[0] and _is_float(common.get_fl_value(group[0][bvec_z_tag])) and
            bval_tag in group[0] and _is_float(common.get_fl_value(group[0][bval_tag])) and
            common.get_fl_value(group[0][bval_tag]) != 0
        ):
            return True

    return False


def _get_grouped_dicoms(dicom_input):
    """
    Search all DICOMs in the DICOM directory, sort and validate them.
    fast_read = True will only read the headers not the data

    + Handles cases where Tag(2001,100A) contains None.
    """
    # If all DICOMs have an instance number, try sorting by instance number; else sort by position
    if [d for d in dicom_input if 'InstanceNumber' in d]:
        dicoms = sorted(dicom_input, key=lambda x: x.InstanceNumber)
    else:
        dicoms = common.sort_dicoms(dicom_input)

    # Initialize grouped DICOMs and timepoint index
    grouped_dicoms = [[]]  # List with first element as a list
    timepoint_index = 0
    previous_stack_position = -1

    # Define the stack position tag
    stack_position_tag = Tag(0x2001, 0x100A)  # Philips "Slice Number MR" or private tag

    # Loop over all sorted DICOMs
    for index in range(0, len(dicoms)):
        dicom_ = dicoms[index]
        stack_position = 0

        # Check if the stack position tag exists and has a valid value
        if stack_position_tag in dicom_:
            tag_value = dicom_[stack_position_tag].value  ## added
            if isinstance(tag_value, (str, int)) and tag_value is not None:
                stack_position = common.get_is_value(dicom_[stack_position_tag])

        # Group slices based on stack position
        if previous_stack_position == stack_position:
            # If the stack position is the same, move to the next timepoint
            timepoint_index += 1
            if len(grouped_dicoms) <= timepoint_index:
                grouped_dicoms.append([])
        else:
            # If the stack position changes, reset to the first timepoint
            timepoint_index = 0

        grouped_dicoms[timepoint_index].append(dicom_)
        previous_stack_position = stack_position

    return grouped_dicoms

dicom2nifti.convert_philips._is_float = _is_float
dicom2nifti.convert_philips._is_bval_type_a = _is_bval_type_a
dicom2nifti.convert_philips._get_grouped_dicoms = _get_grouped_dicoms


def change_dicom_to_nifti(dicom_path, nifti_path):

    check_files(dicom_path)

    empty_dcms = []
    multiple_niftis = []
    existing_fids = []
    if os.path.exists(nifti_path):
        if len(os.listdir(nifti_path))!=0:
            # assert set(os.listdir(dicom_path)) == set([f.replace('_0000', '').replace('.nii.gz', '') for f in os.listdir(nifti_path)]), print(f"! dicom and nifti files don't match ! (dicom folders: {os.listdir(dicom_path)} | nifti files: {os.listdir(nifti_path)})")
            if set(os.listdir(dicom_path)) == set([f.replace('_0000', '').replace('.nii.gz', '') for f in os.listdir(nifti_path)]):
                print("! All dicom folders already exists in nifti files !")
                print(f"- dicom folders : {len(os.listdir(dicom_path))} {os.listdir(dicom_path)}")
                print(f"- nifti files   : {len(os.listdir(nifti_path))} {os.listdir(nifti_path)})")
                return empty_dcms, multiple_niftis
            
            print("! Some of dicom folders already exists in nifti files !")
            print(f"- dicom folders : {len(os.listdir(dicom_path))} {os.listdir(dicom_path)}")
            print(f"- nifti files   : {len(os.listdir(nifti_path))} {os.listdir(nifti_path)})")
            existing_fids = [f.replace('_0000.nii.gz', '') for f in os.listdir(nifti_path) if f.endswith('.nii.gz')]
            # print("> Ignore and generating nifti files from scratch")
            # shutil.rmtree(nifti_path)
            # os.makedirs(nifti_path)

    for idx, (fpath, folders, files) in enumerate(sorted(os.walk(dicom_path))):  # , start=1
        print()
        print(f"[{idx}]  ", fpath, folders, files)
        if fpath == dicom_path:  # model\dataset\inputImgs\dicom 
            continue
        assert len(os.listdir(fpath)) == len(files)

        fid = os.path.basename(fpath)    # model\dataset\inputImgs\dicom\case001

        # Check empty dicom folder
        if len(files)==0:
            print(f"\033[31m> Empty dicom files ... Skip !\033[0m")
            empty_dcms.append(fid)
            continue

        # check if there's nifti file
        if fid in existing_fids:
            print(f"> Already exists in nifti folder : \033[36m{[f for f in glob.glob(f'{nifti_path}/*') if fid in f]}\033[0m ... Skip !")
            continue

        # check if there's non-dicom files or folders
        _dcms = [f for f in files if not f.endswith('.dcm')]
        if _dcms:
            # No dicom files at all
            if len(files) == len(_dcms):
                print(f"\033[31m! It's non-dicom directory... Skip {fpath} !\033[0m")
                continue
            
            dcms = [f for f in files if f not in _dcms]
            
            print(f"\033[31m! There's non-dicom file or folder !\033[0m {len(_dcms)} {_dcms}")
            print(f'- Total files : {len(files)}')
            print(f'- dicoms      : {len(dcms)}')
            print(f'- non-dicoms  : {len(_dcms)} {_dcms}')

            for _f in _dcms:
                os.remove(os.path.join(fpath, _f))
                print(f"> remove : {os.path.join(fpath, _f)}")

        assert len(files) - len(_dcms) == len(os.listdir(fpath))
        print('\033[1m>> (%d dcm files) %s\033[0m' %(len(os.listdir(fpath)), fpath))

        # convert dicom to nifti
        save_fpath = os.path.join(nifti_path, fid)  # model\dataset\inputImgs\nifti\case001
        # assert save_fpath == fpath.replace('dicom', 'nifti')
        if not os.path.exists(save_fpath):
            os.makedirs(save_fpath, exist_ok=True)

        try:
            dicom2nifti.convert_directory(fpath, save_fpath, compression=True, reorient=False) # compression=True, reorient=False *essential*
        except Exception as e:
            print(f"Error during conversion: {e}")
        # assert len(os.listdir(save_fpath)) == 1  # ['401_cor_t2_ssh_fs.nii.gz']
        # save_fname = os.listdir(save_fpath)[0]  # 401_cor_t2_ssh_fs.nii.gz
        if len(os.listdir(save_fpath)) == 1:
            save_fname = os.listdir(save_fpath)[0]
            nii_created = os.path.join(save_fpath, save_fname)
            nii_moved = os.path.join(nifti_path, f"{fid}.nii.gz")
            print('> rename : %s -> \033[36m%s\033[0m' %(nii_created, nii_moved))  # nifti\case001\401_cor_t2_ssh_fs.nii.gz -> nifti\case001.nii.gz
            shutil.move(nii_created, nii_moved)
        else:
            print(f"\033[1;31m! Multiple nifti files created {(os.listdir(save_fpath))} !\033[0m")
            niis = []
            for i, save_fname in enumerate(os.listdir(save_fpath), start=1):
                created = os.path.join(save_fpath, save_fname)
                moved = os.path.join(nifti_path, f"{fid}_{i:02d}.nii.gz")
                niis.append((created, moved))
            print(niis)
            for nii in niis:
                nii_created, nii_moved = nii
                print('> rename : %s -> \033[36m%s\033[0m' %(nii_created, nii_moved))  # nifti\case001\401_cor_t2_ssh_fs.nii.gz -> nifti\case001.nii.gz
                shutil.move(nii_created, nii_moved)
                multiple_niftis.append(fid)
        assert len(os.listdir(save_fpath)) == 0
        gc.collect()
        time.sleep(1)
        shutil.rmtree(save_fpath, ignore_errors=True)

    # print("... Done change_dicom_to_nifti ...")
    print()
    print('- ', len(os.listdir(dicom_path)), dicom_path, os.listdir(dicom_path))
    print('- ', len(os.listdir(nifti_path)), nifti_path, os.listdir(nifti_path))
    print('- (Empty dicom)     ', len(empty_dcms), empty_dcms)
    print('- (Multiple nifties)', len(multiple_niftis), multiple_niftis)
    if multiple_niftis:
        assert len(set(os.listdir(dicom_path)) - set(empty_dcms)) + (len(multiple_niftis) - 1) == len(os.listdir(nifti_path)), f'dicom : {len(os.listdir(dicom_path))}, nifti : {len(os.listdir(nifti_path))}'
    else:
        assert len(set(os.listdir(dicom_path)) - set(empty_dcms)) == len(os.listdir(nifti_path)), f'dicom : {len(os.listdir(dicom_path))}, nifti : {len(os.listdir(nifti_path))}'
    print()

    return empty_dcms, multiple_niftis



def set_image_name_and_dir(input_path, save_path=None):
    
    for idx, f in enumerate(glob.glob(f"{input_path}/*", recursive=True)): # os.path.join(org_img_path, '**.nii.gz')
        if '_0000' not in f:
            nnunet_f = f.replace(".nii.gz", "_0000.nii.gz")
            print(f"[{idx+1}]  {f} -> \033[36m{nnunet_f}\033[0m")
            if save_path:
                shutil.copyfile(f, nnunet_f)
                # shutil.move(f, nnunet_f)
            else:
                os.rename(f, nnunet_f)
        else:
            print(f"[{idx+1}]  Already exists : \033[36m{f}\033[0m")
    
    # print("... Done set_image_name_and_dir ...")
        