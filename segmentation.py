import os
import glob
import shutil
import SimpleITK as sitk
# import nibabel as nib
import numpy as np
from pathlib import Path
import subprocess

# from make_nnunet_dataset import set_image_name_and_dir
# from dicom_to_nifti import change_dicom_to_nifti
# from dicom_to_png import change_dicom_to_png
# from merge_mask import make_dataset_mask

# from model import merge_mask, make_nnunet_dataset
from model.utils_nnunet import change_dicom_to_nifti, set_image_name_and_dir

from utils import check_files

def nnunet_inference(input_path, output_path, 
                     cuda_visible_device=None, 
                     root_path=None,
                     dataset_name='Dataset025_Abdominal_all_merged_zscore',
                     dataset_class='bile_duct/pancreatic_duct/pancreas/bile_in_pancreas',
                     configuration='3d_fullres', 
                     fold='0 1 2 3 4', 
                     checkpoint_pth='checkpoint_final.pth',
                     venv_path=None
                     ):
    
    '''
    # Structure
    model
    |ㅡ nnunet
    |ㅡ paths
    |ㅡ dataset
        |ㅡ inputImgs
        |   |ㅡ dicom
        |   |   |ㅡ 17286516
        |   |   |   |ㅡ 1728651600000000.dcm
        |   |   |   |ㅡ ...
        |   |   |ㅡ 17368021
        |   |   |ㅡ ...
        |   |ㅡ nifti
        |   |   |ㅡ 17286516_0000.nii.gz
        |   |   |ㅡ 17368021_0000.nii.gz
        |   |   |ㅡ ...
        |ㅡ outputImgs
            |ㅡ ...

    # Parameter
    checkpoint     : Name of the checkpoint you want to use             # checkpoint_final.pth, checkpoint_best.pth    
    configuration  : Name of configuration which should be inferenced  # 2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres (must check output trained folder)
    fold           : Fold of the 5-fold cross-validation                # 0, 1, 2, 3, 4 (Should be an int between 0 and 4), all
    '''

    input_path = Path(input_path)
    output_path = Path(output_path)

    if root_path is None:
        root_path = Path(__file__).resolve().parent / "model"
    else:
        root_path = Path(root_path)

    print("\n\033[1m  # change_dicom_to_nifti #  \033[0m")
    if input_path.name.lower() == "dicom":  # \model\dataset\inputImgs\dicom
        print(f"- nnunet input : {input_path}")
        print(f"- nnunet output : {output_path}")

        # input_nifti_path = input_path.replace('dicom', 'nifti')  # # \model\dataset\inputImgs\nifti
        # _empty, _multiple = change_dicom_to_nifti(input_path, input_nifti_path)
        input_nifti_path = input_path.parent / "nifti"
        _empty, _multiple = change_dicom_to_nifti(str(input_path), str(input_nifti_path))
        
        if _multiple:
            print(f"\033[1;31m>> IDs with multiple nifti files (ID counts == nifti counts) : {_multiple}\033[0m")
            print(f"\033[1;31m>> dcm folders : {len(set(os.listdir(input_path)) - set(_empty))} | nifti files : {len(os.listdir(input_nifti_path))}\033[0m")
            assert len(set(os.listdir(input_path)) - set(_empty)) + (len(_multiple) - 1) == len(os.listdir(input_nifti_path)), f'dicom : {len(os.listdir(input_path))}, nifti : {len(os.listdir(input_nifti_path))}'
        else:
            assert len(set(os.listdir(input_path)) - set(_empty)) == len(os.listdir(input_nifti_path)), f'dicom : {len(os.listdir(input_path))}, nifti : {len(os.listdir(input_nifti_path))}'
        
        input_path = input_nifti_path

    assert os.path.exists(input_path), print(f"! Input path does not exists ! ({input_path})")
    print(f"- nnunet input : {input_path}")
    print(f"- nnunet output : {output_path}")

    # check if all input files are in nifti format
    check_files(input_path)
    _nifties = [f for f in os.listdir(input_path) if not f.endswith('.nii.gz')]
    assert not _nifties, print(f"! There's non-nifti file or folder ! (non-nifti : {len(_nifties)} {_nifties})")


    print("\n\033[1m  # set_image_name_and_dir #  \033[0m")
    set_image_name_and_dir(input_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    assert os.path.isdir(input_path)
    assert os.path.isdir(output_path)

    
    print("\n\033[1m  # nnUnet Inference #  \033[0m")
    '''
    # set nnUNet_raw & nnUNet_preprocessed, just to remove warning signs
    nnUNet_raw is not defined and nnU-Net can only be used on data for which preprocessed files are already present on your system. nnU-Net cannot be used for experiment planning and preprocessing like this. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up properly.
    nnUNet_preprocessed is not defined and nnU-Net can not be used for preprocessing or training. If this is not intended, please read documentation/setting_up_paths.md for information on how to set this up.
    '''
    os.environ["nnUNet_raw"] = str(root_path / "nnUnet_raw")
    os.environ["nnUNet_preprocessed"] = str(root_path / "nnUnet_preprocessed")
    os.environ["nnUNet_results"] = str(root_path / "nnUnet_results")
    # export nnUNet_raw = os.path.join(root_path, 'nnUnet_raw')
    # export nnUNet_preprocessed = os.path.join(root_path, 'nnUnet_preprocessed')
    # export nnUNet_results = os.path.join(root_path, 'nnUnet_results')
    print(f" {os.environ['nnUNet_raw']}")
    print(f" {os.environ['nnUNet_preprocessed']}")
    print(f" {os.environ['nnUNet_results']}")
    
    check_files(root_path)
    '''
    (Inference with my models) https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md#run-inference
        nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities # also need to specfy fold num !
    
    (Inference with pretrained models) https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/run_inference_with_pretrained_models.md
        Not yet available for V2 :-(
    '''
    device = 'cuda' if cuda_visible_device else 'cpu'
    cuda_cmd = f"CUDA_VISIBLE_DEVICES={cuda_visible_device} " if cuda_visible_device else ''
    fold_cmd = f'-f {fold} ' if fold != '0 1 2 3 4' else ''
    pth_cmd = f'-chk {checkpoint_pth} ' if checkpoint_pth != 'checkpoint_final.pth' else ''

    if venv_path:
        if os.name == "nt":  # Windows
            nnunet_cmd = os.path.join(venv_path, "Scripts", "nnUNetv2_predict.exe")
        else:  # Linux/macOS
            nnunet_cmd = os.path.join(venv_path, "bin", "nnUNetv2_predict")
        
        if not os.path.exists(nnunet_cmd):  # Linux
            # nnunet_cmd = "/opt/conda/bin/nnUNetv2_predict"
            # print(f"- (! nnUNetv2_predict not found in venv ({nnunet_cmd}) >> trying Conda path...)")
            raise FileNotFoundError(f"nnUNetv2_predict not found in venv: {nnunet_cmd}")
    else:
        nnunet_cmd = "nnUNetv2_predict"  # Use system path

    # print(f"{cuda_cmd}nnUNetv2_predict -i {os.path.join(input_path)} -o {output_path} -d {dataset_name.split('_')[0][-3:]} -c {configuration} -device {device} {fold_cmd}{pth_cmd}--save_probabilities")
    # os.system(f"{cuda_cmd}nnUNetv2_predict -i {os.path.join(input_path)} -o {output_path} -d {dataset_name.split('_')[0][-3:]} -c {configuration} -device {device} {fold_cmd}{pth_cmd}--save_probabilities")
    print(f"{cuda_cmd}{nnunet_cmd} -i {os.path.join(input_path)} -o {output_path} -d {dataset_name.split('_')[0][-3:]} -c {configuration} -device {device} {fold_cmd}{pth_cmd}--save_probabilities")
    os.system(f"{cuda_cmd}{nnunet_cmd} -i {os.path.join(input_path)} -o {output_path} -d {dataset_name.split('_')[0][-3:]} -c {configuration} -device {device} {fold_cmd}{pth_cmd}--save_probabilities")

    # /inference/predict_from_raw_data.py
    # /inference/data_iterators.py
    assert len(os.listdir(output_path)) > 0, "nnUNet inference failed."
    print("... Done nnUnet Inference ...")

    return input_path, output_path
