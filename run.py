import os
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/opt/conda/lib'
print(f"LD_LIBRARY_PATH is set to: {os.environ['LD_LIBRARY_PATH']}")
print()

import glob
import shutil
import SimpleITK as sitk  # (pyinstaller) Auto-detected
# import nibabel as nib
import numpy as np        # (pyinstaller) Auto-detected
import torch              # (pyinstaller) Auto-detected

import argparse
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import time
from statistics import mean

from segmentation import nnunet_inference
from utils import *
from cyst_size_measure import *
from bias_correction import BiasCorrection

from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATASET_DIR = MODEL_DIR / "dataset"


def format_time(seconds):
    minutes, secs = divmod(seconds, 60)
    return f"{int(minutes)}m {secs:.2f}s"

def get_cuda_device(cuda_visible_device):

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"\033[1m** GPU Available: {num_gpus} device(s) **\033[0m")
        if cuda_visible_device is not None:
            if int(cuda_visible_device) >= num_gpus:
                print(f"\033[1m** (selected GPU {cuda_visible_device} is not available!) **\033[0m")
                exit(1)
            else:
                print(f"** GPU Used     : {torch.cuda.get_device_name(int(cuda_visible_device))} **")
        else:
            # gpu_id = str(torch.cuda.current_device())  # Auto-detect available GPU
            # cuda_visible_device = gpu_id
            cuda_visible_device = "0"
            print(f"** Auto-selected GPU : {cuda_visible_device} ({torch.cuda.get_device_name(0)}) **")
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_device
    else:
        print("\033[1m** Running on CPU (NO GPU Available!) **\033[0m")
        cuda_visible_device = None    #  CPU

    return cuda_visible_device


def run_external_script(root_path, fid, idx, device):
    """
    - Windows: `CystAnalysis_2503_no_fit.bat`
    - Linux  : `CystAnalysis_2503_no_fit.sh`

    os.name 
        "posix" (Linux/macOS)
        "nt" (Windows)
    platform.system()
        "Windows"
        "Linux"
        "Darwin" (macOS)
    """
    print(f"\033[1m** OS type : {platform.system()} **\033[0m")
    if platform.system() == "Windows":
        script_name = "CystAnalysis_2503_no_fit.bat"
        if not os.path.exists(script_name):
            raise FileNotFoundError(f"{script_name} not found!")
        
    else:  # Linux/macOS
        script_name = "./CystAnalysis_2503_no_fit.sh"
        if not os.path.exists(script_name):
            raise FileNotFoundError(f"{script_name} not found!")
        
        # remove Windows line endings
        # - Windows line endings (\r\n) cause errors on Linux (\n expected)
        # - Check if ^M exists using: cat -A CystAnalysis_2503_no_fit.sh | head -n 5
        # subprocess.run(["sed", "-i", "s/\r$//", script_name], check=True)
        # Convert Windows CRLF to Unix LF (OS-independent)
        with open(script_name, "rb") as f:
            content = f.read().replace(b"\r\n", b"\n")
        with open(script_name, "wb") as f:
            f.write(content)

        # give execute permission before running
        subprocess.run(["chmod", "+x", script_name], check=True)
        if not os.access(script_name, os.X_OK):
            os.chmod(script_name, 0o755)

    # cmd = f"{script_name} {root_path} {fid} {device}"
    device_arg = "" if device is None else str(device)
    cmd = f"{script_name} {root_path} {fid} {device_arg}"

    print(f"* cmd : {cmd} *")
    # process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    # stdout, stderr = process.communicate()
    # process.wait()  # Ensure process completes properly  
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    stdout, stderr = process.stdout, process.stderr

    exit_code = process.returncode  # Get exit code

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"log_{idx+1}_{fid}.txt") 

    with open(log_path, "w", encoding="utf-8") as log_file:
        if stdout:
            log_file.write(stdout)
            print(stdout, end="")  # Print to console
        if stderr:
            log_file.write(stderr)
            print(stderr, end="")  # Print to console

    if exit_code != 0:
        print(f"\033[1;31m❌ Error: The process failed with exit code {exit_code}\033[0m")
        if stderr:
            print(f"\033[1;31m❌ [ERROR OUTPUT]:\n{stderr}\033[0m")
        sys.exit(1)


def bias_worker(fpath):
    fid = os.path.basename(fpath)
    org_img_fpath = f"{fpath}/{fid}_0000.nii.gz"
    corr_img_fpath = f"{fpath}/{fid}_t2_corr.nii.gz"
    # BiasCorrection(org_img_fpath, corr_img_fpath)
    return BiasCorrection(fid, org_img_fpath, corr_img_fpath)


def run_process(nnunet_input_path, cuda_visible_device=None):
    start_time = time.time()

    device = get_cuda_device(cuda_visible_device)
    t1 = time.time()
    spent_device = t1 - start_time
    print(f"\n\033[32m++++++ Time for setting device : {format_time(spent_device)} ++++++\033[0m\n")


    """1. nnUnet inference"""
    # ## Linux
    # # dataset = nnunet_input_path.split('/')[-2].replace("inputImgs_", "")
    # ## Windows
    # dataset = nnunet_input_path.split('\\')[-2].replace("inputImgs_", "")
    # print(f"\033[1;33m- dataset    : {dataset}\033[0m")
    
    ## Linux
    # # nnunet_output_path = f"model/dataset/outputImgs_{dataset}"  # r"model\dataset\outputImgs"
    # # image_path, mask_path = nnunet_inference(nnunet_input_path, nnunet_output_path, device)                             
    # ## Windows
    # # nnunet_output_path = r"model\dataset\outputImgs"
    # nnunet_output_path = fr"model\dataset\outputImgs_{dataset}"
    # image_path, mask_path = nnunet_inference(nnunet_input_path, 
    #                                          nnunet_output_path, 
    #                                          device, 
    #                                          root_path = r'E:\MI2RL\01_duct_seg_nnunet\codes\ALL_AT_ONCE\model',
    #                                          venv_path = r'E:\MI2RL\01_duct_seg_nnunet\codes\ALL_AT_ONCE\venv_01_duct'
    #                                          )
    
    nnunet_input_path = Path(nnunet_input_path)

    # e.g. model/dataset/inputImgs/dicom -> dataset = "inputImgs"
    # e.g. model/dataset/inputImgs_260108/dicom -> dataset = "260108"
    dataset_dir = nnunet_input_path.parent.name
    if dataset_dir.startswith("inputImgs_"):
        dataset = dataset_dir.replace("inputImgs_", "")
    else:
        dataset = ""
    print(f"\033[1;33m- dataset    : {dataset}\033[0m")

    if dataset:
        nnunet_output_path = DATASET_DIR / f"outputImgs_{dataset}"
        root_path = Path(f"output_{dataset}")
    else:
        nnunet_output_path = DATASET_DIR / "outputImgs"
        root_path = Path("output")

    image_path, mask_path = nnunet_inference(str(nnunet_input_path),
                                             str(nnunet_output_path),
                                             device,
                                             root_path=str(MODEL_DIR),
                                             venv_path=None,
                                             )
    print(f"\033[1;33m- image_path : {image_path}\033[0m")
    print(f"\033[1;33m- mask_path  : {mask_path}\033[0m")

    # image_path = "model/dataset/inputImgs/nifti"
    # mask_path = "model/dataset/outputImgs"
    # print(f"\033[1;33m- image_path : {image_path}\033[0m")
    # print(f"\033[1;33m- mask_path : {mask_path}\033[0m")

    print("\n\n\033[1m ## Move nnUnet output files ##\033[0m")
    # root_path = fr"output_{dataset}"
    # root_path = Path(f"output_{dataset}")
    print(f"(Copy image files)")
    # for idx, f in enumerate(glob.glob(f"{image_path}/*", recursive=True)): # os.path.join(org_img_path, '**.nii.gz')
    for idx, f in enumerate(glob.glob(str(Path(image_path) / "*"), recursive=True)):
        fname = os.path.basename(f)
        fid = fname.replace('_0000.nii.gz', '')
        # fpath = os.path.join(root_path, fid)
        fpath = root_path / fid
        # if not os.path.exists(fpath):
        #     os.makedirs(fpath)
        fpath.mkdir(parents=True, exist_ok=True)
        img_f = f"{fpath}/{fname}"
        print(f"[{idx+1}]  {f} -> \033[36m{img_f}\033[0m")
        shutil.copyfile(f, img_f)
    print(f"(Copy mask files)")
    # for idx, f in enumerate(glob.glob(f"{mask_path}/*.nii.gz", recursive=True)): # os.path.join(org_img_path, '**.nii.gz')
    for idx, f in enumerate(glob.glob(str(Path(mask_path) / "*.nii.gz"), recursive=True)):
        fname = os.path.basename(f)
        fid = fname.replace('.nii.gz', '')
        # fpath = os.path.join(root_path, fid)
        fpath = root_path / fid
        assert os.path.exists(fpath), print(f"! Check {fid} folder exists in {root_path} !")
        mask_f = f"{fpath}/{fname.replace('.nii.gz', '_pred.nii.gz')}"
        print(f"[{idx+1}]  {f} -> \033[36m{mask_f}\033[0m")
        shutil.copyfile(f, mask_f)

    print(f"\033[1;33m- root_path : {root_path}\033[0m")
    print(f"- Total {len(list(root_path.iterdir()))} cases : {os.listdir(root_path)}")


    t2 = time.time()
    spent_seg = t2 - t1
    print(f"\n\033[32m++++++ Time for nnUnet segmentation : {format_time(spent_seg)} ++++++\033[0m\n")

    
    """3. image bias_correction"""
    print("\n\n\033[1m ## Bias Field Correction ##\033[0m")
    # _fpaths = glob.glob(f"{root_path}/*", recursive=True)
    _fpaths = glob.glob(str(root_path / "*"), recursive=True)
    # with tqdm(total=len(_fpaths), desc="Bias Correction", unit="case") as pbar:
    #     for idx, fpath in enumerate(_fpaths, start=1): # os.path.join(org_img_path, '**.nii.gz')
    #         fid = os.path.basename(fpath)
    #         org_img_fpath = f"{fpath}/{fid}_0000.nii.gz"
    #         corr_img_fpath = f"{fpath}/{fid}_t2_corr.nii.gz"
    #         tqdm.write(f"[{idx}] correcting {fid} (Takes long time!) ...")
    #         BiasCorrection(org_img_fpath, corr_img_fpath)
    #         pbar.update(1)

    try:
        # multi-processing
        tqdm.write(">>✅ Using Multiprocessing...")
        print(f">>** CPU cores available: {multiprocessing.cpu_count()}")
        # total_start = time.time()
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(bias_worker, fpath): fpath for fpath in _fpaths}
            with tqdm(total=len(futures), desc="N4 Bias Correcting", unit="case") as pbar:
                for idx, future in enumerate(as_completed(futures), start=1):
                    try:
                        # future.result()
                        logs = future.result()
                        pbar.update(1)
                        # for line in logs:
                        #     tqdm.write(f"[{idx}] {line}")
                        tqdm.write(f"[{idx}] {logs}")
                    except Exception as e:
                        tqdm.write(f"❌ Error in one worker: {e}")
    except Exception as e:
        # single-processing
        tqdm.write(f"\n>>⚠️ Multiprocessing failed: {e}")
        tqdm.write(">> Switching to single-threaded fallback...\n")
        # total_start = time.time()
        with tqdm(total=len(_fpaths), desc="N4 Bias Correcting", unit="case") as pbar:
            for idx, fpath in enumerate(_fpaths, start=1):
                fid = os.path.basename(fpath)
                org_img_fpath = f"{fpath}/{fid}_0000.nii.gz"
                corr_img_fpath = f"{fpath}/{fid}_t2_corr.nii.gz"
                logs = BiasCorrection(fid, org_img_fpath, corr_img_fpath)
                pbar.update(1)
                # for line in logs:
                #     tqdm.write(f"[{idx}] {line}")    
                tqdm.write(f"[{idx}] {logs}")

    # tqdm.write(">> Stick to Single-processing...")
    # total_start = time.time()
    # with tqdm(total=len(_fpaths), desc="N4 Bias Correcting", unit="case") as pbar:
    #     for idx, fpath in enumerate(_fpaths, start=1):
    #         fid = os.path.basename(fpath)
    #         org_img_fpath = f"{fpath}/{fid}_0000.nii.gz"
    #         corr_img_fpath = f"{fpath}/{fid}_t2_corr.nii.gz"
    #         logs = BiasCorrection(fid, org_img_fpath, corr_img_fpath)
    #         pbar.update(1)
    #         # for line in logs:
    #         #     tqdm.write(f"[{idx}] {line}")    
    #         tqdm.write(f"[{idx}] {logs}")
    # total_elapsed = time.time() - total_start
    # print(f"\n⏱️ \033[1mTotal Bias Correction Time: {format_time(total_elapsed)}\033[0m")

    # total_elapsed = time.time() - total_start
    t3 = time.time()
    spent_bias_corr = t3 - t2
    print(f"\n\033[32m++++++ Time for image bias correction : {format_time(spent_bias_corr)} ++++++\033[0m\n")


    # root_path = fr"output_{dataset}"
    # root_path = Path(f"output_{dataset}")
    check_files(root_path)
    # return

    _start = t3
    _spent_cyst_all = []
    # for idx, fpath in enumerate(sorted(glob.glob(f"{root_path}/*", recursive=True)), start=1):
    for idx, fpath in enumerate(sorted(glob.glob(str(root_path / "*"), recursive=True)), start=1):
        print(f"\n\n\033[1;32m=== [[{idx}]]  Processing : {fpath} ======================================================================================\033[0m")
        if not os.path.isdir(fpath):
            print(f"\n\nNot a directory : {fpath} >> pass")
            continue
        # check_files(fpath)
        # assert len(os.listdir(fpath)) == 2  # ['case001_0000.nii.gz', 'case001_pred.nii.gz']

        print("\n\n\033[1m ## Check input files ##\033[0m")
        fid = os.path.basename(fpath)
        print(f"\033[1;33m- ID         : {fid}\033[0m")
        org_img_fpath = f"{fpath}/{fid}_0000.nii.gz"
        pred_fpath = f"{fpath}/{fid}_pred.nii.gz"  # {fid}_merged.nii.gz"
        assert os.path.exists(org_img_fpath), f"there's no input image : {org_img_fpath}"
        assert os.path.exists(pred_fpath), f"there's no prediction : {pred_fpath}"
        print(f"\033[1;33m- image      : {org_img_fpath}\033[0m")
        print(f"\033[1;33m- pred_mask  : {pred_fpath}\033[0m")
        
        # check segmentation gone successful
        _, pred_arr = read_sitk(pred_fpath)
        print(f'\033[0;33m- pred_arr  : {pred_arr.shape} {np.unique(pred_arr, return_counts=True)}\033[0m')
        if len(np.unique(pred_arr)) != 5:
            print("⚠️ Segmentation is not ideal ")
        if len(np.unique(pred_arr)) < 3:
            print("❌ There should be at least 2 mask detected from segmentation (Skip cyst analysis)")
            continue
        if 3 not in np.unique(pred_arr):
            print("❌ There should be pancreas mask from segmentation (Skip cyst analysis)")
            continue


        """2. split mask"""
        print("\n\n\033[1m ## split_mask ##\033[0m")
        bduct_fpath, pduct_fpath, pancreas_fpath, bile_in_pancreas_fpath = split_mask(fid, fpath, pred_fpath)
        assert os.path.exists(bduct_fpath), print(f"there's no bduct mask : {bduct_fpath}")
        assert os.path.exists(pduct_fpath), print(f"there's no pduct mask : {pduct_fpath}")
        assert os.path.exists(pancreas_fpath), print(f"there's no pancreas mask : {pancreas_fpath}")
        assert os.path.exists(bile_in_pancreas_fpath), print(f"there's no bile_in_pancreas mask : {bile_in_pancreas_fpath}")
        print(f"\033[1;33m- bduct_mask            : {bduct_fpath}\033[0m")
        print(f"\033[1;33m- pduct_mask            : {pduct_fpath}\033[0m")
        print(f"\033[1;33m- pancreas_mask         : {pancreas_fpath}\033[0m")
        print(f"\033[1;33m- bile_in_pancreas_mask : {bile_in_pancreas_fpath}\033[0m")

        # """3. image bias_correction"""
        # print("\n\n\033[1m ## BiasCorrection ##\033[0m")
        # img_fpath = f"{fpath}/{fid}_t2_corr.nii.gz"
        # BiasCorrection(org_img_fpath, img_fpath)
        # assert os.path.exists(img_fpath), print(f"there's no {img_fpath}")
        # print(f"\033[1;33m- bias_corrected_image : {img_fpath}\033[0m")
        img_fpath = f"{fpath}/{fid}_t2_corr.nii.gz"
        assert os.path.exists(img_fpath), print(f"there's no bias corrected image {img_fpath}")
        # print(f"\033[1;33m- bias_corrected_image : {corr_img_fpath}\033[0m")
        
        """4. pancreas mask thresholding"""
        print("\n\n\033[1m ## thresholding ##\033[0m")
        run_external_script(str(root_path), fid, idx, device)

        _mask_statistics = f"{fpath}/mask_statistics/{fid}_t2_all_mask_histogram_statistics.csv"
        _mask_histogram = f"{fpath}/mask_statistics/{fid}_t2_all_mask_histogram.png"
        pancreas_S_fpath = f"{fpath}/{fid}_t2_pancreas_scaled.nii.gz"
        pancreas_T_fpath = f"{fpath}/{fid}_t2_pancreas_thresholded.nii.gz"
        assert os.path.exists(_mask_statistics), print(f"there's no mask statistics : {_mask_statistics}")
        assert os.path.exists(_mask_histogram), print(f"there's no mask histogram : {_mask_histogram}")
        assert os.path.exists(pancreas_S_fpath), print(f"there's no scaled-pancreas mask : {pancreas_S_fpath}")
        assert os.path.exists(pancreas_T_fpath), print(f"there's no thresholded-pancreas mask : {pancreas_T_fpath}")
        # print(f"\033[1;33m- _mask_statistics          : {_mask_statistics}\033[0m")
        # print(f"\033[1;33m- _mask_histogram           : {_mask_histogram}\033[0m")
        print(f"\033[1;33m- pancreas_mask_scaled      : {pancreas_S_fpath}\033[0m")
        print(f"\033[1;33m- pancreas_mask_thresholded : {pancreas_T_fpath}\033[0m")

        """5. duct mask dilation"""
        print("\n\n\033[1m ## duct mask dilation ##\033[0m")
        bduct_dilated_fpath = dilate_mask(fid, bduct_fpath)
        assert os.path.exists(bduct_dilated_fpath), print(f"there's no dilated bduct mask : {bduct_dilated_fpath}")
        print(f"\033[1;33m- bduct_dilated_mask : {bduct_dilated_fpath}\033[0m")
        pduct_dilated_fpath = dilate_mask(fid, pduct_fpath)
        assert os.path.exists(pduct_dilated_fpath), print(f"there's no dilated pduct mask : {pduct_dilated_fpath}")
        print(f"\033[1;33m- pduct_dilated_mask : {pduct_dilated_fpath}\033[0m")
        bile_in_pancreas_dilated_fpath = dilate_mask(fid, bile_in_pancreas_fpath)
        assert os.path.exists(bile_in_pancreas_dilated_fpath), print(f"there's no dilated pduct mask : {bile_in_pancreas_dilated_fpath}")
        print(f"\033[1;33m- pduct_dilated_mask : {bile_in_pancreas_dilated_fpath}\033[0m")

        """6. remove duct mask & extract cyst mask"""
        print("\n\n\033[1m ## remove duct mask & extract cyst mask ##\033[0m")
        cyst_path, _, _ = remove_mask(fid, fpath, bduct_fpath, pduct_fpath, bile_in_pancreas_fpath, pancreas_fpath, pancreas_T_fpath)
        # assert os.path.exists(cyst_path), print(f"there's no cyst mask : {cyst_path}")
        print(f"\033[1;33m- cyst_path : {cyst_path}\033[0m" if os.path.exists(cyst_path) else " ! No cyst found !")
        _, cyst_path_rm_duct_dilated, panc_T_path_rm_duct_dilated = remove_mask(fid, fpath, bduct_dilated_fpath, pduct_dilated_fpath, bile_in_pancreas_dilated_fpath, pancreas_fpath, pancreas_T_fpath)
        # assert os.path.exists(cyst_path_rm_duct_dilated), print(f"there's no duct removed cyst mask : {cyst_path_rm_duct_dilated}")
        # assert os.path.exists(panc_T_path_rm_duct_dilated), print(f"there's no dilated-duct removed cyst mask : {panc_T_path_rm_duct_dilated}")
        print(f"\033[1;33m- cyst_path_rm_duct_dilated   : {cyst_path_rm_duct_dilated}\033[0m" if os.path.exists(cyst_path_rm_duct_dilated) else " ! No cyst_rm_duct found !")
        print(f"\033[1;33m- panc_T_path_rm_duct_dilated : {panc_T_path_rm_duct_dilated}\033[0m" if os.path.exists(panc_T_path_rm_duct_dilated) else " ! No cyst found !")


        """7. cyst size measurement"""
        print("\n\n\033[1m ## process cyst size measurement ##\033[0m")
        if os.path.exists(cyst_path_rm_duct_dilated):

            all_area_ratios_values = []
            all_area_ratios_by_id = {}
            all_area_ratios_by_id_values = {}
            all_area_ratios_by_id_min = {}

            save_path = f"{fpath}/cyst_measurements"
            # save_path = f"{fpath}/cyst_measurements_axis"
            # save_path = f"{fpath}/cyst_measurements_area"

            cyst_img, cyst_arr = read_sitk(cyst_path_rm_duct_dilated)
            print(f'-- cyst_arr  : {cyst_arr.shape} {np.unique(cyst_arr, return_counts=True)}')
            
            # Save convex_hull results to mask
            # cyst_info, cyst_removed, area_ratios = process_cyst_analysis(fid, img_fpath, cyst_img, cyst_path_rm_duct_dilated, save_path)
            label_save_path, output_csv_path, output_json_path, area_ratios = process_cyst_analysis(fid, cyst_img, cyst_path_rm_duct_dilated, img_fpath, save_path)
            if os.path.exists(label_save_path):
                print(f"\033[1;33m- label_save_path : {label_save_path}\033[0m")
            if os.path.exists(output_csv_path):
                print(f"\033[1;33m- output_csv_path : {output_csv_path}\033[0m")
            if os.path.exists(output_json_path):
                print(f"\033[1;33m- output_json_path : {output_json_path}\033[0m")

            all_area_ratios_by_id[fid] = sorted(area_ratios, key=lambda x: x[1])
            all_area_ratios_by_id_values[fid] = sorted(set(x[1] for x in area_ratios))
            all_area_ratios_by_id_min[fid] = min(area_ratios, key=lambda x: x[1])  # min(x[1] for x in area_ratios)
            all_area_ratios_values.extend(x[1] for x in area_ratios)

        else:
            print(" ! Skip cyst measurement !")
        
        _end = time.time()
        _spent_cyst_each = _end - _start
        print(f"\n\033[32m   +++ (cyst measurement of {fid} : {format_time(_spent_cyst_each)}) +++   \033[0m\n")
        _start = _end
        _spent_cyst_all.append(_spent_cyst_each)


    t4 = time.time()
    spent_cyst = t4 - t3
    print(f"\n\033[32m++++++ Time for cyst measurement : {format_time(spent_cyst)} ++++++\033[0m\n")


    print(f"\n\033[1;32m###### Total time                               || {format_time(t4 - start_time)} ######\033[0m")
    print(f"\033[32m++++++ Time for device setting                  || {format_time(spent_device)} ++++++\033[0m")
    print(f"\033[32m++++++ Time for nnUnet segmentation             || {format_time(spent_seg)} ++++++\033[0m")
    print(f"\033[32m++++++ Time for image bias correction           || {format_time(spent_bias_corr)} ++++++\033[0m")
    print(f"\033[32m++++++ Time for cyst measurement ({len(os.listdir(root_path)):<3} cases)    || {format_time(spent_cyst)} ++++++\033[0m")
    print(f"\033[32m++++++ (Individual Time for cyst measurement)   || {format_time(mean(_spent_cyst_all))} (min: {format_time(min(_spent_cyst_all))} / max: {format_time(max(_spent_cyst_all))})++++++\033[0m")
    print()
    # # For 1 case
    # ###### Total time                               || 5m 0.96s ######
    # ++++++ Time for device setting                  || 0m 0.06s ++++++
    # ++++++ Time for nnUnet segmentation             || 0m 44.67s ++++++
    # ++++++ Time for image bias correction           || 3m 54.01s ++++++
    # ++++++ Time for cyst measurement (1   cases)    || 0m 22.22s ++++++
    # ++++++ (Individual Time for cyst measurement)   || 0m 22.22s (min: 0m 22.22s / max: 0m 22.22s)++++++

    print("\n\nDone processing all cases...!")
    print("+ Run ./cyst_analysis_post.ipynb to get cyst_count_summary & cyst_largest_summary CSV files...!")


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Segmenatation - Organize files - Split pred mask - Image bias correction - Pncareas mask Thresholding - Duct mask dilation - Rmove Duct & Extract Cyst mask - Cyst size measurement')
    parser.add_argument("-d", "--cuda_visible_device", type=str, default=None, help='GPU device number')  # default='0', 
    # parser.add_argument("-i", "--nnunet_input_path", type=str, default="model/dataset/inputImgs/nifti", help="Root directory for nnUnet segmentation Inference")
    # parser.add_argument("-i", "--nnunet_input_path", type=str, default=r"model\dataset\inputImgs_260108\dicom", help="Root directory for nnUnet segmentation Inference")
    parser.add_argument("-i", "--nnunet_input_path", type=str, default=str(DATASET_DIR / "inputImgs" / "dicom"), help="Root directory for nnU-Net segmentation inference")

    args = parser.parse_args()

    # os.environ['MKL_THREADING_LAYER']='GNU'
    # os.environ['MKL_SERVICE_FORCE_INTEL']='1'
    # os.environ['TORCHDYNAMO_DISABLE']="1"
    # os.environ['OMP_NUM_THREADS']="1"

    run_process(args.nnunet_input_path, cuda_visible_device=args.cuda_visible_device)

    