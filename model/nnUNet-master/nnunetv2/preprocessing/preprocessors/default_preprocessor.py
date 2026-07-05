#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import multiprocessing
import shutil
from time import sleep
from typing import Tuple, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from tqdm import tqdm

import nnunetv2
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


class DefaultPreprocessor(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called
        """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager.transpose_forward]])
        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        # if we have a segmentation, sample foreground locations for oversampling and add those to properties
        if has_seg:
            # reinstantiating LabelManager for each case is not ideal. We could replace the dataset_json argument
            # with a LabelManager Instance in this function because that's all its used for. Dunno what's better.
            # LabelManager is pretty light computation-wise.
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            # when using the ignore label we want to sample only from annotated regions. Therefore we also need to
            # collect samples uniformly from all classes (incl background)
            if label_manager.has_ignore_label:
                collect_for_this.append(label_manager.all_labels)

            # no need to filter background in regions because it is already filtered in handle_labels
            # print(all_labels, regions)
            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
        if np.max(seg) > 127:
            seg = seg.astype(np.int16)
        else:
            seg = seg.astype(np.int8)
        return data, seg

    def run_case(self, image_files: List[str], seg_file: Union[str, None], plans_manager: PlansManager,
                 configuration_manager: ConfigurationManager,
                 dataset_json: Union[dict, str]):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        if isinstance(dataset_json, str):
            dataset_json = load_json(dataset_json)

        rw = plans_manager.image_reader_writer_class()

        # load image(s)
        data, data_properties = rw.read_images(image_files)

        #### added by hyenagatha02 ####
        # Apply additional preprocessing only during inference
        if seg_file is None:  # Inference step
            print("\n++++++ Inference >> Additional preprocessing ++++++")

            # Extract training intensity range from configuration
            intensity_properties = plans_manager.plans['foreground_intensity_properties_per_channel']['0']
            train_min = intensity_properties['min']
            train_max = intensity_properties['max']
            train_percentile_00_5 = intensity_properties['percentile_00_5']
            train_percentile_99_5 = intensity_properties['percentile_99_5']
            print(f"- Train intensity range (min ~ max)             : {train_min:.2f} - {train_max:.2f}")
            print(f"- Train intensity range (percentile 0.5 ~ 99.5) : {train_percentile_00_5:.2f} - {train_percentile_99_5:.2f}")

            # Calculate test data intensity range
            data_min = np.min(data)
            data_max = np.max(data)
            data_percentile_00_5 = np.percentile(data, 0.5)
            data_percentile_99_5 = np.percentile(data, 99.5)
            print(f"- Test intensity range (min ~ max)              : {data_min:.2f} - {data_max:.2f}")
            print(f"- Test intensity range (percentile 0.5 ~ 99.5)  : {data_percentile_00_5:.2f} - {data_percentile_99_5:.2f}")

            ## 1. Min-Max scaling using percentiles
            # Apply scaling if test data range differs
            print(f"\033[36m>> Applying linear scaling (Min-Max) using percentiles (dtype : {data.dtype})\033[0m")  # float32
            if data_percentile_00_5 < train_percentile_00_5 or data_percentile_99_5 > train_percentile_99_5:
                data = data.astype(np.float32, copy=False)
                data = np.clip(data, data_percentile_00_5, data_percentile_99_5)
                # Scale to 0-1 based on the test data's actual range  (Min-Max Normalization)
                data = (data - data_percentile_00_5) / (data_percentile_99_5 - data_percentile_00_5)  # ((x - min)/(max - min))
                # Scale back to the training data's intensity range   (Rescaling)
                data = data * (train_percentile_99_5 - train_percentile_00_5) + train_percentile_00_5  # x_norm ⋅ (B − A) + A
                print(f"\033[1;36m-- New Test intensity range (Min-Max scaling)        : {np.min(data):.2f} - {np.max(data):.2f}\033[0m")
            else:
                print(f"\033[1;36m-- Keep original Test intensity range                : {np.min(data):.2f} - {np.max(data):.2f}\033[0m")


            ## 2. Apply fixed Z-score normalization from training
            # nnU-Net's default Z-score normalization compute mean and std per image (or per channel).
            # This means every test image gets normalized differently.
            # Good for robustness, but not consistent with training data(foreground) distribution if test data deviates heavily.
            # Using train_mean, train_std keeps the input distribution aligned with what the model was trained on.
            print(f"\033[35m>> Applying fixed Z-score normalization for test dataset (dtype : {data.dtype})\033[0m")  # float32
            train_mean = intensity_properties['mean']
            train_std = intensity_properties['std']
            print(f"\033[1;35m-- Train mean / std                             : {train_mean:.2f} / {train_std:.2f}\033[0m")
            # Convert and clip test data
            data = data.astype(np.float32, copy=False)
            # data = np.clip(data, train_percentile_00_5, train_percentile_99_5)  # Optional (necessary if min max scaling is not applied)
            # Apply fixed Z-score normalization
            data = (data - train_mean) / max(train_std, 1e-8)  # (Z = (X - mean) / std)
            print(f"\033[1;35m-- New Test intensity range (Fixed Z-score normalization) : {np.min(data):.2f} - {np.max(data):.2f}\033[0m")
            

            ## 3. Disable nnU-Net's default Z-score step only during inference
            print(f"\033[32m>> Use Fixed Zscore normalization (Disable nnU-Net's default normalization)\033[0m")  # float32
            print(f"\033[1;32m-- Normalization class (default)  : {configuration_manager.normalization_schemes}\033[0m")
            print(f"\033[1;32m-- Normalization class (default)  : {[type(n).__name__ for n in configuration_manager.normalization_schemes]}\033[0m")

            # Override plans to use NoNormalization
            plans_manager.plans["normalization_schemes"] = ["NoNormalization"]
            # Manually override normalization class
            from nnunetv2.preprocessing.normalization.default_normalization_schemes import NoNormalization
            intensity_properties = plans_manager.plans["foreground_intensity_properties_per_channel"]
            configuration_manager._normalization_schemes = [
                NoNormalization(intensityproperties=intensity_properties[str(i)], use_mask_for_norm=False)
                for i in range(len(configuration_manager.normalization_schemes))
            ]
            # print(f"*** Normalization class (used)   : {configuration_manager.normalization_schemes}")  # just copied from plans
            # print(f"*** Normalization class (used)   : {[type(n).__name__ for n in configuration_manager.normalization_schemes]}")
            # print(f"\033[1;32m*** Normalization class (used)  : {configuration_manager._normalization_schemes}\033[0m")  # actually used
            # print(f"\033[1;32m*** Normalization class (used)  : {[type(n).__name__ for n in configuration_manager._normalization_schemes]}\033[0m")
        ####

        # if possible, load seg
        if seg_file is not None:
            seg, _ = rw.read_seg(seg_file)
        else:
            seg = None

        if hasattr(configuration_manager, "_normalization_schemes"):
            print(f"\033[1;32m✔️ (Using manually set Normalization schemes)\033[0m")
        data, seg = self.run_case_npy(data, seg, data_properties, plans_manager, configuration_manager,
                                      dataset_json)
        
        #### added by hyenagatha02 ####
        if isinstance(data, np.ndarray):
            print(f"\033[1;33m>> Final processed data range: {np.min(data):.2f} - {np.max(data):.2f}\033[0m")
        # elif isinstance(data, torch.Tensor):
        #     print(f"Final processed data range: {data.min().item():.2f} - {data.max().item():.2f}")
        else:
            print(f"! Unexpected data type: {type(data)}")
        ####

        return data, seg, data_properties

    def run_case_save(self, output_filename_truncated: str, image_files: List[str], seg_file: str,
                      plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                      dataset_json: Union[dict, str]):
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        # print('dtypes', data.dtype, seg.dtype)
        np.savez_compressed(output_filename_truncated + '.npz', data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + '.pkl')

    @staticmethod
    def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected
            if verbose:
                print(c, target_num_samples)
        return class_locs

    def _normalize(self, data: np.ndarray, seg: np.ndarray, configuration_manager: ConfigurationManager,
                   foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        #### added by hyenagatha02 ####
        # Use override if _normalization_schemes is explicitly provided
        if hasattr(configuration_manager, '_normalization_schemes'):
            print(f"\033[1;32m-- Normalization class (used)  : {configuration_manager._normalization_schemes}\033[0m")  # actually used
            print(f"\033[1;32m-- Normalization class (used)  : {[type(n).__name__ for n in configuration_manager._normalization_schemes]}\033[0m")
            for c in range(data.shape[0]):
                data[c] = configuration_manager._normalization_schemes[c].run(data[c], seg[0])
        #####
        else:
            for c in range(data.shape[0]):
                scheme = configuration_manager.normalization_schemes[c]
                normalizer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                                                            scheme,
                                                            'nnunetv2.preprocessing.normalization')
                if normalizer_class is None:
                    raise RuntimeError(f'Unable to locate class \'{scheme}\' for normalization')
                normalizer = normalizer_class(use_mask_for_norm=configuration_manager.use_mask_for_norm[c],
                                            intensityproperties=foreground_intensity_properties_per_channel[str(c)])
                data[c] = normalizer.run(data[c], seg[0])
        return data

    def run(self, dataset_name_or_id: Union[int, str], configuration_name: str, plans_identifier: str,
            num_processes: int):
        """
        data identifier = configuration name in plans. EZ.
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)

        assert isdir(join(nnUNet_raw, dataset_name)), "The requested dataset could not be found in nnUNet_raw"

        plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + '.json')
        assert isfile(plans_file), "Expected plans file (%s) not found. Run corresponding nnUNet_plan_experiment " \
                                   "first." % plans_file
        plans = load_json(plans_file)
        plans_manager = PlansManager(plans)
        configuration_manager = plans_manager.get_configuration(configuration_name)

        if self.verbose:
            print(f'Preprocessing the following configuration: {configuration_name}')
        if self.verbose:
            print(configuration_manager)

        dataset_json_file = join(nnUNet_preprocessed, dataset_name, 'dataset.json')
        dataset_json = load_json(dataset_json_file)

        output_directory = join(nnUNet_preprocessed, dataset_name, configuration_manager.data_identifier)

        if isdir(output_directory):
            shutil.rmtree(output_directory)

        maybe_mkdir_p(output_directory)

        dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)

        # identifiers = [os.path.basename(i[:-len(dataset_json['file_ending'])]) for i in seg_fnames]
        # output_filenames_truncated = [join(output_directory, i) for i in identifiers]

        # multiprocessing magic.
        r = []
        with multiprocessing.get_context("spawn").Pool(num_processes) as p:
            remaining = list(range(len(dataset)))
            # p is pretty nifti. If we kill workers they just respawn but don't do any work.
            # So we need to store the original pool of workers.
            workers = [j for j in p._pool]

            for k in dataset.keys():
                r.append(p.starmap_async(self.run_case_save,
                                         ((join(output_directory, k), dataset[k]['images'], dataset[k]['label'],
                                           plans_manager, configuration_manager,
                                           dataset_json),)))

            with tqdm(desc=None, total=len(dataset), disable=self.verbose) as pbar:
                while len(remaining) > 0:
                    all_alive = all([j.is_alive() for j in workers])
                    if not all_alive:
                        raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                           'OK jokes aside.\n'
                                           'One of your background processes is missing. This could be because of '
                                           'an error (look for an error message) or because it was killed '
                                           'by your OS due to running out of RAM. If you don\'t see '
                                           'an error message, out of RAM is likely the problem. In that case '
                                           'reducing the number of workers might help')
                    done = [i for i in remaining if r[i].ready()]
                    # get done so that errors can be raised
                    _ = [r[i].get() for i in done]
                    for _ in done:
                        r[_].get()  # allows triggering errors
                        pbar.update()
                    remaining = [i for i in remaining if i not in done]
                    sleep(0.1)

    def modify_seg_fn(self, seg: np.ndarray, plans_manager: PlansManager, dataset_json: dict,
                      configuration_manager: ConfigurationManager) -> np.ndarray:
        # this function will be called at the end of self.run_case. Can be used to change the segmentation
        # after resampling. Useful for experimenting with sparse annotations: I can introduce sparsity after resampling
        # and don't have to create a new dataset each time I modify my experiments
        return seg


def example_test_case_preprocessing():
    # (paths to files may need adaptations)
    plans_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/nnUNetPlans.json'
    dataset_json_file = '/home/isensee/drives/gpu_data/nnUNet_preprocessed/Dataset219_AMOS2022_postChallenge_task2/dataset.json'
    input_images = ['/home/isensee/drives/e132-rohdaten/nnUNetv2/Dataset219_AMOS2022_postChallenge_task2/imagesTr/amos_0600_0000.nii.gz', ]  # if you only have one channel, you still need a list: ['case000_0000.nii.gz']

    configuration = '3d_fullres'
    pp = DefaultPreprocessor()

    # _ because this position would be the segmentation if seg_file was not None (training case)
    # even if you have the segmentation, don't put the file there! You should always evaluate in the original
    # resolution. What comes out of the preprocessor might have been resampled to some other image resolution (as
    # specified by plans)
    plans_manager = PlansManager(plans_file)
    data, _, properties = pp.run_case(input_images, seg_file=None, plans_manager=plans_manager,
                                      configuration_manager=plans_manager.get_configuration(configuration),
                                      dataset_json=dataset_json_file)

    # voila. Now plug data into your prediction function of choice. We of course recommend nnU-Net's default (TODO)
    return data


if __name__ == '__main__':
    example_test_case_preprocessing()
    # pp = DefaultPreprocessor()
    # pp.run(2, '2d', 'nnUNetPlans', 8)

    ###########################################################################################################
    # how to process a test cases? This is an example:
    # example_test_case_preprocessing()
