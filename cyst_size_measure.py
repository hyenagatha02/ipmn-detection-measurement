import os
import glob
import shutil
import numpy as np
import cv2
import csv
import json
import SimpleITK as sitk

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

## CCA
from scipy.ndimage import binary_dilation
## PCA
from sklearn.decomposition import PCA
## PCA (2d)
from skimage.measure import label, regionprops
from skimage.draw import line, polygon_perimeter  # save annotated mask
## ConvexHull
from scipy.spatial import ConvexHull

from utils import check_files, read_sitk, save_sitk

__all__ = ['save_info_to_csv', 'save_info_to_json', 'save_annotated_mask', 'save_visualization', 'save_visualization_with_overlay', 'slice_tracking_with_convex_hull', 'process_cyst_analysis']


def save_info_to_csv(pca_info, output_csv_path):
    """Save PCA information to a CSV file."""
    if not pca_info:
        return
    
    # headers = ["label", "size_mm3", "height_mm", "width_mm", "depth_mm"]
    # headers = ["label", "total_voxel_count", "largest_slice_idx", "largest_slice_voxel_count", "largest_slice_voxel_area(mm2)", "major_axis_length(mm)", "minor_axis_length(mm)", "centroid", "eigenvalues", "eigenvectors"]
    headers = pca_info[0].keys()
    
    with open(output_csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for row in pca_info:
            writer.writerow(row)
    print(f"\033[36m-- Saved info to: {output_csv_path} --\033[0m")


def save_info_to_json(pca_info, output_json_path, removed_info=[], voxel_spacing=[]):
    """Save PCA information to a JSON file with voxel spacing included."""
    if not pca_info:
        return
        
    def convert_types(obj):
        """Helper function to convert NumPy types to native Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # NumPy array -> list
        elif isinstance(obj, (np.int64, np.int32, np.uint8)):
            return int(obj)  # NumPy integers -> Python int
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)  # NumPy floats -> Python float
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(value) for value in obj]
        else:
            return obj  # Return as is if not a NumPy type
            
    # output_data = {
    #     "voxel_spacing": {
    #         "x": voxel_spacing[0],
    #         "y": voxel_spacing[1],
    #         "z": voxel_spacing[2]
    #     } if voxel_spacing else voxel_spacing,
    #     "labels": convert_types(removed_info) if removed_info else removed_info, # removed_info,
    #     "components": convert_types(pca_info), # pca_info,
    # }
    output_data = {}
    if voxel_spacing:
        output_data["voxel_spacing"] = {
            "x": voxel_spacing[0],
            "y": voxel_spacing[1],
            "z": voxel_spacing[2]
            }
    if removed_info:
        output_data["labels"] = convert_types(removed_info)
    output_data["components"] = convert_types(pca_info)

    with open(output_json_path, mode='w') as jsonfile:
        json.dump(output_data, jsonfile, indent=4)
    print(f"\033[36m-- Saved info to: {output_json_path} --\033[0m")



def save_annotated_mask(connected_component_arr, connected_component_img, label, slice_idx, slice_arr, slice_coords, area_mm2, centroid, # eigenvectors, 
                        major_axis_length, minor_axis_length, bbox_min, bbox_max, major_start, major_end, minor_start, minor_end, hull_points=None, save_folder='', save_name=''):
    """
    Save the largest slice mask with annotations (centroid, PCA axes, bounding box) as a NIfTI file.
    """
    annotated_mask = np.zeros_like(connected_component_arr, dtype=np.uint8)
    annotated_mask[slice_idx] = slice_arr
    
    # Mark centroid
    centroid_idx = np.round(centroid).astype(int)
    centroid_idx = np.clip(centroid_idx, [0, 0], slice_arr.shape)  # Clip to bounds
    annotated_mask[slice_idx, centroid_idx[0], centroid_idx[1]] = 3

    # # Annotate major axis
    # proj_major = np.dot(slice_coords - centroid, eigenvectors[0])
    # major_start = centroid + proj_major.min() * eigenvectors[0]
    # major_end = centroid + proj_major.max() * eigenvectors[0]
    rr_major, cc_major = line(
        int(round(major_start[0])), int(round(major_start[1])),
        int(round(major_end[0])), int(round(major_end[1]))
    )
    annotated_mask[slice_idx, rr_major, cc_major] = 4

    # # Annotate minor axis
    # proj_minor = np.dot(slice_coords - centroid, eigenvectors[1])
    # minor_start = centroid + proj_minor.min() * eigenvectors[1]
    # minor_end = centroid + proj_minor.max() * eigenvectors[1]
    if hull_points is not None and hull_points.size > 0:
        # Compute the major axis unit vector
        major_axis_vector = (major_end - major_start) / np.linalg.norm(major_end - major_start)
        # Compute the shift along the major axis
        shift_vector = np.dot((centroid - minor_start), major_axis_vector) * major_axis_vector
        # Move the entire minor axis in the direction of the major axis
        minor_start_adjusted = minor_start + shift_vector
        minor_end_adjusted = minor_end + shift_vector
        rr_minor, cc_minor = line(
            int(round(minor_start_adjusted[0])), int(round(minor_start_adjusted[1])),
            int(round(minor_end_adjusted[0])), int(round(minor_end_adjusted[1]))
        )
    else:
        rr_minor, cc_minor = line(
            int(round(minor_start[0])), int(round(minor_start[1])),
            int(round(minor_end[0])), int(round(minor_end[1]))
        )
    annotated_mask[slice_idx, rr_minor, cc_minor] = 5

    # Annotate bounding box
    rr_bbox, cc_bbox = polygon_perimeter(
        [bbox_min[0], bbox_min[0], bbox_max[0], bbox_max[0]],
        [bbox_min[1], bbox_max[1], bbox_max[1], bbox_min[1]],
        shape=slice_arr.shape
    )
    annotated_mask[slice_idx, rr_bbox, cc_bbox] = 6

    # Save NIfTI
    os.makedirs(save_folder, exist_ok=True)
    # save_name = f"mask__area_{area_mm2:.2f}_major_{major_axis_length:.2f}_minor_{minor_axis_length:.2f}__label_{label:03d}_slice_{slice_idx+1:03d}_annot.nii.gz"  # _annotated.nii.gz
    if not save_name:
        save_name = f"area_{area_mm2:.2f}_major_{major_axis_length:.2f}_minor_{minor_axis_length:.2f}_label_{label:03d}_slice_{slice_idx+1:03d}_annot.nii.gz"  # _annotated.nii.gz
    slice_output_path = os.path.join(save_folder, save_name)
    
    annotated_sitk_image = sitk.GetImageFromArray(annotated_mask)
    annotated_sitk_image.CopyInformation(connected_component_img)
    sitk.WriteImage(annotated_sitk_image, slice_output_path)

    save_sitk(annotated_mask, connected_component_img, slice_output_path)
    # print(f">> Saved ANNOTATED largest slice mask for label {label} to {slice_output_path}")


def save_visualization(label, slice_idx, slice_arr, slice_coords, area_mm2, # centroid, eigenvectors,
                       major_axis_length, minor_axis_length, major_start, major_end, minor_start, minor_end, save_folder='', save_name=''):
    """
    Save a 2D slice with the largest cross-sectional cyst annotated.
    """
    # plt.ioff()  # Disable interactive mode to prevent auto-display in Jupyter
    plt.figure(figsize=(6, 6))
    plt.gca().set_aspect('equal')
    plt.imshow(slice_arr, cmap="gray")
    plt.scatter(slice_coords[:, 1], slice_coords[:, 0], s=1, color="yellow", alpha=0.3, label="Cyst Region")
    
    # # Annotate PCA axes
    # plt.plot(
    #     [centroid[1] - eigenvectors[0, 1] * major_axis_length / 2,
    #      centroid[1] + eigenvectors[0, 1] * major_axis_length / 2],
    #     [centroid[0] - eigenvectors[0, 0] * major_axis_length / 2,
    #      centroid[0] + eigenvectors[0, 0] * major_axis_length / 2],
    #     "r-", linewidth=1.5, label="Major Axis"
    # )
    # plt.plot(
    #     [centroid[1] - eigenvectors[1, 1] * minor_axis_length / 2,
    #      centroid[1] + eigenvectors[1, 1] * minor_axis_length / 2],
    #     [centroid[0] - eigenvectors[1, 0] * minor_axis_length / 2,
    #      centroid[0] + eigenvectors[1, 0] * minor_axis_length / 2],
    #     "b-", linewidth=1.5, label="Minor Axis"
    # )

    # # Project coordinates onto the eigenvectors to find the extent along each axis
    # proj_major = np.dot(slice_coords - centroid, eigenvectors[0])  # Projection onto the major axis
    # proj_minor = np.dot(slice_coords - centroid, eigenvectors[1])  # Projection onto the minor axis

    # major_start = centroid + proj_major.min() * eigenvectors[0]
    # major_end = centroid + proj_major.max() * eigenvectors[0]

    # minor_start = centroid + proj_minor.min() * eigenvectors[1]
    # minor_end = centroid + proj_minor.max() * eigenvectors[1]

    plt.plot([major_start[1], major_end[1]], [major_start[0], major_end[0]],
             "r-", linewidth=0.8, label="Major Axis")
    plt.plot([minor_start[1], minor_end[1]], [minor_start[0], minor_end[0]],
             "b-", linewidth=0.8, label="Minor Axis")
    

    # Add text annotations
    text_x = int(0.025 * slice_arr.shape[1])
    text_y = int(0.05 * slice_arr.shape[0])  # Start dynamically based on image height
    line_spacing = int(0.05 * slice_arr.shape[0])  # Dynamic spacing based on image height
    plt.text(text_x, text_y, f"Area: {area_mm2:.2f} mm²", color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.7))
    plt.text(text_x, text_y + line_spacing, f"Major Axis length: {major_axis_length:.2f} mm", color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.7))
    plt.text(text_x, text_y + 2 * line_spacing, f"Minor Axis length: {minor_axis_length:.2f} mm", color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.7))

    plt.title(f"Label {label} : Slice {slice_idx+1} - Largest Cyst Cross-Section", fontsize=10.5)
    plt.legend()
    plt.axis("off")
    plt.tight_layout()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        # print(f'-- Created {save_folder}')
    if not save_name:
        save_name = f"area_{area_mm2:.2f}_major_{major_axis_length:.2f}_minor_{minor_axis_length:.2f}_label_{label:03d}_slice_{slice_idx+1:03d}.png"
    save_path = os.path.join(save_folder, save_name)
    plt.savefig(save_path) # , bbox_inches="tight")
    plt.close()


def save_visualization_with_overlay(label, mri_slice, slice_idx, slice_arr, slice_coords, area_mm2, 
                                    major_axis_length, minor_axis_length, bbox_min, bbox_max, 
                                    major_start, major_end, minor_start, minor_end, hull_points=None, voxel_spacing=None, save_folder='', save_name=''):
    """
    Save a 2D overlay of the cyst mask on the MRI slice.
    """
    # plt.ioff()  # Disable interactive mode to prevent auto-display in Jupyter
    plt.figure(figsize=(6, 6), dpi=300)
    plt.gca().set_aspect('equal')
    # plt.imshow(mri_slice, cmap="gray")  # Background MRI image
    # plt.imshow(slice_arr, cmap="Reds", alpha=0.5)  # Overlay cyst mask in red with transparency (Reds, Blues, cool, ...)


    def blend_mask_with_image(mri_slice, mask_slice, color=(255, 0, 0), alpha=0.5):
        # Create an RGB version of the MRI slice
        mri_rgb = np.stack([mri_slice] * 3, axis=-1).astype(np.uint8)
        
        # Create the mask overlay
        mask_overlay = np.zeros_like(mri_rgb, dtype=np.uint8)
        mask_overlay[mask_slice > 0] = np.array(color, dtype=np.uint8) # color  # Apply the desired color to the mask
        
        # Blend the MRI and mask
        mask_alpha = (mask_overlay.sum(axis=-1) > 0).astype(np.float32) * alpha
        blended_image = mri_rgb * (1 - mask_alpha[..., None]) + mask_overlay * mask_alpha[..., None]
        return blended_image.astype(np.uint8)

    def blend_mask_with_image_by_regions(mri_slice, mask_slice, alpha=0.5):
        """
        Blend the MRI slice with the mask. If the mask contains multiple regions,
        assign each region a unique color.
        """
        mri_rgb = np.stack([mri_slice] * 3, axis=-1).astype(np.uint8)  # Convert grayscale MRI to RGB
    
        unique_regions = np.unique(mask_slice)
        unique_regions = unique_regions[unique_regions > 0]  # Ignore background (0)
        
        num_regions = len(unique_regions)
        cmap = sns.color_palette("husl", num_regions)  # Generate distinct colors
        cmap = np.array(cmap) * 255  # Convert to 0-255 scale
    
        mask_overlay = np.zeros((*mask_slice.shape, 3), dtype=np.uint8)  # Ensure it's 3D (H, W, 3)
        mask_alpha = np.zeros(mask_slice.shape, dtype=np.float32)  # Alpha should be 2D (H, W)
    
        for i, region in enumerate(unique_regions):
            color = cmap[i].astype(np.uint8)  # Get distinct color
            region_mask = (mask_slice == region)  # Boolean mask
    
            # Corrected: Use `:` to select all color channels at once
            mask_overlay[region_mask, :] = color  # Assign color to the whole region
            mask_alpha[region_mask] = alpha  # Set transparency
    
        # Blend MRI and colored mask
        blended_image = (mri_rgb * (1 - mask_alpha[..., None]) + mask_overlay * mask_alpha[..., None]).astype(np.uint8)
        return blended_image, num_regions

    
    if '_cca' in save_name:
        blended_image, num_regions = blend_mask_with_image_by_regions(mri_slice, slice_arr, alpha=0.5)  # Magenta (255, 0, 255)
        plt.imshow(blended_image)
    else:
        blended_image = blend_mask_with_image(mri_slice, slice_arr, alpha=0.5)  # Magenta (255, 0, 255)
        plt.imshow(blended_image)
       
    plt.plot([major_start[1], major_end[1]], [major_start[0], major_end[0]], "r-", linewidth=0.8, label="Major Axis")
    # plt.plot([minor_start[1], minor_end[1]], [minor_start[0], minor_end[0]], "g-", linewidth=0.8, label="Minor Axis")

    if hull_points is not None and hull_points.size > 0:
        def line_intersection(p1, p2, q1, q2):
            """Compute intersection of two lines given by points p1->p2 and q1->q2."""
            A1, B1 = p2 - p1
            A2, B2 = q2 - q1
            C1 = A1 * p1[1] - B1 * p1[0]
            C2 = A2 * q1[1] - B2 * q1[0]
        
            det = A1 * B2 - A2 * B1
            if abs(det) < 1e-10:
                return None  # Parallel lines, no intersection
        
            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
            return np.array([y, x])
        
        # Compute the original intersection
        intersection_original = line_intersection(major_start, major_end, minor_start, minor_end)
        
        if intersection_original is not None:
            # Compute the major axis unit vector
            major_axis_vector = (major_end - major_start) / np.linalg.norm(major_end - major_start)
            
            # Compute the shift along the major axis
            hull_centroid = (major_start + major_end) / 2  # np.mean(hull_points, axis=0)
            shift_vector = np.dot((hull_centroid - minor_start), major_axis_vector) * major_axis_vector
            
            # Move the entire minor axis in the direction of the major axis
            minor_start_adjusted = minor_start + shift_vector
            minor_end_adjusted = minor_end + shift_vector
            
            # Compute adjusted intersection
            intersection_adjusted = line_intersection(major_start, major_end, minor_start_adjusted, minor_end_adjusted)
        
            # # Ensure distances remain the same
            # adjusted_top = np.linalg.norm(minor_start_adjusted - intersection_adjusted)
            # adjusted_bottom = np.linalg.norm(minor_end_adjusted - intersection_adjusted)
            # original_top = np.linalg.norm(minor_start - intersection_original)
            # original_bottom = np.linalg.norm(minor_end - intersection_original)
            # print("~~~~~~~~~~~~~~~ Original Top Length:", original_top)
            # print("~~~~~~~~~~~~~~~ Original Bottom Length:", original_bottom)
            # print("~~~~~~~~~~~~~~~ Adjusted Top Length:", adjusted_top)
            # print("~~~~~~~~~~~~~~~ Adjusted Bottom Length:", adjusted_bottom)
            # print("~~~~~~~~", np.isclose(original_top, adjusted_top, atol=1e-3))
            # print("~~~~~~~~", np.isclose(original_bottom, adjusted_bottom, atol=1e-3))
    
            if voxel_spacing:
                adjusted_minor_axis_length = np.linalg.norm(minor_end_adjusted - minor_start_adjusted) * voxel_spacing[0]
                # print("~~~~~~~~ minor_axis_length:", minor_axis_length)
                # print("~~~~~~~~ adjusted_minor_axis_length:", adjusted_minor_axis_length)
                assert np.isclose(minor_axis_length, adjusted_minor_axis_length, atol=1e-6)
                
        # hull_points = np.array(hull_points)
        # plt.plot(hull_points[:, 1], hull_points[:, 0], "m-", linewidth=0.6, label="Convex Hull")
        
        # plt.plot([minor_start_adjusted[1], minor_end_adjusted[1]], [minor_start_adjusted[0], minor_end_adjusted[0]], "b-", linewidth=0.8, label="Minor Axis")
        plt.plot([minor_start_adjusted[1], minor_end_adjusted[1]], [minor_start_adjusted[0], minor_end_adjusted[0]], "g-", linewidth=0.8, label="Minor Axis")
    else:
        print(f"! No hull_points available for visualization in Label {label}, Slice {slice_idx+1}.")
        
    
    '''
    - The major and minor axes derived from PCA represent the directions of maximum and minimum variance (spread) of the points within the cyst region. 
      They are mathematical abstractions based on the distribution of the points and are centered around the centroid.
    - Axes are not meant to represent the maximal extent of the cyst but rather its principal orientation and elongation.
      It depends on the variance (eigenvalues) along the respective eigenvector directions, and the variance reflects how the points are distributed, not the maximal distance from the centroid to the border.
    '''
    plt.plot(
        [bbox_min[1], bbox_max[1], bbox_max[1], bbox_min[1], bbox_min[1]],
        [bbox_min[0], bbox_min[0], bbox_max[0], bbox_max[0], bbox_min[0]],
        "c--", linewidth=0.8, label="Bounding Box"
    )

    # Add annotations
    text_x = int(0.025 * mri_slice.shape[1])
    text_y = int(0.05 * mri_slice.shape[0])  # Start dynamically based on image height
    line_spacing = int(0.05 * mri_slice.shape[0])  # Dynamic spacing based on image height
    plt.text(text_x, text_y, f"Area: {area_mm2:.2f} mm²", color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.7))
    plt.text(text_x, text_y + line_spacing, f"Major Axis: {major_axis_length:.2f} mm", color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.7))
    plt.text(text_x, text_y + 2 * line_spacing, f"Minor Axis: {minor_axis_length:.2f} mm", color="white", fontsize=10, bbox=dict(facecolor="black", alpha=0.7))
    if '_cca' in save_name:
        # print('~~~~cca~~~~ ', save_name)
        plt.text(text_x, text_y + 3 * line_spacing, f"CCA: {num_regions} regions", color="yellow", fontsize=10, bbox=dict(facecolor="black", alpha=0.7))
        plt.title(f"Label {label} : Slice {slice_idx+1} - Largest Axis Cyst (CCA) ", fontsize=10.5)
        # plt.title(f"Label {label} : Slice {slice_idx+1} - Largest Area Cyst (CCA) ", fontsize=10.5)
    elif '_region' in save_name:
        # print('~~~~region~~~~ ', save_name)
        region_id = save_name.split('region')[1].split('_')[1]
        plt.text(text_x, text_y + 3 * line_spacing, f"region id: {region_id} ", color="lime", fontsize=10, bbox=dict(facecolor="black", alpha=0.7))
        plt.title(f"Label {label} : Slice {slice_idx+1} - Largest Axis Cyst Region (CCA)", fontsize=10.5)
        # plt.title(f"Label {label} : Slice {slice_idx+1} - Largest Area Cyst (CCA)", fontsize=10.5)
    else: 
        # print('~~~~else~~~~ ', save_name)
        plt.title(f"Label {label} : Slice {slice_idx+1} - Largest Axis Cyst", fontsize=10.5)
        # plt.title(f"Label {label} : Slice {slice_idx+1} - Largest Area Cyst", fontsize=10.5)
    plt.legend()
    plt.axis("off")
    plt.tight_layout()

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        # print(f'(Created {save_folder})')
    if not save_name:
        save_name = f"area_{area_mm2:.2f}_major_{major_axis_length:.2f}_minor_{minor_axis_length:.2f}_label_{label:03d}_slice_{slice_idx+1:03d}.png"
    save_path = os.path.join(save_folder, save_name)
    plt.savefig(save_path) # , bbox_inches="tight")
    plt.close()

    # _png = cv2.imread(save_path)
    # print(_png.shape)
    return save_path



def sklearn_principal_component_analysis(slice_coords, voxel_spacing):
    pca = PCA(n_components=2)
    pca.fit(slice_coords)

    centroid = np.mean(slice_coords, axis=0)
    eigenvectors = pca.components_
    # eigenvalues = pca.explained_variance_
    # assert eigenvalues[0] >= eigenvalues[1]

    # major_axis_length = 2 * np.sqrt(eigenvalues[0]) * voxel_spacing[1]  # Major axis in mm
    # minor_axis_length = 2 * np.sqrt(eigenvalues[1]) * voxel_spacing[0]  # Minor axis in mm
    # assert major_axis_length >= minor_axis_length
    
    # Project all points onto each eigenvector
    proj_major = np.dot(slice_coords - centroid, eigenvectors[0])
    proj_minor = np.dot(slice_coords - centroid, eigenvectors[1])

    # Compute actual lengths based on projection range
    major_axis_length = (proj_major.max() - proj_major.min()) * voxel_spacing[1]
    minor_axis_length = (proj_minor.max() - proj_minor.min()) * voxel_spacing[0]

    # Compute start and end points for each axis
    major_start = centroid + proj_major.min() * eigenvectors[0]
    major_end = centroid + proj_major.max() * eigenvectors[0]
    minor_start = centroid + proj_minor.min() * eigenvectors[1]
    minor_end = centroid + proj_minor.max() * eigenvectors[1]

    # return major_axis_length, minor_axis_length, centroid, eigenvectors, eigenvalues
    return major_axis_length, minor_axis_length, major_start, major_end, minor_start, minor_end, centroid, eigenvectors

def scipy_convex_hull(slice_coords, voxel_spacing):
    
    try:
        ## compute Major Axis (longest distance in Convex Hull)
        hull = ConvexHull(slice_coords)
        hull_points = slice_coords[hull.vertices]

        max_dist = 0
        max_start, max_end = None, None
        for i in range(len(hull_points)):
            for j in range(i + 1, len(hull_points)):
                hull_dist = np.linalg.norm(hull_points[i] - hull_points[j])
                if hull_dist > max_dist:
                    max_dist = hull_dist
                    max_start, max_end = hull_points[i], hull_points[j]
        
        major_axis_length = max_dist * voxel_spacing[1]  # Convert to mm
        major_start, major_end = max_start, max_end
        major_start, major_end = np.array(max_start, dtype=np.float64), np.array(max_end, dtype=np.float64)
        

        ## compute Minor Axis (projection perpendicular to major axis)
        major_axis_vector = (max_end - max_start) / np.linalg.norm(max_end - max_start)
        minor_axis_vector = np.array([-major_axis_vector[1], major_axis_vector[0]])  # Perpendicular to major axis
        
        minor_projection_values = np.dot(hull_points - max_start, minor_axis_vector)
        
        min_index = np.argmin(minor_projection_values)
        max_index = np.argmax(minor_projection_values)
        
        # minor_start = hull_points[min_index]
        # minor_end = hull_points[max_index]
        minor_start = major_start + minor_projection_values[min_index] * minor_axis_vector
        minor_end = major_start + minor_projection_values[max_index] * minor_axis_vector
        minor_start, minor_end = np.array(minor_start, dtype=np.float64), np.array(minor_end, dtype=np.float64)
        
        minor_axis_length = np.linalg.norm(minor_end - minor_start) * voxel_spacing[0]  # Convert to mm
        
        centroid = (major_start + major_end) / 2  # np.mean(hull_points, axis=0)
        centroid = np.array(centroid, dtype=np.float64)
        
        # print(type(major_axis_length), major_axis_length.dtype)  # <class 'numpy.float64'> float64
        # print(type(minor_axis_length), minor_axis_length.dtype)  # <class 'numpy.float64'> float64
        # print(type(major_start), major_start.dtype)              # <class 'numpy.ndarray'> float64
        # print(type(major_end), major_end.dtype)                  # <class 'numpy.ndarray'> float64
        # print(type(minor_start), minor_start.dtype)              # <class 'numpy.ndarray'> float64
        # print(type(minor_end), minor_end.dtype)                  # <class 'numpy.ndarray'> float64
        # print(type(centroid), centroid.dtype)                    # <class 'numpy.ndarray'> float64
        
        dot_product_check = np.dot(major_axis_vector, (minor_end - minor_start) / np.linalg.norm(minor_end - minor_start))
        # print('###################################################', dot_product_check)

    except Exception as e:
        '''
        QH6154 - Initial Simplex is Flat
            : all points lie in a single line (collinear points) or are too close to each other  >>  requires at least 3 non-collinear points
        QH6154 - Roundoff Error (precision error)
            : the floating-point values of the coordinates are too small or too close (the cyst is very small, only have a few pixels)
        QH6013 - Input is Less Than 2D
            : all points have the same X-coordinate or Y-coordinate (exist on a single vertical or horizontal line, a 1D line) 
        '''
        # if len(np.unique(slice_coords, axis=0)) < 3:
        #     # raise ValueError("Not enough unique points for Convex Hull")
        #     print(f"Convex Hull cannot be computed (less than 3 unique points) ({len(np.unique(slice_coords, axis=0))})")
        # if np.ptp(slice_coords[:, 0]) == 0 or np.ptp(slice_coords[:, 1]) == 0:
        #     # raise ValueError("Points form a 1D line, cannot compute Convex Hull")
        #     print(f"Convex Hull cannot be computed (points form a 1D line) ({np.ptp(slice_coords[:, 0])}, {np.ptp(slice_coords[:, 1])})")
        # # print('>> ', e)
        
        major_axis_length = 0  # None
        minor_axis_length = 0  # None
        major_start, major_end = None, None
        minor_start, minor_end = None, None
        centroid = None
        hull_points = None     # np.array([])

    return major_axis_length, minor_axis_length, major_start, major_end, minor_start, minor_end, centroid, hull_points


def slice_tracking_with_convex_hull(fid, label, voxel_spacing, connected_component_img, connected_component_arr, 
                                                       img_path, save_path, min_size_limit=5):

    """1) get voxel indices (coordinates)"""
    voxel_coords = np.argwhere(connected_component_arr == label)  # (z, y, x)
    # print('- total voxel_coords :', voxel_coords)  # total voxel_coords : [[  9 130 105] [  9 131 105] [  9 131 106] [  9 132 105] [  9 132 106] [ 10 129 105] [ 10 129 106] ...]
    # print('- total voxel_coords :', len(voxel_coords))  # total voxel_coords : 136
    assert len(voxel_coords) >= 3

    """2) process coords by slice"""
    ## coords number
    slice_areas = {}
    for coord in voxel_coords:
        z = coord[0]
        slice_areas[z] = slice_areas.get(z, 0) + 1
    # print('- slice_areas :', slice_areas)  # slice_areas : {9: 5, 10: 70, 11: 61}
    
    ## coords area (mm2)
    slice_areas_mm2 = {z : c * voxel_spacing[0] * voxel_spacing[1] for z, c in slice_areas.items()}
    # print('- slice_areas_mm2 : ', slice_areas_mm2)  # slice_areas_mm2 : {9: 7.8125, 10: 109.375, 11: 95.3125}

    
    """3) extract largest 2D slice for PCA (based on major_axis length)"""
    slice_measurements = {}
    for z, _ in slice_areas.items():
        slice_arr = (connected_component_arr[z, :, :] == label).astype(np.uint8)
        slice_coords = np.argwhere(slice_arr > 0)  # 2D coordinates (y, x)

        if len(slice_coords) < 3:  # PCA requires at least 3 points
            # print(f" ! (Skipping Label {label} - Slice {z} : too few points for analysis)")
            continue    
        
        major_axis_length_pca, minor_axis_length_pca, major_start_pca, major_end_pca, minor_start_pca, minor_end_pca, centroid_pca, eigenvectors = sklearn_principal_component_analysis(slice_coords, voxel_spacing)
        major_axis_length_hull, minor_axis_length_hull, major_start_hull, major_end_hull, minor_start_hull, minor_end_hull, centroid_hull, hull_points = scipy_convex_hull(slice_coords, voxel_spacing)
        # print(f"\n++++++ Label {label} - Slice {z} ++++++")
        # print(f" minor_axis_length_pca  : {minor_axis_length_pca} mm")
        # print(f" major_axis_length_pca  : {major_axis_length_pca} mm")
        # print(f" minor_axis_length_hull : {minor_axis_length_hull} mm")
        # print(f" major_axis_length_hull : {major_axis_length_hull} mm")
        
        if major_axis_length_hull==0 or minor_axis_length_hull==0:  # and
            # print(f" ! (Failed Label {label} - Slice {z} : major_axis {major_axis_length_hull}, minor_axis {minor_axis_length_hull})")
            continue
        else:
            slice_measurements[z] = {
                "area" : slice_areas_mm2[z],
                ## PCA
                "minor_axis_length_pca": minor_axis_length_pca,
                "major_axis_length_pca": major_axis_length_pca,
                "major_start_pca": major_start_pca,
                "major_end_pca": major_end_pca,
                "minor_start_pca": minor_start_pca,
                "minor_end_pca": minor_end_pca,
                "centroid_pca": centroid_pca,
                "eigenvectors": eigenvectors,
                ## convex hull
                "major_axis_length_hull": major_axis_length_hull,
                "minor_axis_length_hull": minor_axis_length_hull,
                "major_start_hull": major_start_hull,
                "major_end_hull": major_end_hull,
                "minor_start_hull": minor_start_hull,
                "minor_end_hull": minor_end_hull,
                "centroid_hull": centroid_hull,
                "hull_points" : hull_points,
            }

    print(f"\n++++++ slice_measurements \033[1m(Label {label})\033[0m ++++++")
    if not slice_measurements:
        print(f" ! No valid slices found for Label {label}. Skip this component !")
        # # return None
        return None, None  ## area_ratio
    
    for z, c in slice_measurements.items():
        print(f"slice {z} :")
        # for c_k, c_v in c.items():
        #     print(f" {c_k} : {c_v}")
        # print(f" major_axis_length_pca  : {c['major_axis_length_pca']} ")
        # print(f" minor_axis_length_pca  : {c['minor_axis_length_pca']} ")
        print(f" major_axis_length_hull : {c['major_axis_length_hull']} ")
        print(f" minor_axis_length_hull : {c['minor_axis_length_hull']} ")
        print(f" area                   : {c['area']} ")

    # _valid_slices = [info for info in slice_measurements.values() if info["major_axis_length_pca"] > 0 and info["minor_axis_length_pca"] > 0]
    _valid_slices = [info for info in slice_measurements.values() if info["major_axis_length_hull"] > 0 and info["minor_axis_length_hull"] > 0]
    # print(f"++++++++++++ _valid_slices (Label {label}) ++++++++++++")
    # for s_c in _valid_slices:
    #     print(f"slice {s} :")
    #     print(s_c)
    assert len(_valid_slices) == len(slice_measurements), print(len(_valid_slices), len(slice_measurements))
    
    if not _valid_slices:
        # _max_major = max((info["major_axis_length_pca"] for info in slice_measurements.values()), default=0)
        # _max_minor = max((info["minor_axis_length_pca"] for info in slice_measurements.values()), default=0)
        # print(f" ! No valid Convex Hull measurements for Label {label} !")
        # print(f"   Max major_axis_length_hull: {_max_major:.2f} mm")
        # print(f"   Max minor_axis_length_hull: {_max_minor:.2f} mm")
        _max_major = max((info["major_axis_length_hull"] for info in slice_measurements.values()), default=0)
        _max_minor = max((info["minor_axis_length_hull"] for info in slice_measurements.values()), default=0)
        print(f" ! No valid Convex Hull measurements for Label {label} !")
        print(f"   Max major_axis_length_hull: {_max_major:.2f} mm")
        print(f"   Max minor_axis_length_hull: {_max_minor:.2f} mm")
        
        # return None
        return None, None  ## area_ratio

        
    """4) Calculate Lagest slice information"""
    ## largest : largest major_axis
    # largest_slice_idx = max(slice_measurements, key=lambda z: slice_measurements[z]["major_axis_length_pca"])
    largest_slice_idx = max(slice_measurements, key=lambda z: slice_measurements[z]["major_axis_length_hull"])
    ## largest : largest area
    # largest_slice_idx = max(slice_measurements, key=lambda z: slice_areas_mm2[z])

    # largest_major_axis_length = slice_measurements[largest_slice_idx]["major_axis_length_pca"]
    # largest_minor_axis_length = slice_measurements[largest_slice_idx]["minor_axis_length_pca"]
    largest_major_axis_length = slice_measurements[largest_slice_idx]["major_axis_length_hull"]
    largest_minor_axis_length = slice_measurements[largest_slice_idx]["minor_axis_length_hull"]
    
    # largest_major_start = slice_measurements[largest_slice_idx]["major_start_pca"]
    # largest_major_end = slice_measurements[largest_slice_idx]["major_end_pca"]
    # largest_minor_start = slice_measurements[largest_slice_idx]["minor_start_pca"]
    # largest_minor_end = slice_measurements[largest_slice_idx]["minor_end_pca"]
    # largest_centroid = slice_measurements[largest_slice_idx]["centroid_pca"]
    # largest_eigenvectors = slice_measurements[largest_slice_idx]["eigenvectors"]
    largest_major_start = slice_measurements[largest_slice_idx]["major_start_hull"]
    largest_major_end = slice_measurements[largest_slice_idx]["major_end_hull"]
    largest_minor_start = slice_measurements[largest_slice_idx]["minor_start_hull"]
    largest_minor_end = slice_measurements[largest_slice_idx]["minor_end_hull"]
    largest_centroid = slice_measurements[largest_slice_idx]["centroid_hull"]
    largest_hull_points = slice_measurements[largest_slice_idx]["hull_points"]
    
    largest_slice_voxel_count = slice_areas[largest_slice_idx]
    largest_slice_voxel_area = slice_areas_mm2[largest_slice_idx]
    largest_slice_arr = (connected_component_arr[largest_slice_idx, :, :] == label).astype(np.uint8)
    largest_slice_coords = np.argwhere(largest_slice_arr > 0)  # 2D coordinates (y, x)

    # bbox_size = slice_coords.ptp(axis=0) * voxel_spacing[:2]
    bbox_min = largest_slice_coords.min(axis=0) # * voxel_spacing[:2]
    bbox_max = largest_slice_coords.max(axis=0) # * voxel_spacing[:2]
    bbox_size = (bbox_max - bbox_min) * voxel_spacing[:2]
    bbox_area = bbox_size[0] * bbox_size[1]
    bbox_diagonal = np.linalg.norm(bbox_size)
    # print('- bbox_min, bbox_max (pixel coords)', bbox_min, bbox_max)
    # print('- bbox_min, bbox_max (physical dimensions, mm)', bbox_min* voxel_spacing[:2], bbox_max* voxel_spacing[:2])
    # print('- bbox_size (physical dimensions, mm)', bbox_size)
    # print('- bbox_area (physical dimensions, mm2)', bbox_area)
    # print('- bbox_diagonal (physical dimensions, mm)', bbox_diagonal)


    # _largest_slice_idx = max(slice_measurements, key=lambda z: slice_measurements[z]["major_axis_length_hull"])
    # print(f">>>> largest_slice_idx (pca)  : {largest_slice_idx} <<<<")
    # print(f">>>> largest_slice_idx (hull) : {_largest_slice_idx} <<<<")
    # # assert _largest_slice_idx == largest_slice_idx
    # _largest_major_axis_length = slice_measurements[largest_slice_idx]["major_axis_length_hull"]
    # _largest_minor_axis_length = slice_measurements[largest_slice_idx]["minor_axis_length_hull"]
    _largest_slice_idx = max(slice_measurements, key=lambda z: slice_measurements[z]["major_axis_length_pca"])
    # print(f">>>> largest_slice_idx (pca)  : {_largest_slice_idx} <<<<")
    # print(f">>>> largest_slice_idx (hull) : {largest_slice_idx} <<<<")
    # assert _largest_slice_idx == largest_slice_idx
    _largest_major_axis_length = slice_measurements[largest_slice_idx]["major_axis_length_pca"]
    _largest_minor_axis_length = slice_measurements[largest_slice_idx]["minor_axis_length_pca"]

    
    ## get area_ratio (largest area cyst vs largest axis cyst)
    max_area_slice_idx = max(slice_measurements, key=lambda z: slice_areas_mm2[z])
    max_area = slice_areas_mm2[max_area_slice_idx]
    area_ratio = largest_slice_voxel_area / max_area
    if not np.isclose(area_ratio, 1.0, atol=1e-6):
        # print(f"  **** Label {label} - Slice {largest_slice_idx}: \033[1marea_ratio : {area_ratio}\033[0m (max area {max_area}  / largest slice area {largest_slice_voxel_area})")
        print(f" area_ratio             : {area_ratio} (max area {max_area}  / largest slice area {largest_slice_voxel_area})")
    # max_axis_slice_idx = max(slice_measurements, key=lambda z: slice_measurements[z]["major_axis_length_hull"])
    # max_axis_area = slice_areas_mm2[max_axis_slice_idx]
    # area_ratio = max_axis_area / largest_slice_voxel_area
    # if not np.isclose(area_ratio, 1.0, atol=1e-6):
    #     print(f"  **** Label {label} - Slice {largest_slice_idx}: \033[1marea_ratio : {area_ratio}\033[0m (max axis area {max_axis_area}  / largest slice area {largest_slice_voxel_area})")
    #     print(f" area_ratio             : {area_ratio} (max axis area {max_axis_area}  / largest slice area {largest_slice_voxel_area})")


    """5) (visualization)"""
    # # visualization of the largest slice (mask only)
    # save_folder = f"{save_path}/slice"
    # save_visualization(
    #     label, largest_slice_idx, largest_slice_arr, largest_slice_coords, largest_slice_voxel_area, # largest_centroid, largest_eigenvectors,
    #     largest_major_axis_length, largest_minor_axis_length, largest_major_start, largest_major_end, largest_minor_start, largest_minor_end, save_folder=save_folder
    # )

    # # Save visualization of the largest slice (mask overlaied) >> only size more than 5mm
    # if largest_slice_voxel_area >= 20:
    def normalize_image(image_array, norm_values=(0, 400)):
        # min_val, max_val = norm_values
        # min_val, max_val = np.min(image_array), np.max(image_array)
        min_val, max_val = np.min(image_array), np.percentile(image_array, 99)
        # print('- (vis) min_val, max_val :', np.min(image_array), np.max(image_array))
        # print('- (vis) clipped min_val, max_val :', min_val, max_val)
        image_array = np.clip(image_array, min_val, max_val)
        return (image_array - min_val) / (max_val - min_val)
        
    mri_img, mri_arr = read_sitk(img_path)
    mri_arr_normalized = normalize_image(mri_arr)
    mri_arr_normalized = (mri_arr_normalized * 255).astype(np.uint8)  # Convert to uint8 for RGB blending
    mri_slice = mri_arr_normalized[largest_slice_idx]

    save_folder_overlayed = f"{save_path}/slice"  # slice_overlayed"  # f"{save_path}/slice_overlayed_20mm2"
    # save_path_overlayed = save_visualization_with_overlay(
    #     label, mri_slice, largest_slice_idx, largest_slice_arr, largest_slice_coords, largest_slice_voxel_area,
    #     largest_major_axis_length, largest_minor_axis_length, bbox_min, bbox_max, 
    #     largest_major_start, largest_major_end, largest_minor_start, largest_minor_end, 
    #     save_folder=save_folder_overlayed, 
    # )
    save_path_overlayed = save_visualization_with_overlay(
        label, mri_slice, largest_slice_idx, largest_slice_arr, largest_slice_coords, largest_slice_voxel_area,
        largest_major_axis_length, largest_minor_axis_length, bbox_min, bbox_max, 
        largest_major_start, largest_major_end, largest_minor_start, largest_minor_end, 
        hull_points=largest_hull_points, 
        voxel_spacing=voxel_spacing,
        save_folder=save_folder_overlayed, 
    )
    # print(f'>> Saved OVERLAYED slice mask to {save_path_overlayed}')
    
    # Extract large cyst fit to condition
    if largest_major_axis_length >= 10 or largest_minor_axis_length >= 10:
        save_folder_overlayed_10mm = f"{save_path}/slice_10mm"  # slice_overlayed_over_10mm"  # f"{save_path}/slice_overlayed_20mm2_10mm"
        if not os.path.exists(save_folder_overlayed_10mm):
            os.makedirs(save_folder_overlayed_10mm)
            # print(f'(Created {save_folder_overlayed_10mm})')
        shutil.copyfile(save_path_overlayed, save_path_overlayed.replace(save_folder_overlayed, save_folder_overlayed_10mm))
        # print(f">> Save over 10mm OVERLAYED slice mask to {save_path_overlayed.replace(save_folder_overlayed, save_folder_overlayed_10mm)})") 


    """6) (save largest slice mask)"""
    if largest_major_axis_length >= 10 or largest_minor_axis_length >= 10:
        print()
        print(f'\033[1m >> largest_slice_idx : {largest_slice_idx}\033[0m')
        print(f'\033[1m >> largest_major_axis {largest_major_axis_length}\033[0m')
        print(f'\033[1m >> largest_minor_axis {largest_minor_axis_length}\033[0m')

        largest_slice_mask = np.zeros_like(connected_component_arr, dtype=np.uint8)
        largest_slice_mask[largest_slice_idx] = largest_slice_arr
        
        save_folder_overlayed_10mm = f"{save_path}/slice_10mm"  # slice_overlayed_over_10mm"  # f"{save_path}/slice_overlayed_20mm2_10mm"
        if not os.path.exists(save_folder_overlayed_10mm):
            os.makedirs(save_folder_overlayed_10mm)
        save_name_10mm = f"area_{largest_slice_voxel_area:.2f}_major_{largest_major_axis_length:.2f}_minor_{largest_minor_axis_length:.2f}_label_{label:03d}_slice_{largest_slice_idx+1:03d}.nii.gz"
        save_path_10mm = os.path.join(save_folder_overlayed_10mm, save_name_10mm)
        save_sitk(largest_slice_mask, connected_component_img, save_path_10mm)
        # print(f">> Saved largest slice mask for Label {label} to {save_path_10mm}")

        # save_annotated_mask(
        #     connected_component_arr, connected_component_img, label, largest_slice_idx, largest_slice_arr, largest_slice_coords, largest_slice_voxel_area, largest_centroid,  # largest_eigenvectors, 
        #     largest_major_axis_length, largest_minor_axis_length, bbox_min, bbox_max, 
        #     largest_major_start, largest_major_end, largest_minor_start, largest_minor_end, 
        #     save_folder=save_folder_overlayed_10mm
        # )
        save_annotated_mask(
            connected_component_arr, connected_component_img, label, largest_slice_idx, largest_slice_arr, largest_slice_coords, largest_slice_voxel_area, largest_centroid,  # largest_eigenvectors, 
            largest_major_axis_length, largest_minor_axis_length, bbox_min, bbox_max, 
            largest_major_start, largest_major_end, largest_minor_start, largest_minor_end, 
            hull_points=largest_hull_points,
            save_folder=save_folder_overlayed_10mm
        )

        # print('- total_voxel_count :', len(voxel_coords))  # total voxel count (3d)
        # print('- slice_areas :', slice_areas)  # slice_areas : {9: 5, 10: 70, 11: 61}
        # print('- slice_areas_mm2 : ', slice_areas_mm2)  # slice_areas_mm2 : {9: 7.8125, 10: 109.375, 11: 95.3125}
        # print('- largest_slice_idx :', largest_slice_idx)  # largest_slice_idx, largest_slice_voxel_area : 10 109.375
        # print('- largest_slice_voxel_count :', largest_slice_voxel_count)
        # print('- largest_slice_voxel_area :', largest_slice_voxel_area)
        # print('- largest_slice_area_coords :', len(largest_slice_coords))  # total voxel_coords in slice (2d) : 136 
        # # print('- largest_centroid', largest_centroid)
        # # print('- largest_eigenvectors', largest_eigenvectors)
        # # print('- largest_eigenvalues', largest_eigenvalues)
        # print('- largest_major_axis_length (Convex Hull) :', largest_major_axis_length)
        # print('- largest_minor_axis_length (Convex Hull) :', largest_minor_axis_length)
        # print('- largest_major_axis_length (PCA)         :', _largest_major_axis_length)
        # print('- largest_minor_axis_length (PCA)         :', _largest_minor_axis_length)

        
        # """6-1) (save CCA processed largest slice mask)"""
        # save_folder_overlayed_10mm_cca = f"{save_path}/slice_10mm_cca"  # slice_overlayed_over_10mm_cca"  # f"{save_path}/slice_overlayed_20mm2_10mm"
        # if not os.path.exists(save_folder_overlayed_10mm_cca):
        #     os.makedirs(save_folder_overlayed_10mm_cca)
        #     # print(f'(Created {save_folder_overlayed_10mm_cca})')
            
        # # Run CCA to count separate cyst regions
        # num_labels, region_labels, _, _ = cv2.connectedComponentsWithStats(largest_slice_arr, connectivity=8)
        # num_regions = num_labels - 1  # ignore background (label 0)
        # # print(f'****** Label {label} - Slice {largest_slice_idx}: CCA with {num_regions} separated cyst regions ******')
        # print(f'\033[1m >> CCA regions        {num_regions}\033[0m')

        # largest_slice_cca_mask = np.zeros_like(connected_component_arr, dtype=np.uint8)
        # largest_slice_cca_mask[largest_slice_idx] = region_labels
        
        # save_name_cca = f"area_{largest_slice_voxel_area:.2f}_major_{largest_major_axis_length:.2f}_minor_{largest_minor_axis_length:.2f}_label_{label:03d}_slice_{largest_slice_idx+1:03d}_cca.nii.gz"
        # save_path_cca = os.path.join(save_folder_overlayed_10mm_cca, save_name_cca)
        # save_sitk(largest_slice_cca_mask, connected_component_img, save_path_cca)
        # # print(f">> Saved SEPERATED 2d CCA on largest slice mask for Label {label} to {save_path_cca}")
        
        # # save_annotated_mask(
        # #     connected_component_arr, connected_component_img, label, largest_slice_idx, region_labels, largest_slice_coords, largest_slice_voxel_area, largest_centroid,  # largest_eigenvectors, 
        # #     largest_major_axis_length, largest_minor_axis_length, bbox_min, bbox_max, 
        # #     largest_major_start, largest_major_end, largest_minor_start, largest_minor_end, 
        # #     save_folder=save_folder_overlayed_10mm_cca,
        # #     save_name=save_name_cca.replace('.nii.gz', '_annot.nii.gz')  # '_annotated.nii.gz'
        # # )
        # save_annotated_mask(
        #     connected_component_arr, connected_component_img, label, largest_slice_idx, region_labels, largest_slice_coords, largest_slice_voxel_area, largest_centroid,  # largest_eigenvectors, 
        #     largest_major_axis_length, largest_minor_axis_length, bbox_min, bbox_max, 
        #     largest_major_start, largest_major_end, largest_minor_start, largest_minor_end, 
        #     hull_points=largest_hull_points,
        #     save_folder=save_folder_overlayed_10mm_cca,
        #     save_name=save_name_cca.replace('.nii.gz', '_annot.nii.gz')  # '_annotated.nii.gz'
        # )

        # # save_visualization_with_overlay(
        # #     label, mri_slice, largest_slice_idx, region_labels, largest_slice_coords, largest_slice_voxel_area,
        # #     largest_major_axis_length, largest_minor_axis_length, bbox_min, bbox_max, 
        # #     largest_major_start, largest_major_end, largest_minor_start, largest_minor_end, 
        # #     save_folder=save_folder_overlayed_10mm_cca, 
        # #     save_name=save_name_cca.replace('.nii.gz', '.png')
        # # )
        # save_visualization_with_overlay(
        #     label, mri_slice, largest_slice_idx, region_labels, largest_slice_coords, largest_slice_voxel_area,
        #     largest_major_axis_length, largest_minor_axis_length, bbox_min, bbox_max, 
        #     largest_major_start, largest_major_end, largest_minor_start, largest_minor_end, 
        #     hull_points=largest_hull_points,
        #     voxel_spacing=voxel_spacing,
        #     save_folder=save_folder_overlayed_10mm_cca, 
        #     save_name=save_name_cca.replace('.nii.gz', '.png')
        # )

        # """6-2) (save each separated CCA region as an individual mask & image)"""
        # region_measurements = {}
        # if num_regions > 1:
        #     save_folder_region = os.path.join(save_folder_overlayed_10mm_cca, f"label_{label:03d}_slice_{largest_slice_idx+1:03d}_cca")
        #     if not os.path.exists(save_folder_region):
        #         os.makedirs(save_folder_region)
            
        #     region_measurements = []
        #     for region_id in range(1, num_regions + 1):  # Skip background (0)
                
        #         region_arr = (region_labels == region_id).astype(np.uint8)
        #         region_coords = np.argwhere(region_arr > 0)
            
        #         if len(region_coords) < 3:  # Ensure valid computation
        #             # print(f" ! (Skipping Region {region_id} : too few points for analysis)")
        #             continue 
    
        #         # Compute PCA & Convex Hull measurements
        #         region_major_axis_length_pca, region_minor_axis_length_pca, region_major_start_pca, region_major_end_pca, region_minor_start_pca, region_minor_end_pca, region_centroid_pca, region_eigenvectors = sklearn_principal_component_analysis(region_coords, voxel_spacing)
        #         region_major_axis_length_hull, region_minor_axis_length_hull, region_major_start_hull, region_major_end_hull, region_minor_start_hull, region_minor_end_hull, region_centroid_hull, region_hull_points = scipy_convex_hull(region_coords, voxel_spacing)

        #         if region_major_axis_length_hull==0 and region_minor_axis_length_hull==0:
        #             # print(f" ! (Failed Convex Hull for Region {region_id} : major_axis {region_major_axis_length_hull}, minor_axis {region_minor_axis_length_hull})")
        #             continue
                    
        #         region_voxel_count = len(region_coords)
        #         region_voxel_area = len(region_coords) * voxel_spacing[0] * voxel_spacing[1]
                
        #         region_measurements.append({
        #             "label": label,
        #             "slice_idx": largest_slice_idx+1,  # largest_slice_idx
        #             # "slice_voxel_count": largest_slice_voxel_count,
        #             # "slice_voxel_area(mm2)": largest_slice_voxel_area,
        #             "region": region_id,
        #             "region_voxel_count": region_voxel_count,
        #             "region_voxel_area(mm2)": region_voxel_area,
        #             ## PCA
        #             # "minor_axis_length(mm)": region_minor_axis_length_pca,
        #             # "major_axis_length(mm)": region_major_axis_length_pca,
        #             ## convex hull
        #             "major_axis_length(mm)": region_major_axis_length_hull,
        #             "minor_axis_length(mm)": region_minor_axis_length_hull,
        #         })
                
        #         region_bbox_min = region_coords.min(axis=0) # * voxel_spacing[:2]
        #         region_bbox_max = region_coords.max(axis=0) # * voxel_spacing[:2]
        #         region_bbox_size = (region_bbox_max - region_bbox_min) * voxel_spacing[:2]
        #         region_bbox_area = region_bbox_size[0] * region_bbox_size[1]
        #         region_bbox_diagonal = np.linalg.norm(region_bbox_size)
    
        #         # Save each region as an individual NIfTI mask
        #         region_mask = np.zeros_like(connected_component_arr, dtype=np.uint8)
        #         region_mask[largest_slice_idx] = region_arr
    
        #         # save_name_region = f"area_{region_voxel_area:.2f}_major_{region_major_axis_length_pca:.2f}_minor_{region_minor_axis_length_pca:.2f}_region_{region_id:02d}_slice_{largest_slice_idx+1:03d}.nii.gz"
        #         save_name_region = f"area_{region_voxel_area:.2f}_major_{region_major_axis_length_hull:.2f}_minor_{region_minor_axis_length_hull:.2f}_region_{region_id:02d}_slice_{largest_slice_idx+1:03d}.nii.gz"
        #         save_path_region = os.path.join(save_folder_region, save_name_region)
        #         save_sitk(region_mask, connected_component_img, save_path_region)
        #         # print(f">> Saved region {region_id} of Label {label} to {save_path_region}")

        #         # save_annotated_mask(
        #         #     connected_component_arr, connected_component_img, label, largest_slice_idx, region_arr, region_coords, region_voxel_area, region_centroid_pca,  # region_eigenvectors, 
        #         #     region_major_axis_length_pca, region_minor_axis_length_pca, region_bbox_min, region_bbox_max, 
        #         #     region_major_start_pca, region_major_end_pca, region_minor_start_pca, region_minor_end_pca, 
        #         #     save_folder=save_folder_region, 
        #         #     save_name=save_name_region.replace('.nii.gz', '_annot.nii.gz')  # '_annotated.nii.gz'
        #         # )
        #         save_annotated_mask(
        #             connected_component_arr, connected_component_img, label, largest_slice_idx, region_arr, region_coords, region_voxel_area, region_centroid_hull,  # region_eigenvectors, 
        #             region_major_axis_length_hull, region_minor_axis_length_hull, region_bbox_min, region_bbox_max, 
        #             region_major_start_hull, region_major_end_hull, region_minor_start_hull, region_minor_end_hull, 
        #             hull_points=region_hull_points,
        #             save_folder=save_folder_region, 
        #             save_name=save_name_region.replace('.nii.gz', '_annot.nii.gz')  # '_annotated.nii.gz'
        #         )
                
        #         # save_visualization_with_overlay(
        #         #     label, mri_slice, largest_slice_idx, region_arr, region_coords, region_voxel_area,
        #         #     region_major_axis_length_pca, region_minor_axis_length_pca, region_bbox_min, region_bbox_max, 
        #         #     region_major_start_pca, region_major_end_pca, region_minor_start_pca, region_minor_end_pca, 
        #         #     save_folder=save_folder_region, 
        #         #     save_name=save_name_region.replace('.nii.gz', '.png'), 
        #         # )
        #         save_visualization_with_overlay(
        #             label, mri_slice, largest_slice_idx, region_arr, region_coords, region_voxel_area,
        #             region_major_axis_length_hull, region_minor_axis_length_hull, region_bbox_min, region_bbox_max, 
        #             region_major_start_hull, region_major_end_hull, region_minor_start_hull, region_minor_end_hull, 
        #             hull_points=region_hull_points,
        #             voxel_spacing=voxel_spacing,
        #             save_folder=save_folder_region, 
        #             save_name=save_name_region.replace('.nii.gz', '.png'), 
        #         )
                
        #     region_measurements = sorted(region_measurements, key=lambda x: (x["region_voxel_area(mm2)"], x["region_voxel_count"], x["major_axis_length(mm)"]), reverse=True)

        #     # Save PCA results to CSV and JSON
        #     region_csv_path = os.path.join(save_folder_region, f"label_{label:03d}_slice_{largest_slice_idx+1:03d}_cca_measurements.csv")
        #     save_info_to_csv(region_measurements, region_csv_path)
        #     # region_json_path = os.path.join(save_folder_region, f"{save_folder_region}_measurements.json")
        #     # save_info_to_json(region_measurements, region_json_path, voxel_spacing=cyst_img.GetSpacing())
            
    
    """7) (validate measured cyst size and shape)"""
    # # Skip validation for very small cysts
    # if bbox_area < 5 or (largest_major_axis_length + largest_minor_axis_length) < 5:
    #     # print(f"WARNING: Skipping validation for Label {label} (bbox_area: {bbox_area:.2f}, sum of axes: {largest_major_axis_length + largest_minor_axis_length:.2f} mm)")
    #     pass
    # else:
    #     # Validate bounding box area is greater than or equal to the cyst area
    #     tolerance = 0.01 * largest_slice_voxel_area  # Allow a small tolerance (1% tolerance)
        
    #     # **Bounding Box Validation (Convex Hull)**
    #     if bbox_area + tolerance < largest_slice_voxel_area:
    #         print(f"WARNING: Bounding box area ({bbox_area:.2f} mm²) is smaller than the cyst area ({largest_slice_voxel_area:.2f} mm²)! "
    #               f"This may indicate an irregular shape.")
            
    #     # **Ellipse Validation (Convex Hull)**
    #     if (largest_major_axis_length / largest_minor_axis_length) < 3:  # Apply only for less elongated shapes
    #         ellipse_area = np.pi * (largest_major_axis_length / 2) * (largest_minor_axis_length / 2)
    #         # print(f"- Ellipse area (mm²): {ellipse_area:.2f}")
    #         if abs(largest_slice_voxel_area - ellipse_area) > largest_slice_voxel_area * 0.2:  # Allow 20% discrepancy
    #             print(f"Note: Large discrepancy between cyst area ({largest_slice_voxel_area:.2f} mm²) and ellipse area ({ellipse_area:.2f} mm²). "
    #                   f"and ellipse area ({ellipse_area:.2f} mm²). This may indicate an irregular shape.")
                
    # Append information
    return {
        "label": label,
        "total_voxel_count": len(voxel_coords),
        "largest_slice_idx": largest_slice_idx+1,  # largest_slice_idx
        "largest_slice_voxel_count": largest_slice_voxel_count,
        "largest_slice_voxel_area(mm2)": largest_slice_voxel_area,
        "major_axis_length(mm)": largest_major_axis_length,  # major_axis_length,
        "minor_axis_length(mm)": largest_minor_axis_length,  # minor_axis_length,
    }, area_ratio


def process_cyst_analysis(fid, cyst_img, cyst_path_rm_duct_dilated, img_path, save_path):

    fname = os.path.basename(cyst_path_rm_duct_dilated)

    # Connected Component Analysis (CCA)
    connected_component_img = sitk.ConnectedComponent(cyst_img > 0)
    connected_component_arr = sitk.GetArrayFromImage(connected_component_img)
    print(f"-- Original mask (z, x, y):        {sitk.GetArrayFromImage(cyst_img).shape}") # (35, 256, 256)
    print(f"-- Connected components (x, y, z): {connected_component_img.GetSize()} {type(connected_component_img)}") # (256, 256, 35) <class 'SimpleITK.SimpleITK.Image'>

    # cca_save_path = os.path.join(save_path, fname.replace(".nii.gz", "_CCA.nii.gz"))
    # save_sitk(connected_component_arr, connected_component_img, output_path=cca_save_path)
    
    label_stats = sitk.LabelShapeStatisticsImageFilter() # automatically excludes label 0 (background)
    label_stats.Execute(connected_component_img)

    voxel_spacing = connected_component_img.GetSpacing() # (spacing_x, spacing_y, spacing_z)
    assert voxel_spacing == cyst_img.GetSpacing()
    print(f'-- voxel_spacing: {voxel_spacing}')
    print()

    # Principal component analysis (PCA)
    cyst_info = []
    label_too_small = []  # track removed labels
    label_unable = []
    area_ratios = []  # [(label, area_ratio), ...]
    # print(f"Total CCA labels: {len(label_stats.GetLabels())}")
    for idx, label in enumerate(label_stats.GetLabels()):
        if label_stats.GetNumberOfPixels(label) < 3:  # count voxels
            # print(f"(Skipping label {label} with {label_stats.GetNumberOfPixels(label)} voxels)")
            label_too_small.append(label) 
            continue

        # print(f'------ Processing label {label} ------')
        # info_by_slice = slice_tracking_with_convex_hull(fid, label, voxel_spacing, connected_component_img, connected_component_arr, img_path, save_path)
        info_by_slice, area_ratio = slice_tracking_with_convex_hull(fid, label, voxel_spacing, connected_component_img, connected_component_arr, img_path, save_path)
        if info_by_slice is None:
            label_unable.append(label)
        else:
            assert info_by_slice["major_axis_length(mm)"] > 0, print(info_by_slice["major_axis_length(mm)"])
            assert info_by_slice["minor_axis_length(mm)"] > 0, print(info_by_slice["minor_axis_length(mm)"])
            cyst_info.append(info_by_slice)
            # area_ratios.append(area_ratio)
            area_ratios.append((label, area_ratio))
        
    # print(f"------ Labels Too Small         : {len(label_too_small)}")
    # print(f"------ Labels Unable to Process : {len(label_unable)}")
    # print(f"------ Final processed labels   : {len(label_stats.GetLabels()) - len(label_too_small) - len(label_unable)}")
    
    # removed_info = {
    #     'total' : len(label_stats.GetLabels()),
    #     'removed' : label_too_small,
    #     'pca' : len(label_stats.GetLabels()) - len(label_too_small)
    # }
    removed_info = {
        'label_stats_total' : len(label_stats.GetLabels()),
        'label_too_small' : label_too_small,
        'label_unable' : label_unable,
        'label_analysis' : len(label_stats.GetLabels()) - len(label_too_small) - len(label_unable)
    }
    cyst_info = sorted(cyst_info, key=lambda x: (x["major_axis_length(mm)"], x["largest_slice_voxel_area(mm2)"], x["total_voxel_count"]), reverse=True)

    """save PCA processed mask"""
    # Exclude the removed labels by setting them to 0
    for label in label_too_small:
        connected_component_arr[connected_component_arr == label] = 0
    for label in label_unable:
        connected_component_arr[connected_component_arr == label] = 0
        
    print("\n=============================================================================================================")
    print(f"\033[1m-- Total CCA labels: {len(label_stats.GetLabels())}\033[0m")
    unique_labels = np.unique(connected_component_arr)
    print(f"\033[1m-- Label Too Small            : {len(label_too_small)} {label_too_small}\033[0m")
    print(f"\033[1m-- Label Unable to Process    : {len(label_unable)} {label_unable}\033[0m")
    print(f"\033[1m-- Label after removal        : {len(unique_labels)} {unique_labels}\033[0m")  # {len(unique_labels[0])}
    print(f"\033[1m-- Label area_ratios          : {len(area_ratios)} {sorted(area_ratios, key=lambda x: x[1])}\033[0m")
    print(f"\033[1m-- Label area_ratios (unique) : {len(set(x[1] for x in area_ratios))} {sorted(set(x[1] for x in area_ratios))}\033[0m")
    
    label_save_path = os.path.join(save_path, fname.replace(".nii.gz", "_labels.nii.gz"))
    save_sitk(connected_component_arr, cyst_img, output_path=label_save_path)

    _saved_img = sitk.ReadImage(label_save_path)
    _saved_arr = sitk.GetArrayFromImage(_saved_img)
    # print(f"\033[1m-- (Check labels of saved mask : {len(np.unique(_saved_arr))} {np.unique(_saved_arr)})\033[0m")
    assert len(unique_labels) == len(np.unique(_saved_arr))
    assert unique_labels.all() == np.unique(_saved_arr).all()
    print("=============================================================================================================")


    # Save PCA results to CSV
    output_csv_path = os.path.join(save_path, fname.replace(".nii.gz", "_measurements.csv"))  # "_PCA_2d.csv"
    save_info_to_csv(cyst_info, output_csv_path)
    
    # Save PCA results to JSON
    output_json_path = os.path.join(save_path, fname.replace(".nii.gz", "_measurements.json"))  # "_PCA_2d.json"
    save_info_to_json(cyst_info, output_json_path, removed_info, cyst_img.GetSpacing())

    # return cyst_info, removed_info, area_ratios
    return label_save_path, output_csv_path, output_json_path, area_ratios


