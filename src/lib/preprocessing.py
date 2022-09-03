from glob import glob
from nilearn.image import resample_to_img, resample_img
from nilearn import datasets
import nibabel as nib
import numpy as np
import os.path as op
import os

def get_imlist(images):
    '''
    Search for the list of images in the repository "images" that are in NiFti file format.
    
    Parameters:
        - images: str, path to the directory containing images or list, list of the paths to images
        
    Return:
        - files: list, list containing all paths to the images
        - inpdir: boolean, True if images is the directory containing the images, False otherwise
    '''
    if op.isdir(images):
        files = sorted(glob(op.join(images, "*.nii*"), recursive=True))
        inpdir = True
    else:
        files = [images]
        inpdir = False
    return files, inpdir

def preprocessing(data_dir, output_dir):
    '''
    Preprocess all maps that are stored in the 'original' repository of the data_dir. 
    Store these maps in subdirectories of the data_dir corresponding to the preprocessing step applied.


    Parameters:
        - data_dir, str: path to directory where 'original' directory containing all original images is stored

    '''
    # Get image list to preprocess
    img_list, input_dir = get_imlist(op.join(data_dir, 'original'))
        
    # Create dirs to save images
    if not op.isdir(op.join(output_dir, 'resampled')):
        os.mkdir(op.join(output_dir, 'resampled'))
    if not op.isdir(op.join(output_dir, 'resampled_masked')):
        os.mkdir(op.join(output_dir, 'resampled_masked'))
    if not op.isdir(op.join(output_dir, 'resampled_masked_normalized')):
        os.mkdir(op.join(output_dir,  'resampled_masked_normalized'))
    if not op.isdir(op.join(output_dir, 'resampled_normalized')):
        os.mkdir(op.join(output_dir, 'resampled_normalized'))


    # Load mask to apply to images    
    mask = datasets.load_mni152_brain_mask(resolution=4, threshold=0.1)
    mask_affine = mask.affine
    mask_data= mask.get_fdata()[:48, :56, :48]
    res_mask = nib.Nifti1Image(mask_data, mask_affine)
    
    for idx, img in enumerate(img_list):
        print('Image', img)

        nib_img = nib.load(img)
        img_data = nib_img.get_fdata()
        img_data = np.nan_to_num(img_data)
        img_affine = nib_img.affine
        nib_img = nib.Nifti1Image(img_data, img_affine)
        
        print('Original shape of image ', idx+1, ':',  nib_img.shape)

        try:
            if op.isfile(op.join(output_dir, 'resampled', op.basename(img))) and \
            op.isfile(op.join(output_dir, 'resampled_masked', op.basename(img))) and \
            op.isfile(op.join(output_dir,'resampled_masked_normalized', op.basename(img))) and \
            op.isfile(op.join(output_dir, 'resampled_normalized', op.basename(img))):
                continue

            print("Resampling image {0} of {1}...".format(idx + 1, len(img_list)))
            
            res_img = resample_to_img(nib_img, datasets.load_mni152_template(resolution = 4), 
                                      interpolation='nearest', clip = True)
            res_img_affine = res_img.affine
            res_img_data= res_img.get_fdata()[:48, :56, :48]
            res_img = nib.Nifti1Image(res_img_data, res_img_affine)

            print('New shape for image', idx, res_img.shape)

            nib.save(res_img, op.join(output_dir, 'resampled', op.basename(img))) # Save original image only resampled
            
            print("Masking image {0} of {1}...".format(idx + 1, len(img_list)))
            
            res_mask_data = res_mask.get_fdata()
            res_img_data = res_img.get_fdata()
            
            res_masked_img_data = res_img_data * res_mask_data
            
            res_masked_img = nib.Nifti1Image(res_masked_img_data, res_img_affine)
            
            nib.save(res_masked_img, op.join(output_dir,'resampled_masked', op.basename(img))) # Save original image resampled and masked

            print('Min-Max normalizing image', idx)

            res_norm_img_data = res_img_data.copy().astype(float)
            res_norm_img_data = np.nan_to_num(res_norm_img_data)
            res_norm_img_data *= 1.0/np.abs(res_norm_img_data).max()

            res_norm_img = nib.Nifti1Image(res_norm_img_data, res_img_affine)
            
            nib.save(res_norm_img, op.join(output_dir, 'resampled_normalized', op.basename(img))) # Save original image resampled and normalized

            print('Min-Max normalizing masked image', idx)

            res_masked_norm_img_data = res_masked_img_data.copy().astype(float)
            res_masked_norm_img_data = np.nan_to_num(res_masked_norm_img_data)
            res_masked_norm_img_data *= 1.0/np.abs(res_masked_norm_img_data).max()

            res_masked_norm_img = nib.Nifti1Image(res_masked_norm_img_data, res_img_affine)
            
            nib.save(res_masked_norm_img, op.join(output_dir, 'resampled_masked_normalized', op.basename(img))) # Save original image resampled, masked and normalized

            print(f"Image {idx} : DONE.")

        except Exception as e:
            print("Failed!")
            print(e)
            continue