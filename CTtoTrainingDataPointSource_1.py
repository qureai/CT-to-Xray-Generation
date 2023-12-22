import os
import numpy as np
import subprocess
import torch
from scipy.io import loadmat
from scipy.io import savemat
from scipy.ndimage import rotate
from skimage import exposure
from dicomHandler import dicomHandler
from readNPY import readNPY
import imageio 
import re
import SimpleITK as sitk
from scipy.interpolate import griddata
from scipy.interpolate import interpn
from scipy.interpolate import RegularGridInterpolator
import pydicom
import sys 
import cv2
sys.path.append("/home/users/shreshtha.singh/qct_utils")
sys.path.append("/home/users/shreshtha.singh/qct_utils/src")
sys.path.append("/home/users/shreshtha.singh/qct_utils/src/qct_utils")
sys.path.append("/home/users/shreshtha.singh/qct_utils/src/qct_utils/schema")
sys.path.append("/home/users/shreshtha.singh/qct_utils/src/qct_utils/schema/ct/")
import qct_utils
from qct_utils.schema.ct import CTScan

def fillmissing(array):
    # Create a grid of coordinates for the original array
    grid = np.indices(array.shape)
    
    # Mask the missing values
    mask = np.isnan(array)
    
    # Extract the coordinates and values of the non-missing values
    coords_known = [grid[i][~mask] for i in range(array.ndim)]
    values_known = array[~mask]
    
    # Create an interpolator
    interpolator = RegularGridInterpolator(coords_known, values_known, method='linear', bounds_error=False, fill_value=np.nan)
    
    # Generate coordinates for all points
    coords_all = np.stack(np.meshgrid(*[np.arange(size) for size in array.shape], indexing='ij'), axis=-1).reshape(-1, array.ndim)
    
    # Interpolate missing values
    interpolated_values = interpolator(coords_all)
    
    # Reshape the result to the original array shape
    interpolated_values = interpolated_values.reshape(array.shape)
    
    # Replace the missing values with the interpolated values
    array[mask] = interpolated_values[mask]
    
    return array





def CTtoTrainingDataPointSource(CTFolderName,specificationsFileName):
    np.random.seed(42)  # Set a seed for reproducibility
    # write_ct = True
    write_nodules = True

    # if os.path.exists('numpy_nodules'):
    #     os.rmdir('numpy_nodules')
    # # CTnum = int(CTFolderName[8])
    #     # Extract numerical part from CTFolderName
    # match = re.search(r'\d+', CTFolderName)
    # if match:
    #     CTnum = int(match.group())
    # else:
    #     print("Error: Couldn't find a numerical value in CTFolderName.")
    #     sys.exit(1)

        
    # xray_dir = f'chestXRays'
    # if os.path.exists(xray_dir):
    #     os.rmdir(xray_dir)
    # os.makedirs(xray_dir)

    # # read in the CT data
    # CTarrayOriginal, floatVoxelDims = dicomHandler(CTFolderName)
    # CTarrayOriginal = CTarrayOriginal - 1000 
    # CTarrayOriginal = CTarrayOriginal.astype(float) ## added
    # CTarrayOriginal[CTarrayOriginal > 500] = np.nan 
    # Count the number of NaN values
    # nan_count = np.isnan(CTarrayOriginal).sum()
    # print(f"Number of NaN values in the array: {nan_count}")
    # CTarrayOriginal = fillmissing(CTarrayOriginal)
    
    # non_nan_indices = np.where(~np.isnan(CTarrayOriginal))[0]
    # CTarrayOriginal = np.interp(np.arange(len(CTarrayOriginal)), non_nan_indices, CTarrayOriginal[non_nan_indices])
    # print(np.shape(CTarrayOriginal))
    # CTarrayOriginal = np.nan_to_num(CTarrayOriginal, copy=False)
    # print(np.shape(CTarrayOriginal))
    print("hello")
    # CTarrayOriginal = CTarrayOriginal.reshape(CTarrayOriginal.shape[0],-1)
    # print(CTarrayOriginal)
    # dirlist = os.listdir(CTFolderName)
    # info = pydicom.dcmread(os.path.join(CTFolderName, dirlist[0]))
    # floatVoxelDims = [info.SliceThickness, info.PixelSpacing[0], info.PixelSpacing[1]]
    # floatVoxelDims = [1.25,1,1]
    # floatVoxelDims = [1,0.888671875,0.888671875]
    # floatVoxelDims = [1,0.74609375,0.74609375]
    # floatVoxelDims = [3.0,0.771484375,0.771484375]
    # CTarrayOriginal = CTScan.load(CTFolderName, readtype="dcm", scan_type="chestct")
    # 'nhs_brompton.pt': [1, 3, 4, 5, 7, 10, 12],
    #  'carpl.pt': [0, 3, 4]}
    # 'incepto_sample.pt': [2]
    # 'segmed_cancer_samples.pt': [3, 13, 53, 80]
    # ‘mgh: [47, 50, 71, 282, 294, 347, 364, 367, 435, 484, 638, 645, 655, 712, 746, 764, 780, 786, 798, 816, 825, 829, 836, 857]
    # 'iota_cancer_sample.pt': [2, 3, 5, 6, 8, 9, 15, 16, 18, 19, 25, 26, 28, 29, 31, 32, 34, 35, 41, 42, 44, 45, 47, 53, 54, 56, 57, 59, 60, 62, 63, 65, 66]
    # 'medoro_test_samples.pt': [0, 1, 2, 3, 4, 5, 6, 7, 8]
    for CT_Num_ in [1, 3, 4, 5, 7, 10, 12]:
        print("------------------CT_NUM-------------", CT_Num_)
        name = "nhs_brompton.pt"
        CTnum = CT_Num_
        x = torch.load("/cache/fast_data_nas72/qct/data_governance/series_dicts/"+ name)
        sids = list(x.keys())
        CTarrayOriginal = CTScan.load(x[sids[CTnum]], readtype="dcm", scan_type="chestct")
        spacing = CTarrayOriginal.spacing 
        floatVoxelDims = [spacing.z,spacing.x,spacing.y]
        print(CTarrayOriginal.array.shape)
        print(CTarrayOriginal.spacing)
        print(CTarrayOriginal.origin)
        print(floatVoxelDims)
        CTarrayOriginal_= CTarrayOriginal.array
        CTarrayOriginal_[CTarrayOriginal_ > 2000] = np.nan   
        CTarrayOriginal_ = np.nan_to_num(CTarrayOriginal_, copy=False)
        # CTarrayOriginal_ = np.where(CTarrayOriginal_ < -1000,
        #                  np.interp(CTarrayOriginal_, (CTarrayOriginal_.min(), -1000), (-1001, -1000)),
        #                  CTarrayOriginal_)
        # Clip array values below -1000 to -1000
        CTarrayOriginal_ = np.clip(CTarrayOriginal_, -1000, None)
        CTarrayOriginal_ = np.where(CTarrayOriginal_ >= 0,
                    np.interp(CTarrayOriginal_, (0, CTarrayOriginal_.max()), (0, 3000)),
                    CTarrayOriginal_)
        CTarrayOriginal_array = np.reshape(np.transpose(CTarrayOriginal_, (0, 2, 1)), (CTarrayOriginal_.shape[0], -1))
        print("shape: ",CTarrayOriginal_array.shape)
        print("max :",CTarrayOriginal_array.max())
        print("min :",CTarrayOriginal_array.min())
        with open(f'textCTs/CT_{CTnum}.txt', 'w') as f:
            print("yes")
            np.savetxt(f, CTarrayOriginal_array,delimiter=',')   ### fyi - ig i have corrected the code till here - saved the CTarrayOriginal in right format to .txt file for further processing 

            
        scaleFactor = float(floatVoxelDims[0] / floatVoxelDims[2])


    ### nodule insertion part 
        with open(specificationsFileName) as file:
            numPositions = sum(1 for line in file)

        numPositions = 1

        nodulePositions = np.full((numPositions, 3), np.nan)
        noduleDimensions = np.full((numPositions, 3), np.nan)
        noduleHUs = np.full((numPositions, 1), np.nan)
        noduleSizes = np.full((numPositions, 1), np.nan)

        with open(specificationsFileName) as file:
            next(file)  # Skip the header line
            xraynum = 0

            for line in file :
                if xraynum>0 : 
                    break
                line = np.array(line.replace(',', ' ').split(), dtype=float)

                xraynum += 1

                position_x, position_y, position_z = line[:3]
                nodulePositions[xraynum - 1, :] = [position_x, position_y, position_z]

                HU = np.random.randint(80, 151)
                noduleHUs[xraynum - 1] = HU

                size_nodule = np.random.uniform(2, 3)
                noduleSizes[xraynum - 1] = size_nodule

                command_str = f'python random_shape_generator.py -d {floatVoxelDims[0]} -s {size_nodule}'
                subprocess.run(command_str, shell=True)

                nodule = readNPY(f'./numpy_nodules/nodule_{xraynum}.npy')

                nodule = np.resize(nodule, (nodule.shape[0], int(nodule.shape[1] * scaleFactor),
                                            int(nodule.shape[2] * scaleFactor)))

                d1, d2, d3 = nodule.shape
                noduleDimensions[xraynum - 1, :] = [d1, d2, d3]

                if write_nodules:
                    np.savetxt(f'textNodules/nodule_{xraynum}.txt', nodule.flatten(), fmt='%d', delimiter=',')

        with open(f'nodule_specs_{CTnum}.txt', 'w') as specs_file:
            specs_file.write('xraynumber,positions(3),size(3),size(cm),HU\n')
            specs_file.write('0,Nan,Nan,Nan,Nan,Nan,Nan,Nan,Nan\n')
            for xraynum, (pos, dim, size, HU) in enumerate(zip(nodulePositions, noduleDimensions, noduleSizes, noduleHUs), 1):
                specs_file.write(f'{xraynum},{pos[0]},{pos[1]},{pos[2]},{dim[0]},{dim[1]},{dim[2]},{size[0]:.2f},{HU[0]}\n')
        
        subprocess.call("make", shell=True)
        # if make_result.returncode != 0:
        #     print("Error in 'make' command")
        #     print("Make output:", make_result.stdout.decode())
        #     print("Make error:", make_result.stderr.decode())
            # Handle the error or exit the script as needed

        cmd = [
            './lungnodulesynthesizer',
            f'textCTs/CT_{CTnum}.txt',
            f'nodule_specs_{CTnum}.txt',
            str(floatVoxelDims[2]),  
            str(floatVoxelDims[0]),
        ]    
        subprocess.call(cmd)
        print("yes")

        files = [f for f in os.listdir('textXRays') if os.path.isfile(os.path.join('textXRays', f))]
        for k, filename in enumerate(files, 1):
            xraynum = k - 1
            img = np.genfromtxt(os.path.join('textXRays', filename), delimiter=',', filling_values=np.nan)
            img = img[:, :-1]
            # print(filename)
            # with open(os.path.join('textXRays', filename), 'r') as file:   
                # img = np.array([float(value) if value != '' else 0.0 for value in file.read().split(',')])
                # img = np.loadtxt([float(value) if value.strip() != '' else 0.0 for value in file.read().split(',')])
            if xraynum:
                imagename = f'chestXRays/Xray{CTnum}'
            else:
                imagename = f'chestXRays/Xray{CTnum}'

            img = 1 - img
            print(img.shape)
            img = rotate(img, 90)
            im_adjusted = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
            img = exposure.equalize_adapthist(im_adjusted)

            img *= 255  # Assuming 8-bit images
            img = img.astype(np.uint8)
            imageio.imwrite(f'{imagename}.png', img)

if __name__ == "__main__":
    # Specify the CTFolderName and specificationsFileName as command-line arguments
    import sys
    if len(sys.argv) != 3:
        print("Usage: python your_script_name.py CTFolderName ")
        sys.exit(1)

    CTFolderName = sys.argv[1]
    specificationsFileName = sys.argv[2]

    CTtoTrainingDataPointSource(CTFolderName,specificationsFileName)
