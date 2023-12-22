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

def CTtoTrainingDataPointSource(CTFileName,specificationsFileName):
    np.random.seed(42)  # Set a seed for reproducibility
    write_nodules = True
    for CT_Num_ in [1,3]:
        print("------------------CT_NUM-------------", CT_Num_)
        CTnum = CT_Num_
        x = torch.load(CTFileName)
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
        CTarrayOriginal_ = np.clip(CTarrayOriginal_, -1000, None)
        CTarrayOriginal_ = np.where(CTarrayOriginal_ >= 0,
                    np.interp(CTarrayOriginal_, (0, CTarrayOriginal_.max()), (0, 3000)),
                    CTarrayOriginal_)
        CTarrayOriginal_array = np.reshape(np.transpose(CTarrayOriginal_, (0, 2, 1)), (CTarrayOriginal_.shape[0], -1))
        with open(f'textCTs/CT_{CTnum}.txt', 'w') as f:
            np.savetxt(f, CTarrayOriginal_array,delimiter=',')    

            
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
        
        ### Make file for 
        subprocess.call("make", shell=True)
        cmd = [
            './lungnodulesynthesizer',
            f'textCTs/CT_{CTnum}.txt',
            f'nodule_specs_{CTnum}.txt',
            str(floatVoxelDims[2]),  
            str(floatVoxelDims[0]),
        ]    
        subprocess.call(cmd)


    files = [f for f in os.listdir('textXRays') if os.path.isfile(os.path.join('textXRays', f))]
    for k, filename in enumerate(files, 1):
        num = filename.split('_')[-1][:-4]
        img = np.genfromtxt(os.path.join('textXRays', filename), delimiter=',', filling_values=np.nan)
        img = img[:, :-1]
        imagename = f'chestXRays/Xray_{num}'

        img = 1 - img
        img = rotate(img, 90)
        im_adjusted = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        img = exposure.equalize_adapthist(im_adjusted)

        img *= 255  # Assuming 8-bit images
        img = img.astype(np.uint8)
        imageio.imwrite(f'{imagename}.png', img)

if __name__ == "__main__":
    # Specify the CTFileName and specificationsFileName as command-line arguments
    import sys
    if len(sys.argv) != 3:
        print("Usage: python your_script_name.py CTFolderName ")
        sys.exit(1)

    CTFilerName = sys.argv[1]
    specificationsFileName = sys.argv[2]

    CTtoTrainingDataPointSource(CTFilerName,specificationsFileName)
