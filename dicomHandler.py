import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
import cv2

### just for my note this is python file is good to go, Slice location was not present so i did the sorting based on Instance number.
### not sure if that is the correct thing to do. ask ashish/ also check why do we actually need to order the slices 
 
def dicomHandler(folderName):
    dirlist = os.listdir(folderName)
    maxSN = len(dirlist)
    info = pydicom.dcmread(os.path.join(folderName, dirlist[10]))
    empty = -9999
    N_XY = 512
    dArr = np.zeros((maxSN, N_XY, N_XY), dtype=np.uint16) + empty   ## added 


    slicesPos = np.zeros(maxSN) + empty  #(10,)

    place = 0
    for i in range(maxSN):
        x = dirlist[i]

        if x[0] == '1':
            info = pydicom.dcmread(os.path.join(folderName, x))
            print(info)
            slicesPos[place] = info[(0x0020, 0x0013)].value ## patient id 
            place += 1
    print(slicesPos)
    slicesPos = np.sort(np.unique(slicesPos))
    print(np.shape(slicesPos))

    for i in range(maxSN):
        x = dirlist[i]
        print("x:",x)
        if x[0] == '1':
            try : 
                info = pydicom.dcmread(os.path.join(folderName, x))
                # resized_array = cv2.resize(info.pixel_array, (512, 512), interpolation=cv2.INTER_LINEAR)   ## added 
                dArr[slicesPos == info[(0x0020, 0x0013)].value, :, :] = np.uint16(info.pixel_array)
            except Exception as e : 
                print(e)
                pass 
    print(np.shape(dArr))
    print("dArr:" ,dArr)
    dArr = dArr[~np.any(dArr == empty, axis=(1, 2)), :, :]
    print(np.shape(dArr))
    myIm = np.squeeze(dArr[0, :, :]) - 1000

    # Check the data range of myIm
    print("Data Range:", myIm.min(), myIm.max())

    # Display the image
    # plt.figure()
    plt.imshow(myIm, cmap='gray', vmin=myIm.min(), vmax=myIm.max())  # Adjust vmin and vmax if needed
    plt.savefig('my_image.png')
    plt.show()

    voxelDims = [info.SliceThickness, info.PixelSpacing[0], info.PixelSpacing[1]]

    return dArr, voxelDims

# Example usage:
# CTarray, voxelDims = dicomHandler('your_folder_path')
