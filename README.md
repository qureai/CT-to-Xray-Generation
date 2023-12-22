# Synthetic X-Ray Project
Based on papers 

A. Moturu and A. Chang, “Creation of synthetic x-rays to train a neural network to detect lung cancer.” http://www.cs.toronto.edu/pub/reports/na/Project_Report_Moturu_Chang_1.pdf, 2018.

A. Chang and A. Moturu, "Detecting Early Stage Lung Cancer using a Neural Network Trained with Patches from Synthetically Generated X-Rays." http://www.cs.toronto.edu/pub/reports/na/Project_Report_Moturu_Chang_2.pdf, 2019. 

Creation of Synthetic X-Rays to Train a Neural Network to Detect Lung Cancer

Directory:  
>   chestCT0/  
	Chunk.hpp  
	Constants.h  
	Coordinate.cpp  
	Coordinate.hpp  
	CTtoTrainingDataParallel.m  
	CTtoTrainingDataPointSource.m  
	dicomHandler.m  
	Lung Segmentation/
	main.cpp  
	Makefile  
	methods.cpp  
	methods.hpp  
	NoduleSpecs.hpp  
	Pixel.hpp  
	positions_0.txt  
	ProjectionPlane.hpp  
	random_shape_generator.py  
	readNPY.m  
	readNPYheader.m  
	SimulatedRay.cpp  
	SimulatedRay.hpp  
	textCTs/  
	textNodules/  
	textXRays/  
	Voxel.cpp  
	Voxel.hpp  

The Lung Segmentation folder contains the segment_lungs.py file which segments lungs for randomized nodule placement,  
but positions_0.txt contains manually selected points as of now.  

Running this program will create X-rays that are placed in the chestXRays0 folder.  
The textCTs, textNodules, and textXRays folders are used in the process of making point-source X-rays.  
The code is not adding synthetic nodules. Can be tweaked to do the same. 
 
Example run:  python CTtoTrainingDataPointSource.py  nhs_brompton.pt  positions_0.txt

**CTtoTrainingDataPointSource.py** is the main program that makes point-source X-rays (can have 7 different point sources ).  
Dependencies:  
>	random_shape_generator.py  
	--  
	readNPY.m  
	readNPYheader.m  
	dicomHandler.m  
	--  
	positions_0.txt (etc.)  
	chestCT0/ (etc.)  
	--  
	Constants.h  
	Chunk.hpp  
	Coordinate.hpp  
	methods.hpp  
	SimulatedRay.hpp  
	ProjectionPlane.hpp  
	Voxel.hpp  
	NoduleSpecs.hpp  
	Pixel.hpp  
	--  
	main.cpp  
	methods.cpp  
	Coordinate.cpp  
	SimulatedRay.cpp  
	Voxel.cpp  
	--  
	Makefile

To run in python, type into the console:  python CTtoTrainingDataPointSource.py CTFileName specificationsFileName
>	CTFileName is the .pt file containing CTs 
	specificationsFileName is the file that contains nodule positions(positions_0.txt provided)  

Example run: python CTtoTrainingDataPointSource.py  /cache/fast_data_nas72/qct/data_governance/series_dicts/nhs_brompton.pt  positions_0.txt 
