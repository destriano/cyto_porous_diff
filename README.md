# cyto_porous_diff
Codes used for generation of the experiments, simulation, and model-experiment comparison results of the manuscript "Cytoplasmic crowding acts as a porous medium reducing macromolecule diffusion".

Biorxiv preprint available at 
www.biorxiv.org/content/10.1101/2025.07.28.667145v1

Test data for each experiment processing and full fitting data are available at 
https://drive.google.com/drive/folders/1_dm-Qdy5PZaqRbkB8h7PlyJ6r3B3WpCI?usp=sharing

Full experimental data (>1To) can be provided upon reasonable request



**Code description**

**FRAP_extraction_and_analysis_2025_10_21:** to process raw confocal microscopy (Zeiss .czi format) timelapse images to infer GFP diffusivity. (test data provided in FRAP folder of the gdrive)
How to test the code: Simply run the routine in a folder containing the metadata .xls file and example .czi files. The code returns a list containing infered diffusivity from each .czi file ('L_d_gfp') and lists with associated metadata for each diffusivity value (examples: cell region studied 'L_zone', cell number 'L_cell', experiment type "L_effet").
Packages used in conda environment: spyder, python3, aicspylibczi, numpy, pandas, scipy



**FCS_extraction_and_analysis_2025_10_21:** to process autocorrelation curves obtained from raw FCS data by another software (we used Symphotime64), to infer GFP inverse residence time in effective FCS volume. (test data provided in FCS folder of the gdrive).
How to test the code: Simply run the routine in a folder containing the metadata .xls file. Here, the metadata file contains the non-averaged autocorrelation curves obtained from Symphotime64 software. The code returns a list containing infered inverse residence times from the average autocorrelation curve obtained from each cytoplasmic region in each cell ('L_tau') and lists with associated metadata for each inverse residence time value (examples: cell region studied 'L_zone', cell number 'L_cellule', experiment type "L_effet").
Packages used in conda environment: spyder, python3, pandas, scipy, numpy, matplotlib, seaborn



**SEGM_extraction_and_analysis_2025_10_21:** to process raw confocal microscopy (Zeiss .czi format) zstack images to infer cell volume and obstacle fractions in user defined cytoplasmic regions. Require an intermediate manual processing using a basic image analysis software such as paint. (test data including intermediate manually processed files provided in SEGM folder of the gdrive)
How to test the code: First, run the routine in a folder containing the metadata .xls, a subfolder 'data/' with raw czi zstack data, and two empty subfolders "files_to_be_manually_marked/" and "files_manually_marked/". The routine will run until it stops at line 798. Then, the "marked" and 'light' files generated in  "files_to_be_manually_marked/" have to be processed using a basic bitmap image processing tool such as paint. These two files are 2D binned and processed versions of the 3D zstack data. 

In the marked file, the nucleoplasmic region (which serves as a fluorescence reference), the LP region, and the HP region have to be tagged with respectively with the paint standard red color (rgb [237, 28, 36]), paint standard blue (rgb [63, 72, 204]), and paint standard green (rgb [34, 177, 76]). In our study, a single 2D slice was used for tagging each of these three regions, but tagging several slices for each region should in principle possible. To help recover the regions used during experiment for FRAP measurements, coordinates indicated in the metadata file are used to automatically print blue and green squares in the marked file at approximate HP and LP positions. The colors used to print these squares are not standard blue nor standard green, and thus are not automatically counted for the rest of the processing. 

The light file has to be opened for removal of neighboring cells with tools such as paint "free selection". Pixels corresponding to neighboring cells must be replaced with white pixels. Then, the light file, the marked file, and the heavy file must be copied in the folder "files_manually_marked/".

Then, the python routine second block (starting line 799) can be run, and use the three abovementioned files to compute the cell volume and obstacle fractions in marked regions. It returns a char array containing the region name, the computed cell volume, and accessible and excluded volume frations. 

Packages used in conda environment: spyder, python3, aicspylibczi, pandas, scipy, numpy, matplotlib, skimage



**FIT_all_data_sets_OSMOTIC_2025_10_22:** to compare experimental and multiscale model results. Allows for reproduction of figures 5 and 6 of the main manuscript (full data provided in FIT folder of the gdrive)
How to test the code: Simply run the code in a folder containing the two .npy files and the two subfolders present in the gdrive "Cyto_porous_diff_data/FIT" shared folder. The routine will run analyses and produce a great number of plots, many of them being displayed in the manuscript, though with slightly different formating and legends.

The results from FRAP analysis are contained in the char array "charray_frap[...].npy" and those from SEGM analysis in the char array "charray_segm[...].npy". 

Packages used in conda environment: spyder, python3, pandas, scipy, numpy, matplotlib, seaborn



**P25_plot_theroretical_multiscale_model_curves_2025_10_29:** allows to plot diverse multiscale model predictions, including reproduction of fig4 predictions. (full data provided in the THEORETICAL_PLOT folder of the gdrive)
How to test the code: Simply run the routine in a folder containing file 'results_3D_aleatoire[...].xls' and subfolder 'porous_hindrances_updated' from the shared gdrive. Numerous plots should open, including several plots from manuscript, with a slightly different formating.

Packages used in conda environment: spyder, python3, pandas, numpy, matplotlib



**P25_all_fenics_computations_2D_3D_T_H_2025_10_29:** allows to resolve numerically using finite elements method the closure problems for tortuosity and permeability. 
How to test the code: In a WSL2 (Windows Subsystem for Linux 2, Ubuntu 20.04 LTS GNU/Linux 6.6.87.2-microsoft-standard-WSL2 x86_64) console, simply run the code in a folder containing the file P25_all_fenics_functions_2D_3D_T_H. The different blocks in the routine can be used for testing different geometries. It shows the meshed geometry, and upon closure of the gmsh popup window (which pauses the routine) it prints the results: obstacles volume fractions, tortuosity, permeability.

Packages used in a WSL2 conda environment: spyder, python3, numpy, matplotlib, dolfin, gmsh 4.9.5, fenics 2019.1.0 and meshio 5.3.4





All computations were run on a Dell laptop XPS15 9560 with Intel core i7 7th gen and 16Go RAM.
