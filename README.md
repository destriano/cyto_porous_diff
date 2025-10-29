# cyto_porous_diff
Codes used for generation of the experiments, simulation, and model-experiment comparison results of the manuscript "Cytoplasmic crowding acts as a porous medium reducing macromolecule diffusion".

Biorxiv preprint available at 
www.biorxiv.org/content/10.1101/2025.07.28.667145v1

Test data for each experiment processing and full fitting data are available at 
https://drive.google.com/drive/folders/1_dm-Qdy5PZaqRbkB8h7PlyJ6r3B3WpCI?usp=sharing

Full experimental data (>1To) can be provided upon reasonable request


Code description
- FRAP_extraction_and_analysis_2025_10_21: to process raw confocal microscopy (Zeiss .czi format) timelapse images to infer GFP diffusivity. (test data provided in FRAP folder of the gdrive)
- FCS_extraction_and_analysis_2025_10_21: to process autocorrelation curves obtained from raw FCS data by another software (we used Symphotime64), to infer GFP inverse residence time in effective FCS volume. (test data provided in FCS folder of the gdrive)
- SEGM_extraction_and_analysis_2025_10_21: to process raw confocal microscopy (Zeiss .czi format) zstack images to infer cell volume and obstacle fractions in user defined cytoplasmic regions. Require an intermediate manual processing using a basic image analysis software such as paint. (test data provided in SEGM folder of the gdrive)
- FIT_all_data_sets_OSMOTIC_2025_10_22: to compare experimental and multiscale model results. Allows for reproduction of figures 5 and 6 of the main manuscript (full data provided in FIT folder of the gdrive)
- P25_plot_theroretical_multiscale_model_curves_2025_10_29 : allows to plot diverse multiscale model predictions, including reproduction of fig4 predictions. (full data provided in the THEORETICAL_PLOT folder of the gdrive)
- P25_all_fenics_computations_2D_3D_T_H_2025_10_29: allows to resolve numerically using finite elements method the closure problems for tortuosity and permeability. Require the functions present in P25_all_fenics_functions_2D_3D_T_H


The code "P25_all_fenics_computations_2D_3D_T_H_2025_10_29" was run on WSL2 (Ubuntu 20.04 LTS GNU/Linux 6.6.87.2-microsoft-standard-WSL2 x86_64) on a conda environment containing not exhaustively gmsh 4.9.5, fenics 2019.1.0 and meshio 5.3.4.

All computations were run on a Dell laptop XPS15 9560 with Intel core i7 7th gen and 16Go RAM.
