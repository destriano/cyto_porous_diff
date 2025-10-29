# cyto_porous_diff
Codes used for generation of the experiments, simulation, and model-experiment comparison results of the manuscript "Cytoplasmic crowding acts as a porous medium reducing macromolecule diffusion".

Biorxiv preprint available at 
www.biorxiv.org/content/10.1101/2025.07.28.667145v1

Test data for each code are available at 
https://drive.google.com/drive/folders/1_dm-Qdy5PZaqRbkB8h7PlyJ6r3B3WpCI?usp=sharing

Full data (>1To) can be provided upon request


Code description
- CLEAN_FRAP_extraction_and_analysis_2025_10_21: to process raw confocal microscopy (Zeiss .czi format) timelapse images to infer GFP diffusivity.
- CLEAN_FCS_extraction_and_analysis_2025_10_21: to process autocorrelation curves obtained from raw FCS data by another software (we used Symphotime64), to infer GFP inverse residence time in effective FCS volume.
- CLEAN_SEGM_extraction_and_analysis_2025_10_21: to process raw confocal microscopy (Zeiss .czi format) zstack images to infer cell volume and obstacle fractions in user defined cytoplasmic regions. Require an intermediate manual processing using a basic image analysis software such as paint.
- CLEAN_FIT_all_data_sets_OSMOTIC_2025_10_22: xxxx
- eee
- eee
- eee

  The codes XX, YY, and ZZ were run on WSL2 (Ubuntu 20.04 LTS GNU/Linux 6.6.87.2-microsoft-standard-WSL2 x86_64) on a conda environment containing not exhaustively gmsh 4.9.5, fenics 2019.1.0 and meshio 5.3.4.


All computations were run on a Dell laptop XPS15 9560 with Intel core i7 7th gen and 16Go RAM.
