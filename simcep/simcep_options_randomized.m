%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for population level simulation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the size of the simulated image and part of the image where cells
% will be simulated. For example, ones(500), results as a 500 x 500 image
% where cells can be simulated in every part of the image.
population.template = ones(224,224);

% Amount of cells simulated in the image
population.N = randi(15);

% Amount of clusters
population.clust = randi(5);

% Probability for assigning simulated cell into a cluster. Otherwise the
% cell is uniformly distributed on the image.
population.clustprob = rand;

% Variance for clustered cells
population.spatvar = 0.05;

% Amount of allowed overlap for cells [0,1]. For example, 0 = no overlap
% allowed and 1 = overlap allowed.
population.overlap = rand*0.6; % proportion overlap allowed, e.g. 0.5 = 50%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters for the measurement system
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Energy of illumination compared to the energy of cells
measurement.illumscale = max(0, 0.4*randn+1);

% Misalignment of illumination source in x and y direction
measurement.misalign_x = randn*0.5;
measurement.misalign_y = randn*0.5;

% Energy of autofluorescence compared to the energy of cells
measurement.autofluorscale = max(0, randn*0.05+0.05);

% Variance of noise for ccd detector
measurement.ccd = max(0, 0.001*10^(randn*0.4));

% Amount of compression artefacts
measurement.comp = 0; % max(0, randn*0.6); % no-one uses compressed data.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cell level parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cytoplasm radius
cell_obj.cytoplasm.radius = 25+randn*5;
% Parameters for random shape
cell_obj.cytoplasm.shape = [0.3+randn*0.1 0.05+randn*0.02];
% Parameters for texture: persistence, 1st octave, last octave, and
% intensity bias
cell_obj.cytoplasm.texture = [0.9 2 8 0.2];

%%% Nuclei (see cytoplasm parameters for details)
cell_obj.nucleus.radius = max(5, 10+randn*3);
cell_obj.nucleus.shape = [0.1+randn*0.05 0.1+randn*0.05];
cell_obj.nucleus.texture = [0.5 2 5 0.2];

% Decide which parts to include:
cell_obj.nucleus.include = 1;
if rand < 0.5 && 0
    cell_obj.cytoplasm.include = 1;
else
    cell_obj.cytoplasm.include = 0;
end

if rand < 0.5 && cell_obj.cytoplasm.include == 1
    %%% Subcellular parts (modeled as objects inside the cytoplasm; note cytoplasm
    %%% simulation needed for simulation of subcellular parts).
    cell_obj.subcell.include = 1;
    % Number of subcellular objects
    cell_obj.subcell.ns = randi(8);
    % Radius of single object
    cell_obj.subcell.radius = (1+randn*0.2)*3*cell_obj.cytoplasm.radius/25;
    cell_obj.subcell.shape = [0.1+randn*0.1 0.1+randn*0.1];
    cell_obj.subcell.texture = [0.5 2 5 0.2];
else
    cell_obj.subcell.include = 0;
end

% Is the overlap measured on nuclei (=1) level, or cytoplasm (=2) level
if cell_obj.cytoplasm.include == 0
    population.overlap_obj = 1; %Overlap: nuclei = 1, cytoplasm = 2
else
    population.overlap_obj = 2;
end
