% Distances
geo.DSD = 1500;                             % Distance Source Detector      (mm)
geo.DSO = 1000;                             % Distance Source Origin        (mm)
% Detector parameters
geo.nDetector=[384; 150];					% number of pixels              (px)
geo.dDetector=[0.2; 0.2]; 					% size of each pixel            (mm)
geo.sDetector=geo.nDetector.*geo.dDetector; % total size of the detector    (mm)
% Image parameters
geo.nVoxel=[256;256;100];                   % number of voxels              (vx)
geo.dVoxel=[0.2; 0.2; 0.2];                 % size of each voxel            (mm)
geo.sVoxel=geo.nVoxel.*geo.dVoxel;          % total size of the image       (mm)
% Offsets
geo.offOrigin =[0;0;0];                     % Offset of image from origin   (mm)              
geo.offDetector=[0; 0];                     % Offset of Detector            (mm)
                                            % These two can be also defined
                                            % per angle
% Auxiliary 
geo.accuracy=0.5;                           % Variable to define accuracy of
                                            % 'interpolated' projection
                                            % It defines the amoutn of
                                            % samples per voxel.
                                            % Recommended <=0.5             (vx/sample)

% Optional Parameters
geo.mode='cone';                            % Or 'parallel'. Geometry type.  

                                            
% plotgeometry(geo,-pi/6);
phaseNumber = 2;
phantoms = zeros(geo.nVoxel(1),geo.nVoxel(2),geo.nVoxel(3), phaseNumber,'single');
for phaseIdx = 1 : phaseNumber
    fid = fopen('Phantom_atn_1.bin','rb');
    phantomPhase = single(fread(fid,'float'));
    fclose(fid);
    phantoms(:,:,:,phaseIdx) = reshape(phantomPhase,geo.nVoxel(1),geo.nVoxel(2),geo.nVoxel(3));
end

% plotImg(phases(:,:,:,1),'Dim','Z');

% define projection angles (in radians)
angles=linspace(0,2*pi,30);

% Simulate forward projection.
% Strongly suggested to use 'iterpolated' option for more accurate
% projections. reduce geo.accuracy for better results
projections = zeros(geo.nDetector(2),geo.nDetector(1),length(angles), phaseNumber);
noise_projections = zeros(geo.nDetector(2),geo.nDetector(1),length(angles), phaseNumber);
for phaseIdx = 1 : phaseNumber
    projections(:,:,:,phaseIdx) = Ax(phantoms(:,:,:,phaseIdx),geo,angles,'interpolated');
    noise_projections(:,:,:,phaseIdx) = addCTnoise(projections(:,:,:,phaseIdx),'Poisson',1e5,'Gaussian',[0 10]);
end

% % Plot Projections
% plotProj(projections(:,:,:,1),angles)
% % plot noise
% plotProj(projections(:,:,:,1)-noise_projections(:,:,:,1),angles)

% obj = Atb((projections(:,:,:,1)),geo,angles,'matched');

qualmeas={'RMSE','MSSIM'};
reconPhantoms = zeros(size(phantoms));
for phaseIdx = 1 : phaseNumber
[imgSART,errL2SART,qualitySART]=SART_TV(noise_projections(:,:,:,phaseIdx),geo,angles,30,...
                            'QualMeas',qualmeas);
reconPhantoms(:,:,:,phaseIdx) = imgSART;
end

DVFsForward = zeros([geo.nVoxel(1),geo.nVoxel(2),geo.nVoxel(3) , 3 , phaseNumber]);
DVFsBackward = zeros([geo.nVoxel(1),geo.nVoxel(2),geo.nVoxel(3) , 3 , phaseNumber]);
for phaseIdx = 1 : phaseNumber
    [D1,~] = imregdemons(gpuArray(reconPhantoms(:,:,:,1)),gpuArray(reconPhantoms(:,:,:,phaseIdx)),[500 400 200],...
        'AccumulatedFieldSmoothing',1.3);
    [D2,~] = imregdemons(gpuArray(reconPhantoms(:,:,:,phaseIdx)),gpuArray(reconPhantoms(:,:,:,1)),[500 400 200],...
        'AccumulatedFieldSmoothing',1.3);
    DVFsForward(:,:,:,:,phaseIdx) = gather(D1);
    DVFsBackward(:,:,:,:,phaseIdx) = gather(D2);
end
clear D1 D2;

% plotImg(imgSART,'Dim','Z');
% recon from all the phases
ART_TV_iteration = 10;
TViter = 20;
TVlambda = 0.7;
for i = 1 : ART_TV_iteration
	for phaseIdx =  1 : phaseNumber
		phaseImage = imwarp(reconPhantoms(:,:,:,1),DVFsForward(:,:,:,:,phaseIdx));
        [phaseImage,errL2SART,qualitySART]=SART(noise_projections(:,:,:,phaseIdx),geo,angles,30,...
                            'QualMeas',qualmeas,'Init','image','InitImg',phaseImage);
		DVF = -DVFsForward(:,:,:,:,phaseIdx);
		phase1 = imwarp(phaseImage,DVF);
		reconPhantoms(:,:,:,1) = im3DDenoise(phase1,'TV',TViter,TVlambda);
	end 
end 

% Update DVF in projection domain
updateNumbers = 10;
for phaseIdx = 2 : phaseNumber
    for idx = 1 : updateNumbers
        forwardDVF = DVFsForward(:,:,:,:,phaseIdx);
        [forwardDVF fvalForward] = forwardUpdate(forwardDVF);
        DVFsForward(:,:,:,:,phaseIdx) = forwardDVF;
        backwardDVF = DVFsBackward(:,:,:,:,phaseIdx);
        [backwardDVF fvalBackward] = forwardUpdate(backwardDVF);
        DVFsBackward(:,:,:,:,phaseIdx) = backwardDVF;
    end
end