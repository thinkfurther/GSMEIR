function [x fval] = backwardUpdate(x0,options)
if nargin < 2
    options = optimoptions('fmincon','Algorithm','interior-point');
end

[x fval] = fmincon(@objectiveFunction,x0,[],[],[],[],[],[], ...
    @backwardConstraint,options);

function f = objectiveFunction(DVFBackward)
global idx;
global reconPhantoms;
global noise_projections;
beta = 0.5;
image = imwarp(reconPhantoms(:,:,:,idx),DVFBackward);

[FX,FY,FZ] = gradient(DVFBackward(:,:,:,1));
smoothConstraint = sum(FX.^2,'all') + sum(FY.^2,'all') + sum(FZ.^2,'all');
[FX,FY,FZ] = gradient(DVFBackward(:,:,:,2));
smoothConstraint = smoothConstraint + sum(FX.^2,'all') + sum(FY.^2,'all') + sum(FZ.^2,'all');
[FX,FY,FZ] = gradient(DVFBackward(:,:,:,3));
smoothConstraint = smoothConstraint + sum(FX.^2,'all') + sum(FY.^2,'all') + sum(FZ.^2,'all');

f = norm(noise_projections(:,:,:,1) - Ax(image,geo,angles,'interpolated') )...
         + beta * smoothConstraint;

function [c,ceq] = backwardConstraint(DVFBackward)
global DVFsForward;
global idx;
c = [];
ceq(1) = imwarp(DVFBackward(:,:,:,1),DVFsForward(:,:,:,:,idx));
ceq(2) = imwarp(DVFBackward(:,:,:,2),DVFsForward(:,:,:,:,idx));
ceq(3) = imwarp(DVFBackward(:,:,:,3),DVFsForward(:,:,:,:,idx));