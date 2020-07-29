function [x fval] = forwardUpdate(x0,options)
if nargin < 2
    options = optimoptions('fmincon','Algorithm','interior-point');
end

[x fval] = fmincon(@objectiveFunction,x0,[],[],[],[],[],[], ...
    @forwardConstraint,options);

function f = objectiveFunction(DVFForward)
global idx;
global reconPhantoms;
global noise_projections;
beta = 0.5;
image = imwarp(reconPhantoms(:,:,:,1),DVFForward);

[FX,FY,FZ] = gradient(DVFForward(:,:,:,1));
smoothConstraint = sum(FX.^2,'all') + sum(FY.^2,'all') + sum(FZ.^2,'all');
[FX,FY,FZ] = gradient(DVFForward(:,:,:,2));
smoothConstraint = smoothConstraint + sum(FX.^2,'all') + sum(FY.^2,'all') + sum(FZ.^2,'all');
[FX,FY,FZ] = gradient(DVFForward(:,:,:,3));
smoothConstraint = smoothConstraint + sum(FX.^2,'all') + sum(FY.^2,'all') + sum(FZ.^2,'all');

f = norm(noise_projections(:,:,:,idx) - Ax(image,geo,angles,'interpolated') )...
         + beta * smoothConstraint;

function [c,ceq] = forwardConstraint(DVFForward)
global DVFsBackward;
global idx;
c = [];
ceq(1) = imwarp(DVFForward(:,:,:,1),DVFsBackward(:,:,:,:,idx));
ceq(2) = imwarp(DVFForward(:,:,:,2),DVFsBackward(:,:,:,:,idx));
ceq(3) = imwarp(DVFForward(:,:,:,3),DVFsBackward(:,:,:,:,idx));