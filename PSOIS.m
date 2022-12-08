%% Fatty Liver Level Recognition Using Particle Swarm optimization (PSO) Image Segmentation and Analysis
% https://ieeexplore.ieee.org/document/9960108
% DOI: 10.1109/ICCKE57176.2022.9960108
% Please cite below:
% Mousavi, S. M. H., Victorovich, L. V., Ilanloo, A., & Mirinezhad, S. Y. (2022, November).
% Fatty Liver Level Recognition Using Particle Swarm optimization (PSO) Image Segmentation
% and Analysis. In 2022 12th International Conference on Computer and 
% Knowledge Engineering (ICCKE) (pp. 237-245). IEEE.
%% -------------------------------------------------------------------------------

clear;
clc;
close all;
warning('off');
% Loading
img=imread('fat.jpg');
img=im2double(img);
imgtemp=img;
img = histeq(img);
gray=rgb2gray(img);
gray=imadjust(gray);
% Reshaping image to vector
X=gray(:);

%% Starting PSO Segmentation
k = 2; % Number of segments

%---------------------------------------------------
CostFunction=@(m) ClusterCost(m, X);     % Cost Function
VarSize=[k size(X,2)];           % Decision Variables Matrix Size
nVar=prod(VarSize);              % Number of Decision Variables
VarMin= repmat(min(X),k,1);      % Lower Bound of Variables
VarMax= repmat(max(X),k,1);      % Upper Bound of Variables

%% PSO Parameters

MaxIt = 50;      % Maximum Number of Iterations
nPop = 5;        % Population Size (Swarm Size)
% PSO Parameters
w = 1;            % Inertia Weight
wdamp = 0.99;     % Inertia Weight Damping Ratio
c1 = 1.5;         % Personal Learning Coefficient
c2 = 2.0;         % Global Learning Coefficient
% Velocity Limits
VelMax = 0.1*(VarMax-VarMin);
VelMin = -VelMax;

%% Initialization
empty_particle.Position = [];
empty_particle.Cost = [];
empty_particle.Out = [];
empty_particle.Velocity = [];
empty_particle.Best.Position = [];
empty_particle.Best.Cost = [];
empty_particle.Best.Out = [];
particle = repmat(empty_particle, nPop, 1);
GlobalBest.Cost = inf;
for i = 1:nPop
% Initialize Position
particle(i).Position = unifrnd(VarMin, VarMax, VarSize);
% Initialize Velocity
particle(i).Velocity = zeros(VarSize);
% Evaluation
[particle(i).Cost particle(i).Out] = CostFunction(particle(i).Position);
% Update Personal Best
particle(i).Best.Position = particle(i).Position;
particle(i).Best.Cost = particle(i).Cost;
    particle(i).Best.Out=particle(i).Out;
% Update Global Best
if particle(i).Best.Cost<GlobalBest.Cost
GlobalBest = particle(i).Best;
end
end
BestCost = zeros(MaxIt, 1);

%% PSO Main Loop
for it = 1:MaxIt
for i = 1:nPop
% Update Velocity
particle(i).Velocity = w*particle(i).Velocity ...
+c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
+c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);
% Apply Velocity Limits
particle(i).Velocity = max(particle(i).Velocity, VelMin);
particle(i).Velocity = min(particle(i).Velocity, VelMax);
% Update Position
particle(i).Position = particle(i).Position + particle(i).Velocity;
% Velocity Mirror Effect
IsOutside = (particle(i).Position<VarMin | particle(i).Position>VarMax);
particle(i).Velocity(IsOutside) = -particle(i).Velocity(IsOutside);
% Apply Position Limits
particle(i).Position = max(particle(i).Position, VarMin);
particle(i).Position = min(particle(i).Position, VarMax);
% Evaluation
[particle(i).Cost particle(i).Out] = CostFunction(particle(i).Position);
% Update Personal Best
if particle(i).Cost<particle(i).Best.Cost
particle(i).Best.Position = particle(i).Position;
particle(i).Best.Cost = particle(i).Cost;
            particle(i).Best.Out=particle(i).Out;
% Update Global Best
if particle(i).Best.Cost<GlobalBest.Cost
GlobalBest = particle(i).Best;
end
end
end
BestCost(it) = GlobalBest.Cost;
disp(['In Iteration ' num2str(it) ': PSO Best Cost is = ' num2str(BestCost(it))]);
w = w*wdamp;
BestSol = GlobalBest;
end
PSOlbl=BestSol.Out.ind;

%% Converting cluster (segment) centers and its indexes into image 
gray2=reshape(PSOlbl(:,1),size(gray));
segmented = label2rgb(gray2); 
GTComp=imbinarize(rgb2gray(segmented));
% GTComp = imcomplement (GTComp);
GT=load("GT.mat");
GT=GT.forGT;
% Plot Results 
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,3,1)
imshow(imgtemp,[]);title('Original');
subplot(2,3,2)
imshow(img,[]);title('Pre-Processed');
subplot(2,3,3)
imshow(segmented,[]);title('Segmented Image');
subplot(2,3,4)
imshow(GT,[]);title('Ground-Truth');
subplot(2,3,[5 6])
plot(BestCost,'k','LineWidth',2);
xlabel('Iteration');
ylabel('Best Cost');
ax = gca; 
ax.FontSize = 10; 
ax.FontWeight='bold';
grid on;

%% Compare
% Otsu
thresh = multithresh(gray,k);
seg_I = imquantize(gray,thresh);
Otsu = label2rgb(seg_I); 
% Watershed
bw=imbinarize(gray);
D = bwdist(~bw);D = -D;
L = watershed(D);L(~bw) = 0;
WaterShed = label2rgb(L,'jet',[.1 .1 .1]);
% K-means
[L,Centers] = imsegkmeans(single(gray),k);
KMeans = labeloverlay(gray,L);

% Plot Compare
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
imshow(Otsu); title ('Otsu');
subplot(2,2,2)
imshow(WaterShed); title ('Watershed');
subplot(2,2,3)
imshow(KMeans); title ('K-Means');
subplot(2,2,4)
imshow(segmented,[]); title ('PSO');

%% Statistics
[Accuracy, Sensitivity, Fmeasure, Precision,...
MCC, Dice, Jaccard, Specitivity] = SegPerformanceMetrics(GT, GTComp);
disp(['Accuracy is : ' num2str(Accuracy) ]);
disp(['Precision is : ' num2str(Precision) ]);
disp(['Recall or Sensitivity is : ' num2str(Sensitivity) ]);
disp(['F-Score or Fmeasure is : ' num2str(Fmeasure) ]);
disp(['Dice is : ' num2str(Dice) ]);
disp(['Jaccard is : ' num2str(Jaccard) ]);
disp(['Specitivity is : ' num2str(Specitivity) ]);
