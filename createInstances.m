rng(0,'twister');
yalmip('clear');
close all;
clear;


%----------------------------------------------------------------------
% Increase TimeLength: More Integer, Binary and Linear Constraints
% Number of  
% 
TimeLength = 30; 

% Increase numTrainingTraj: 
numTrainingTraj = 200; 
%----------------------------------------------------------------------

%----------------------------------------------------------------------
% CPP parameters. 
delta = 0.1;
quantileIndex = ceil((numTrainingTraj+1)*(1-delta));
if(quantileIndex>numTrainingTraj)
    error('[ERROR] Increase Num Training Trajectories or delta!');
end
%----------------------------------------------------------------------

% Dynamical model parameters
g = 10;               % gravity
mu = [0;0;0;0];       % mean of process noise
                      
Sigma = 0.011*eye(4); % cov of process noise
                      % Large covariance -> harder constraints


% World Parmeters                     
% [xlo ylo;xhi yhi]
Obstacle = [4 4;8 8]; % Obstacle bigger = harder
Goal = [10 10;12 12]; % Goal bigger = easier
Init = [0 0;2 2];     % This is not important, don't change
World = [-1 -1;20 20]; % This is not important

% state vector : (x,y,x-dot,y-dot)
% control input : (x-dot-dot, y-dot-dot)
% DeltaT = sampling time-step
DeltaT = 0.5;
A = [1 0 DeltaT 0;
    0 1 0 DeltaT;
    0 0 1 0;
    0 0 0 1];
B = [0.5*(DeltaT^2) 0;
    0 0.5*(DeltaT^2);
    DeltaT 0;
    0 DeltaT];

% Currently we pick a unique initial state at random
init = Init(1,:)+diff(Init).*rand(1,2);
init = [init';zeros(2,1)];

% STL property:
% With high prob:
% 1. Avoid Obstacle
% 2. Bounds on Velocity 
% 3. Eventually within [15 to 20 seconds] reach goal
% Input constraints:
% 1. X, Y Acceleration to be [-g,g]
% 2. X, Y velocities restricted to velBounds
% 3. X, Y positions restricted to World
accBounds = [-0.1*g -0.1*g;0.1*g 0.1*g];    % [acc_x_min, acc_y_min; acc_x_max acc_y_max]
velBounds = [0 0;20 20]; % [v_x_min v_y_min; v_x_max v_y_max]
goalInterval = [10 15];

xx = sdpvar(4,TimeLength);
uu = sdpvar(2,TimeLength-1);
bigM = 1e10;
epsi = 1e-10;

% Number of variables:
% GoalTimes = (diff(goalInterval) + 1)
% Continuous: 2*(TimeLength - 1)
% Binary: 
%       Obstacle : 4*(TimeLength-1)
%       Goal     : 4*(GoalTimes) + 1
% Integer:
%       Obstacle : (TimeLength-1) + 1
%       Goal     : GoalTimes + 1

constraints = [];
GoalBinVars = cell(numTrainingTraj,1);
ObsBinVars = cell(numTrainingTraj,1);
GoalConjunct = cell(numTrainingTraj,1);
GoalDisjunct = cell(numTrainingTraj,1);
ObsDisjunct = cell(numTrainingTraj,1);
ObsConjunct = cell(numTrainingTraj,1);
TraceSat = intvar(numTrainingTraj,1);
bigM2 = [bigM;bigM];
epsi2 = [epsi;epsi];
noises = zeros(numTrainingTraj,TimeLength-1,4);
for jj=1:numTrainingTraj
    for kk=1:(TimeLength-1)
        noises(jj,kk,:) = mu + Sigma*randn(4,1);
    end
end

for jj=1:numTrainingTraj
    ObsBinVars{jj} = binvar(4,TimeLength+1); % first index is unused      
    ObsDisjunct{jj} = intvar(1,TimeLength);
    ObsConjunct{jj} = intvar(1,1); %#ok<*SAGROW>
    myObsVars = ObsBinVars{jj};
    myd = ObsDisjunct{jj};

    GoalBinVars{jj} = binvar(4,diff(goalInterval)+1); 
    GoalConjunct{jj} = intvar(1,diff(goalInterval)+1);
    GoalDisjunct{jj} = intvar(1,1);
    myGoalVars = GoalBinVars{jj};  
    myc = GoalConjunct{jj};
    
    noise = reshape(noises(jj,:),4,TimeLength-1);
    for tt=1:1:(TimeLength-1)
        if (tt==1)
            xx(:,tt+1) = A*init + B*uu(:,tt) + noise(:,tt);
        else
            xx(:,tt+1) = A*xx(:,tt) + B*uu(:,tt) + noise(:,tt);
        end
        nuLow  = -xx(1:2,tt+1)+(Obstacle(1,:)')-[0.5;0.5];
        nuHigh =  xx(1:2,tt+1)-(Obstacle(2,:)')-[0.5;0.5];
        c1a = (nuLow <= bigM2.*myObsVars(1:2,tt+1) - epsi2);
        c1b = (-nuLow <= bigM2.*([1;1]-myObsVars(1:2,tt+1)) - epsi2);
        c2a = (nuHigh <= bigM2.*myObsVars(3:4,tt+1) - epsi2);
        c2b = (-nuHigh <= bigM2.*([1;1]-myObsVars(3:4,tt+1)) - epsi2);

        c3a = (uu(:,tt) >= accBounds(1,:)');
        c3b = (uu(:,tt) <= accBounds(2,:)');       
        c4a = xx(3:4,tt+1) >= velBounds(1,:)';
        c4b = xx(3:4,tt+1) <= velBounds(2,:)';
        c5a = xx(1:2,tt+1) >= World(1,:)';
        c5b = xx(1:2,tt+1) <= World(2,:)';

        constraints = [constraints;c1a;c1b;c2a;c2b;c3a;c3b;c4a;c4b;c5a;c5b]; %#ok<*AGROW>
        constraints = [constraints;repmat(myd(1,tt+1),4,1)>=myObsVars(:,tt+1)];
        constraints = [constraints;myd(1,tt+1)<=sum(myObsVars(:,tt+1))]; 

        if ((tt+1)>=goalInterval(1)) && ((tt+1) <=goalInterval(2))
            index = ((tt+1)-goalInterval(1))+1;
            muLow = xx(1:2,tt+1)-(Goal(1,:)');
            muHigh = -xx(1:2,tt+1)+(Goal(2,:)');
            c15a = (muLow <= bigM2.*myGoalVars(1:2,index) - epsi2);
            c15b = (-muLow <= bigM2.*([1;1]-myGoalVars(1:2,index)) - epsi2);
            c16a = (muHigh <= bigM2.*myGoalVars(3:4,index) - epsi2);
            c16b = (-muHigh <= bigM2.*([1;1]-myGoalVars(3:4,index)) - epsi2);
            constraints = [constraints;c15a;c15b;c16a;c16b];
            constraints = [constraints;myc(1,index)>=1-4+sum(myGoalVars(1:4,index))];
            constraints = [constraints;repmat(myc(1,index),4,1)<=myGoalVars(1:4,index)];
        end

    end
    constraints = [constraints;ObsConjunct{jj}>=1-TimeLength+sum(myd)];
    constraints = [constraints;repmat(ObsConjunct{jj},1,TimeLength)<=myd(1,:)];    

    constraints = [constraints;GoalDisjunct{jj}<=sum(myc)];
    constraints = [constraints;repmat(GoalDisjunct{jj},1,diff(goalInterval)+1)>=myc];

    constraints = [constraints;TraceSat(jj)<=GoalDisjunct{jj};TraceSat(jj)<=ObsConjunct{jj}];
    constraints = [constraints;TraceSat(jj)>=-1+GoalDisjunct{jj}+ObsConjunct{jj}];
end

% This is to make all trajectories satisfy the spec
% for jj=1:numTrainingTrajs
%   constraints = [constraints;TraceSat(jj)>=1];
% end
%%
constraints = [constraints;sum(TraceSat)>=quantileIndex];
objective = sum(xx(3:4,TimeLength)); % demanding to minimize the terminal velocity
diagnostics = optimize(constraints,objective);
%%
[model,~] = export(constraints,objective,sdpsettings('solver','gurobi'));
%%
usol = value(uu(:,:));
figure;
hold on;
drawSquare(Obstacle,'m');
drawSquare(Goal,'g');
trajs = printTraces(init,A,B,usol,numTrainingTraj,TimeLength,noises);
%%
figure;
h1 = subplot(3,1,1);
title('Open Loop Control');
h2 = subplot(3,1,2);
title('x(t),y(t)');
h3 = subplot(3,1,3);
title('x-dot(t),y-dot(t)');
hold on;
plot(h1,1:(TimeLength-1),usol(1,:),'-m',1:(TimeLength-1),usol(2,:),'-k');
for jj=1:numTrainingTraj    
    traj = trajs{jj};
    plot(h2,1:TimeLength,traj(1,:),'-r.',1:TimeLength,traj(2,:),'-b');   
    plot(h3,1:TimeLength,traj(3,:),'-r.',1:TimeLength,traj(4,:),'-b');
end
    
%%
% numCalibrationTraj = 100;


function drawSquare(coords,c)
    fill([coords(1,1) coords(2,1) coords(2,1) coords(1,1)],...
         [coords(1,2) coords(1,2) coords(2,2) coords(2,2)],c);
end

function trajs = printTraces(init,A,B,usol,numTrainingTraj,TimeLength,noises)
    trajs = {};
    traj = zeros(4,TimeLength);
    traj(:,1) = init;
    for jj=1:numTrainingTraj
        noise = reshape([noises(jj,:)],4,TimeLength-1);
        for tt=1:1:(TimeLength-1)            
            traj(:,tt+1) = A*traj(:,tt) + B*usol(:,tt) + noise(:,tt);           
        end
        trajs{end+1}=traj;        
        plot(traj(1,:),traj(2,:),'-k.','MarkerSize',10);
        plot(traj(1,end),traj(2,end),'kpentagram','MarkerSize',20)
    end
    plot(init(1),init(2),'rsq','MarkerSize',20);
end

