rng(0,'twister');
yalmip('clear');
close all;
clear;


%----------------------------------------------------------------------
% Increase TimeLength: Makes it harder. 
TimeLength = 15; 

% Increase numTrainingTraj: Makes it harder. 
numTrainingTraj = 20; 
%----------------------------------------------------------------------

%----------------------------------------------------------------------
% CPP parameters. 
delta = 0.1; % this has no effect on number of constraints or runtime
quantileIndex = ceil((numTrainingTraj+1)*(1-delta));
if(quantileIndex>numTrainingTraj)
    error('[ERROR] Increase Num Training Trajectories or delta!');
end
%----------------------------------------------------------------------

% Dynamical model parameters
g = 10;               % gravity
mu = [0;0;0;0];       % mean of process noise
                      
% Covariance of 0.02 breaks the system, anything lower than 0.011 works
       % lower the covariance, easier it is to find a solution.
Sigma = 0.01*eye(4); % cov of process noise
                      % Large covariance -> harder constraints


% World Parmeters                     
% [xlo ylo;xhi yhi]
Obstacle = [4 4;6 6]; % Obstacle bigger = harder to find solution

% Goals bigger = easier to find solution
Goal1 = [6 1;8 3]; 
Goal2 = [2 8;3 10];
% bigger interval adds more constraints and variables (though it will make
% the problem easier) 
% goaljInterval cannot start with 0 -- stupid encoding thing
goalInterval1 = [2 6];
goalInterval2 = [3 7];

Init = [0 0;2 2];     % This is not important, no reason to change now
World = [-1 -1;20 20]; % This is not important, just puts bounds on all
                       % continuous variables

% state vector : (x,y,x-dot,y-dot)
% control input : (x-dot-dot, y-dot-dot)
% DeltaT = sampling time-step; only used for dynamics, all times in
%          formulas are just integers (and not real-time). To get real-time
%          multiply integral values by DeltaT
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

% wider acceleration bounds makes the problem easier
accBounds = 0.5*[-g -g;g g];    % [acc_x_min, acc_y_min; acc_x_max acc_y_max]

% wider velocity bounds makes the problem easier but also makes it easier
% to "jump" over obstacle. 
velBounds = [-10 -10;10 10]; % [v_x_min v_y_min; v_x_max v_y_max]
xx = sdpvar(4,TimeLength+1);
uu = sdpvar(2,TimeLength);
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

%% The various variables needed for STL encoding
Goal1Disjunct = cell(numTrainingTraj,1);
Goal1Conjunct = cell(numTrainingTraj,1);
Goal1BinVars = cell(numTrainingTraj,1);
Goal2Disjunct = cell(numTrainingTraj,1);
Goal2Conjunct = cell(numTrainingTraj,1);
Goal2BinVars = cell(numTrainingTraj,1);

ObsBinVars = cell(numTrainingTraj,1);
ObsDisjunct = cell(numTrainingTraj,1);
ObsConjunct = cell(numTrainingTraj,1);
TraceSat = intvar(numTrainingTraj,1);
%% bigM and smallM
bigM2 = [bigM;bigM];
epsi2 = [epsi;epsi];
%% sample all noise up-front
noises = zeros(numTrainingTraj,TimeLength,4);
for jj=1:numTrainingTraj
    for kk=1:(TimeLength-1)
        noises(jj,kk,:) = mu + Sigma*randn(4,1);
    end
end
%% create constraints for each trajectory
for jj=1:numTrainingTraj
    % These binary vars are used to encode being always outside the obstacle 
    ObsBinVars{jj} = binvar(4,TimeLength); % first index is unused      
    ObsDisjunct{jj} = intvar(1,TimeLength);
    ObsConjunct{jj} = intvar(1,1); %#ok<*SAGROW>
    myObsVars = ObsBinVars{jj};
    myd = ObsDisjunct{jj};
    noise = reshape(noises(jj,:),4,TimeLength);
    for tt=1:TimeLength
        % generate symbolic trajectory
        if (tt==1)
            xx(:,tt+1) = A*init + B*uu(:,tt) + noise(:,tt);
        else
            xx(:,tt+1) = A*xx(:,tt) + B*uu(:,tt) + noise(:,tt);
        end        
        % obstacle encoding
        nuLow  = -xx(1:2,tt+1)+(Obstacle(1,:)')-[1;1];
        nuHigh =  xx(1:2,tt+1)-(Obstacle(2,:)')-[1;1];
        c1a = (nuLow <= bigM2.*myObsVars(1:2,tt) - epsi2);
        c1b = (-nuLow <= bigM2.*([1;1]-myObsVars(1:2,tt)) - epsi2);
        c2a = (nuHigh <= bigM2.*myObsVars(3:4,tt) - epsi2);
        c2b = (-nuHigh <= bigM2.*([1;1]-myObsVars(3:4,tt)) - epsi2);

        % bounds on control inputs, velocity, and position
        c3a = (uu(:,tt) >= accBounds(1,:)');
        c3b = (uu(:,tt) <= accBounds(2,:)');       
        c4a = xx(3:4,tt+1) >= velBounds(1,:)';
        c4b = xx(3:4,tt+1) <= velBounds(2,:)';
        c5a = xx(1:2,tt+1) >= World(1,:)';
        c5b = xx(1:2,tt+1) <= World(2,:)';
        constraints = [constraints;c1a;c1b;c2a;c2b;c3a;c3b;c4a;c4b;c5a;c5b]; %#ok<*AGROW>

        % encode obstacle
        constraints = [constraints;repmat(myd(1,tt),4,1)>=myObsVars(:,tt)];
        constraints = [constraints;myd(1,tt)<=sum(myObsVars(:,tt))]; 
    end
    % obstacle encoding (conjunction over all times)
    constraints = [constraints;ObsConjunct{jj}>=1-TimeLength+sum(myd)];
    constraints = [constraints;repmat(ObsConjunct{jj},1,TimeLength)<=myd(1,:)];    
     

    [const,Goal1BinVars{jj},Goal1Conjunct{jj},Goal1Disjunct{jj},...
           Goal2BinVars{jj},Goal2Conjunct{jj},Goal2Disjunct{jj}] = ...
        createNestedGoalEncoding(xx,bigM2,epsi2,Goal1,goalInterval1,Goal2,goalInterval2);
    
    % conjunction of the two specs
    constraints = [constraints;const];
    constraints = [constraints;...
                   TraceSat(jj)<=Goal1Disjunct{jj};...                  
                   TraceSat(jj)<=ObsConjunct{jj}];
    constraints = [constraints;TraceSat(jj)>=1-2+...                              
                               Goal1Disjunct{jj}+...
                               ObsConjunct{jj}];
                              

end

% This is to make all trajectories satisfy the spec: not CPP!
% for jj=1:numTrainingTrajs
%   constraints = [constraints;TraceSat(jj)>=1];
% end
%% CPP encoding
constraints = [constraints;sum(TraceSat)>=quantileIndex];
z = sdpvar(2,TimeLength);
for jj=1:TimeLength
    constraints = [constraints;z(:,jj) >= uu(:,jj);z(:,jj) >= -uu(:,jj)];
end
objective = sum(z,"all");
diagnostics = optimize(constraints,objective);
%% export to gurobi model to .mat
[model,~] = export(constraints,objective,sdpsettings('solver','gurobi'));
fname = sprintf('%s-(numTraces-%d)-(Horizon-%d)-model.mat',datetime("today"),numTrainingTraj,TimeLength);
save(fname,'model');
%% pretty plotting
usol = value(uu(:,:));
figure;
hold on;
drawSquare(Obstacle,'m');
drawSquare(Goal1,'y');
drawSquare(Goal2,'g')
trajs = printTraces(init,A,B,usol,numTrainingTraj,TimeLength,noises);
%% plots for debugging
figure;
h1 = subplot(3,1,1);
title('Open Loop Control');
h2 = subplot(3,1,2);
title('x(t),y(t)');
h3 = subplot(3,1,3);
title('x-dot(t),y-dot(t)');
hold on;
plot(h1,1:TimeLength,usol(1,:),'-m',1:TimeLength,usol(2,:),'-k');
for jj=1:numTrainingTraj    
    traj = trajs{jj};
    plot(h2,1:TimeLength+1,traj(1,:),'-r.',1:TimeLength+1,traj(2,:),'-b');   
    plot(h3,1:TimeLength+1,traj(3,:),'-r.',1:TimeLength+1,traj(4,:),'-b');
end
%%
% numCalibrationTraj = 100;

%% helper functions to draw stuff
function drawSquare(coords,c)
    fill([coords(1,1) coords(2,1) coords(2,1) coords(1,1)],...
         [coords(1,2) coords(1,2) coords(2,2) coords(2,2)],c);
end

%% helper function to draw trajectories
function trajs = printTraces(init,A,B,usol,numTrainingTraj,TimeLength,noises)
    trajs = {};
    traj = zeros(4,TimeLength+1);
    traj(:,1) = init;
    for jj=1:numTrainingTraj
        noise = reshape([noises(jj,:)],4,TimeLength);
        for tt=1:1:TimeLength
            traj(:,tt+1) = A*traj(:,tt) + B*usol(:,tt) + noise(:,tt);           
        end
        trajs{end+1}=traj;        
        plot(traj(1,:),traj(2,:),'-k.','MarkerSize',10);
        plot(traj(1,end),traj(2,end),'kpentagram','MarkerSize',20)
    end
    plot(init(1),init(2),'rsq','MarkerSize',20);
end

function [constraints,bvars1,cvars1,dvars1,bvars2,cvars2,dvars2] = createNestedGoalEncoding(state,m2,e2,goal1,interval1,goal2,interval2)
    numSteps1 = diff(interval1)+1;
    % we cannot constrain state(:,1) as it is the initial state. 
    assert(interval1(1)>0);
    bvars1 = binvar(4,numSteps1);
    cvars1 = intvar(1,numSteps1);
    dvars1 = intvar(1,1);
    constraints = [];

    numSteps2 = diff(interval1+interval2)+1;
    cvars2 = intvar(1,numSteps2);
    bvars2 = binvar(4,numSteps2);
    dvars2 = intvar(1,numSteps1);
    % matlab arrays start at index 1 which is time 0, so all time indices are +1
    for tt=(interval1(1)+1):(interval1(2)+1) 
        index1 = tt-interval1(1);
        muLow1 = state(1:2,tt)-(goal1(1,:)');
        muHigh1 = -state(1:2,tt)+(goal1(2,:)');
        low1A = (muLow1 <= m2.*bvars1(1:2,index1) - e2);
        low1B = (-muLow1 <= m2.*([1;1]-bvars1(1:2,index1)) - e2);
        high1A = (muHigh1 <= m2.*bvars1(3:4,index1) - e2);
        high1B = (-muHigh1 <= m2.*([1;1]-bvars1(3:4,index1)) - e2);

        % establishing binary variables for x(tt) in goal1 (4 vars for 4 edges)
        constraints = [constraints;low1A;low1B;high1A;high1B];        
               
        for vv=(tt+interval2(1)):(tt+interval2(2))
            index2 = vv-(interval1(1)+interval2(1));
            muLow2 = state(1:2,vv)-(goal2(1,:)');
            muHigh2 = -state(1:2,vv)+(goal2(2,:)');            
            low2A = (muLow2 <= m2.*bvars2(1:2,index2) - e2);
            low2B = (-muLow2 <= m2.*([1;1]-bvars2(1:2,index2)) - e2);
            high2A = (muHigh2 <= m2.*bvars2(3:4,index2) - e2);
            high2B = (-muHigh2 <= m2.*([1;1]-bvars2(3:4,index2)) - e2);

            %  binary variables for being in goal
            constraints = [constraints;low2A;low2B;high2A;high2B];

            % establishing binary variables cvars2() for each x(uu) in goal2 
            constraints = [constraints;cvars2(1,index2)>=1-4+sum(bvars2(1:4,index2))];
            constraints = [constraints;repmat(cvars2(1,index2),4,1)<=bvars2(1:4,index2)];
        end
        
        index1end = index1+diff(interval2);
        % OR_{vv in tt+interval2} (x(uu) in goal2)
        constraints = [constraints;dvars2(1,index1)<=sum(cvars2(1,index1:index1end))];
        constraints = [constraints;repmat(dvars2(1,index1),1,...
                                          diff(interval2)+1)>=cvars2(1,index1:index1end)];        
        
        % (x(tt) in goal1):4 edges and OR_{uu in tt+interval2} (x(uu) in goal2)
        constraints = [constraints;cvars1(1,index1)>=1-5+(sum(bvars1(1:4,index1))+dvars2(1,index1))];
        constraints = [constraints;repmat(cvars1(1,index1),5,1)<=[bvars1(1:4,index1);dvars2(1,index1)]];
    end    
    constraints = [constraints;dvars1<=sum(cvars1)];
    constraints = [constraints;repmat(dvars1,1,numSteps1)>=cvars1];
end

% function [constraints,bvars,cvars,dvars] = createGoalEncoding(state,m2,e2,goal,interval)
%     % reminder, interval cannot start at 0 for stupid reasons
%     assert(interval(1)>0);
%     [constraints,bvars,cvars,dvars] = createGoalEncodingAtTime(0,state,m2,e2,goal,interval);
% end
% 
% function [constraints,bvars,cvars,dvars] = createGoalEncodingAtTime(state,m2,e2,goal,interval)
%     numSteps = diff(interval)+1;
%     bvars = binvar(4, numSteps);
%     cvars = intvar(1,numSteps);
%     dvars = intvar(1,1);
%     constraints = [];
%     interval = startTime + interval;
%     for tt=interval(1):1:interval(2)
%         index = tt-interval(1)+1;
%         muLow = state(1:2,tt)-(goal(1,:)');
%         muHigh = -state(1:2,tt)+(goal(2,:)');
%         lowA = (muLow <= m2.*bvars(1:2,index) - e2);
%         lowB = (-muLow <= m2.*([1;1]-bvars(1:2,index)) - e2);
%         highA = (muHigh <= m2.*bvars(3:4,index) - e2);
%         highB = (-muHigh <= m2.*([1;1]-bvars(3:4,index)) - e2);
%         % goal encoding
%         constraints = [constraints;lowA;lowB;highA;highB];
%         constraints = [constraints;cvars(1,index)>=1-4+sum(bvars(1:4,index))];
%         constraints = [constraints;repmat(cvars(1,index),4,1)<=bvars(1:4,index)];
%     end
%     constraints = [constraints;dvars<=sum(cvars)];
%     constraints = [constraints;repmat(dvars,1,numSteps)>=cvars];
% end