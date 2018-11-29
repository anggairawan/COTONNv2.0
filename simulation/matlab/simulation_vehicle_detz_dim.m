%% clear all
clc
clear all


%% load nn
cd nn
vehicle_detz
cd ..

%% controller parameters

% state space
tmp = size(s_ll);
s_dim = tmp(2);
s_ipd = calculateGridHelper(s_eta, s_ll, s_ur);

% input space
tmp = size(u_ll);
u_dim = tmp(2);
u_ipd = calculateGridHelper(u_eta, u_ll, u_ur);


%% nn parameters
tmp = size(W);
layers = tmp(2);

tmp = size(W{1});
inputs = tmp(1);

tmp = size(W{layers});
outputs = tmp(2);

%% simulation parameters
tau = 0.3;
%s = [9.4 0.67 -1.1]; line_color = 3;
%s = [5.5 9.5 0]; line_color = 3;
%s = [0 0 0]; line_color = 1;
%s = [6.7 0 0]; line_color = 5;
s = [6.2 0 -3.4]; line_color = 5;

s_list = s;
u_list = [];
u_flag_list = [];

loop = 500;

% target set
lb=[9 0];
ub=lb+0.5;

%% simulate system
while(loop>0)
    s = s_list(end,:);

    % check if state is (still) within the controller boundaries
    inside = true;
    for i=1:s_dim
       if(s(i) < s_ll(i) || s(i) > s_ur(i)) 
          inside = false;
       end
    end

    if(inside == false)
       disp("State is out of controller bounds")
       break
    end

    % stop when goal is reached
    if (lb(1) <= s(end,1) & s(end,1) <= ub(1) && lb(2) <= s(end,2) & s(end,2) <= ub(2))
        break;
    end 

    % get state binary
    s_bin = x2ssbin(s, s_ipd, s_eta, s_ll, s_ur, inputs);

    % get input for given state
    h = neuralNetworkSigmoid(s_bin, W{1}, b{1});
    for i = 2:layers-1
        h = neuralNetworkSigmoid(h, W{i}, b{i});
    end
    u_bin = neuralNetworkSoftmax(h, W{layers}, b{layers});
    u_bin_r = u_bin;

    [max_value, u_abs] = max(u_bin_r); 

    if(u_abs == outputs)
        loop
        disp("the action is not a valid action");
        break
    end

    u = i2x(u_abs-1, u_ipd, u_eta, u_ll);

    % numerically integrate one tau
    u_list = [u_list; u];
    u_flag_list = [u_flag_list; u_bin_r(end)];
    [t s] = ode45(@ode_vehicle, [0 tau], s_list(end,:), odeset('abstol',1e-12,'reltol',1e-12), u);
    s_list = [s_list; s];
    % s(end,:)
    loop = loop - 1;
end

%% plot system
colors = get(groot, 'DefaultAxesColorOrder');

% plot trajectory
hold on
plot(s_list(:,1),s_list(:,2),'k.-','color',colors(line_color,:),'markersize',0.1)
plot(s_list(1,1),s_list(1,2),'.','color',zeros(1,3),'markersize',20)
hold on

%%

box on
axis([-.5 10.5 -.5 10.5])

alpha=0.2;
v=[9 0; 9.5  0; 9 0.5; 9.5 .5];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(2,:),'edgec',colors(2,:));

% % plot boundary
% v = [s_ll(1) s_ll(2);...
%      s_ur(1) s_ll(2);...
%      s_ll(1) s_ur(2);...
%      s_ur(1) s_ur(2)];
% patch('vertices',v,'faces',[1 2 4 3],'facecolor','none','edgec',colors(2,:),'linew',1)

v=[1     0  ;1.2  0   ; 1     9    ; 1.2 9   ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[2.2   0  ;2.4  0   ; 2.2   5    ; 2.4 5   ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[2.2   6  ;2.4  6   ; 2.2   10   ; 2.4 10  ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[3.4   0  ;3.6  0   ; 3.4   9    ; 3.6 9   ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[4.6   1  ;4.8  1   ; 4.6   10   ; 4.8 10  ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[5.8   0  ;6    0   ; 5.8   6    ; 6   6   ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[5.8   7  ;6    7   ; 5.8   10   ; 6   10  ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[7     1  ;7.2  1   ; 7     10   ; 7.2 10  ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[8.2   0  ;8.4  0   ; 8.2   8.5  ; 8.4 8.5 ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[8.4   8.3;9.3  8.3 ; 8.4   8.5  ; 9.3 8.5 ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[9.3   7.1;10   7.1 ; 9.3   7.3  ; 10  7.3 ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[8.4   5.9;9.3  5.9 ; 8.4   6.1  ; 9.3 6.1 ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[9.3   4.7;10   4.7 ; 9.3   4.9  ; 10  4.9 ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[8.4   3.5;9.3  3.5 ; 8.4   3.7  ; 9.3 3.7 ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));
v=[9.3   2.3;10   2.3 ; 9.3   2.5  ; 10  2.5 ];
patch('vertices',v,'faces',[1 2 4 3],'facea',alpha,'facec',colors(1,:),'edgec',colors(1,:));

% w = scatter(x{1},x{2},3,'r',"filled");
% figure()
% wd = scatter3(x{1},x{2},x{3},3,'r',"filled");

xlabel('x_1') 
ylabel('x_2') 

%%