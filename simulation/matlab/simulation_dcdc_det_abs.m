%% clear all
clc
clear all


%% load nn
cd nn
nn
cd ..

winning_x

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
tau = 0.5;
s = [0.6 5.3];

s_list = s;
u_list = [];
u_flag_list = [];

loop = 4000;

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
        
    % get state binary
    s_bin = x2bin(s, s_ipd, s_eta, s_ll, inputs);

    % get input for given state
    h = neuralNetworkSigmoid(s_bin, W{1}, b{1});
    for i = 2:layers-1
        h = neuralNetworkSigmoid(h, W{i}, b{i});
    end
    u_bin = neuralNetworkSigmoid(h, W{layers}, b{layers});
    u_bin = round(u_bin);
    
    if(u_bin(end) == 0)
        disp("the action is not a valid action");
        break
    end
    
    % bin input to input id
    u = bin2x(u_bin(1:end-1), u_ipd, u_eta, u_ll, outputs-1);
    
    % numerically integrate one tau
    u_list = [u_list; u];
    u_flag_list = [u_flag_list; u_bin(end)];
    [t s] = ode45(@ode_dcdc, [0 tau], s, odeset('abstol', 1e-10, 'reltol', 1e-10), u);
    s_list = [s_list; s];
    % s(end,:)
    loop = loop - 1;
end

%% plot system
colors = get(groot, 'DefaultAxesColorOrder');
box on

% plot domain
hold on
plot([s_ll(1) s_ur(1)],[s_ll(2) s_ur(2)],'.','color',0.6*ones(3,1))

% plot trajectory
plot(s_list(:,1),s_list(:,2),'k.-','color',colors(1,:),'markersize',1)

% plot initial state
plot(s_list(1,1), s_list(1,2),'.','color',colors(5,:),'markersize',20)

% plot boundary
v = [s_ll(1) s_ll(2);...
     s_ur(1) s_ll(2);...
     s_ll(1) s_ur(2);...
     s_ur(1) s_ur(2)];
patch('vertices',v,'faces',[1 2 4 3],'facecolor','none','edgec',colors(2,:),'linew',1)

grid on
axis([s_ll(1)-0.1 s_ur(1)+0.1 s_ll(2)-0.1 s_ur(2)+0.1])

wd = scatter(x{1},x{2},3,'r',"filled");
alpha(wd,0.5);
