%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generates the control table which contains information about the leg
% sequencing as a function of time. 
%
% inputs:
% dc            segment actuation duty cycle
% ph            segment phase difference
% para          system parameters
% 
% output: 
% ctrlTable     control table Format [time, contact #1, contact #2, ...; time2, ...]    
%               contact #k right leg: k, contact #k left leg: -k
%               fill with zeros if not enough contacts to fill row
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ctrlTable = legSequencer(dc,ph,para)
T = 1;

% leg 1 starts on right side of segment
boundsR =[mod(linspace(0,(para.n-1)*ph*T,para.n),T);...
          mod(linspace(dc*T,(para.n-1)*ph*T+dc*T,para.n),T)];
boundsL =[mod(linspace(0 + 0.5*T,(para.n-1)*ph*T + 0.5*T,para.n),T);...
          mod(linspace(dc*T + 0.5*T,(para.n-1)*ph*T+dc*T + 0.5*T,para.n),T)];
      
% contact switch times
t = [boundsR(1,:),boundsL(1,:)];
t = sort(t); td = t(2:end)-t(1:end-1)==0; t = [t(~td),t(end)];
t = repmat(t', 1, ceil(para.tSpan(end))); t = t' + T*(0:size(t',1)-1)';
t = reshape(t',1,size(t,1)*size(t,2));
t = t(t < para.tSpan(end));

% initialize control table
ctrlTable = zeros(size(t,2),1+para.n);
ctrlTable(:,1) = t;
t = mod(t+1e-10,T);

% Print active contact indices to control table
for i=1:length(t)
    aIdxR = (t(i) >= boundsR(1,:) & t(i) < boundsR(2,:) ) | ((t(i) >= boundsR(1,:) | t(i) < boundsR(2,:)) & boundsR(1,:) > boundsR(2,:));
    aIdxL = (t(i) >= boundsL(1,:) & t(i) < boundsL(2,:) ) | ((t(i) >= boundsL(1,:) | t(i) < boundsL(2,:)) & boundsL(1,:) > boundsL(2,:));

    aIdx = aIdxR - aIdxL;
    idx = find(abs(aIdx)==1);
    ctrlTable(i,2:length(idx)+1) = aIdx(idx).*idx;
end
end