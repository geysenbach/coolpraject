%% Animate results

repSpeed = 1; % replay speed

%% Load python solution %%%%%%%%%%%%%%%%%%%%%%%%%%%%
load sol.mat
tStore = sol(:,1);
xStore = sol(:, 2:end);

[tStore,tIdx] = unique(tStore);
xStore = xStore(tIdx,:);

caseName = 'centipede';
caseList
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% compute leg positions for animation
rNVec = zeros(size(xStore,1),2*para.n); rCVec = zeros(size(xStore,1),2*para.n);
tOld = 0; cInfo0 = [];
for i = 1:size(xStore,1)-1
    q = xStore(i,1:para.dim)'; %%change 13 back to para.dim
    [cc,alpha,tEvent] = currentContacts(tStore(i),para);
    cInfo = contactInfo(cc,cInfo0,alpha,q,para);
    cInfo0 = cInfo;
    if ~isempty(cc)
        if tEvent~=tOld || i==1
            rCVec(i,1:2*size(cc,1)) = reshape(cInfo(:,3:4)',1,2*size(cInfo,1));
        else
            rCVec(i,1:2*size(cc,1)) = rCVec(i-1,1:2*size(cc,1));
        end
        rNVec(i,1:2*size(cc,1)) = reshape(cInfo(:,5:6)',1,2*size(cInfo,1));
    end
    tOld = tEvent;
end

n = para.n;
data = [tStore,xStore];
figure('units','normalized','outerposition',[0.25 0 0.5 1])
set(gcf,'color','w');

if strcmp(para.bendType, 'shape')
    subplot(1,2,2)
    %plot(para.sAmp*para.TMask(0),0:length(para.TMask(0))-1,'-o')
    r0 = interp1(data(:,1),data(:,2:3),0.01)';
    axis equal
    grid on
    hold on
    ka = line('color','k','LineWidth',1.5);
    
    
    axis([min(data(:,2))-5-para.g0 max(data(:,2))+5+para.g0 r0(2)-7 r0(2)+7+n])
    
    subplot(1,2,1)
end

%% Initialize line elements
for i = 1:n
    b = line('color','k','linewidth',2);
    l = line('color',[0.5 0.5 0.5],'linewidth',2);
    m = line('color','b','LineStyle','none','Marker','o','MarkerFaceColor','b');
    cLine = sprintf('l%i',i);    % Name of active pendulum in cLine (l_p1,l_p2,...)
    lineBody.(cLine) = b;   % Create struct "lines" with properties of each pendulum l_p1,l_p2,...
    lineLeg.(cLine) = l;
    marker.(cLine) = m;
end
rT = line('color','b','linewidth',2);

hold on;
axis equal;
axis([min(data(:,2))-0.2-1 max(data(:,2))+0.2+1 min(data(:,3))-3 max(data(:,3))+3+n])

grid on;
rS = []; % steps
rr = [];

%% animation
tic
while toc < data(end,1)/repSpeed
    tQ = toc*repSpeed; % Time of query
    if tQ<data(1,1)
        tQ = data(1,1);
    end
    [~,iter] = min(abs(data(:,1)-toc*repSpeed));
    
    r0 = interp1(data(:,1),data(:,2:3),tQ)';
    phi = interp1(data(:,1),data(:,4:3+n),tQ);
    %rn = interp1(data(:,1),rNVec,tQ);
    %rc = interp1(data(:,1),rCVec,tQ);
    rn = rNVec(iter,:);
    rc = rCVec(iter,:);
    
    rr = [rr;r0(1),r0(2)];
    set(rT,'XData',rr(:,1),'YData',rr(:,2));
    
    if para.bT ~= 0
        bState = para.TMask(mod(tQ,1));%sin(2*pi*(tQ - para.phiS - para.phiL));
        bState = [bState;0] ;%- [0;bState];
    end
    axis([min(data(:,2))-5-para.g0 max(data(:,2))+5+para.g0 r0(2)-7 r0(2)+7+n])
    
    
    for u = 1:n
        cLine = sprintf('l%u',u);
        rBody = 1/2*[sin(phi(:,u));-cos(phi(:,u))];
        if u>1
            r0 = r0 + 1/2*[-sin(phi(:,u-1));cos(phi(:,u-1))] +1/2*[-sin(phi(:,u));cos(phi(:,u))];
        end
        
        set(lineBody.(cLine),'XData', [r0(1)+rBody(1);r0(1)-rBody(1)],'YData', [r0(2)+rBody(2);r0(2)-rBody(2)]);
        if para.bT ~= 0
            col = [0.5 0.5 0.5] + 1*bState(u)*[0 0.5 0.5];
            col(col>1) = 1; col(col<0) = 0;
            set(lineBody.(cLine),'Color',col)
        end
        set(lineLeg.(cLine),'XData',[rn(1+2*(u-1));rc(1+2*(u-1))],'YData',[rn(2+2*(u-1));rc(2+2*(u-1))]);
        set(marker.(cLine),'XData',r0(1),'YData',r0(2));
        
    end
    drawnow
    
    if strcmp(para.bendType, 'shape')
        subplot(1,2,2)
        set(ka,'XData',para.sAmp*para.TMask(data(iter,1)),'YData',0:length(para.TMask(data(iter,1)))-1);
        %plot(para.sAmp*para.TMask(data(iter,1)),0:length(para.TMask(data(iter,1)))-1,'-o')
        r0 = interp1(data(:,1),data(:,2:3),0.01)';
        axis equal
        grid on
        
        axis([min(data(:,2))-5-para.g0 max(data(:,2))+5+para.g0 r0(2)-7 r0(2)+7+n])
        
        subplot(1,2,1)
    end
end

%% return current active contacts and next impact event time
function [cc,alpha,tEvent] = currentContacts(t,p)
% count from bottom element
% positive values are right body side, negative left
cc = []; alpha = []; tEvent = 0;
idx = find([p.ctrlTable(:,1);p.tSpan(end)]>t);
if ~isempty(idx) && idx(1)>1
    if idx(1)>size(p.ctrlTable,1)
        tEvent = p.tSpan(end);
    else
        tEvent = p.ctrlTable(idx(1),1); % time of next contact event
    end
    cc = p.ctrlTable(idx(1)-1,2:end)';
    cc = cc(cc~=0);
    alpha = sign(cc).*ones(length(cc),1)*p.alpha0; % current leg collision angles w.r.t. cc
    cc = abs(cc); % current closed contact indices
end

end

%% compute contact vectors
function cInfo = contactInfo(cc,cInfo0,alpha,q,p)
% cInfo encodes leg contact information: [element #, alpha, rC', rN']
% rC active contact point vectors, rN active element CoM vectors

if ~isempty(cc)
    rC = zeros(size(cc,1),2); rN = zeros(size(cc,1),2);
    
    % check if contact closed or switched leg
    if isempty(cInfo0)
        cLogic = zeros(p.n,1);
    else
        cc0 = cInfo0(:,1);
        alpha0 = cInfo0(:,2);
        
        aCmp = zeros(p.n,1); aCmp0 = zeros(p.n,1);
        aCmp(cc) = alpha; aCmp0(cc0) = alpha0;
        
        cLogic = sign(aCmp) == sign(aCmp0); % encodes if contact closed or switched leg
    end
    
    Jq0 = [eye(2).*q(1:2),1/2*[-sin(q(3:2+p.n))';cos(q(3:2+p.n))']]; % sum or Jq0 rows yields element com position
    
    % compute contact and CoM vectors
    for i=1:size(cc,1)
        if cc(i)==1
            rN(i,:) = q(1:2)';
        elseif cc(i)>2
            mask = [zeros(2,3),ones(2,cc(i)-2),zeros(2,p.dim-cc(i)-1)];
            Jq0m = Jq0 + Jq0.*mask;
            rN(i,:) = (Jq0m(:,1:2+cc(i))*ones(2+cc(i),1))';
        else
            rN(i,:) = (Jq0(:,1:2+cc(i))*ones(2+cc(i),1))';
        end
        if cLogic(cc(i)) ~= 1
            rC(i,:) = rN(i,:)+ p.g0*[sin(alpha(i)+q(cc(i)+2)),-cos(alpha(i)+q(cc(i)+2))];
        else
            rC(i,:) = cInfo0(cInfo0(:,1)==cc(i),3:4);
        end
    end
    
    cInfo = [cc,alpha,rC,rN];
else
    cInfo = [];
end
end


