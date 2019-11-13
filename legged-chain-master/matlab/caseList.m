switch caseName
    case 'newt'
        % parameters
        para.n = 2; % number of rigid bodies
        para.dim = 2+para.n; % dimension of generalized coordinates
        para.g0 = 1/3; % leg length to body length ratio
        para.b0 = 1/12; % segment moment of inertia
        para.alpha0 = pi/1.5; % Initial leg angle
        
        % initial conditions
        q0 = zeros(para.dim,1); % generalized coordinates
        u0 = zeros(para.dim,1); % generalized velocities
        
        % ingegration time span
        para.tSpan = [0 20];
        
        % control table
        ph = 0.5; % segment phase (relative to period)
        dc = 0.5; % leg duty cycle (relative to period)
        para.ctrlTable = legSequencer(dc,ph,para);
        
        % forces and torques
        para.T = -3e-6/(0.005*0.09^2*(1)^2); % non-dimensional leg torque (T/(ml^2f^2))
        para.k = 5e-5/(0.005*0.09^2*1^2); % non-dimensional inter-segment stiffness (k/(ml^2f^2))
        para.d = 2e-7/(0.005*0.09^2*1); % non-dimensional inter-segment damping (d/(ml^2f))
        para.bendType = 'none'; % no segment bending activation
        para.bT = 0; % non-dimensional segment bending torque
        
    case 'centipede'
        % "low gear" Gray Animal locomotion p366
        g_lbd = 1/1.6;
        g_fwbw = 8.5/1.5;
        
        para.n = 21; % number of segments
        para.dim = 2+para.n; % dimension of generalized coordinates
        para.g0 = 1.5; % leg length to body length ratio
        para.b0 = 1/12; % segment moment of inertia
        para.alpha0 = pi/2; % Initial leg angle
        
        % initial conditions
        q0 = zeros(para.dim,1); % generalized coordinates
        u0 = zeros(para.dim,1); % generalized velocities
        
        % ingegration time span
        para.tSpan = [0 30];
        
        % control table
        ph = 1-1/(g_lbd*para.n); % inter segment leg cycle phase difference (normalized) --> from Gray wavelength
        dc = 1/(1+g_fwbw); % leg duty cycle --> from Gray fw/bw cycle
        para.ctrlTable = legSequencer(dc,ph,para);
        para.bendType = 'none'; % no segment bending activation
        
        
        % forces and torques
        para.bT = 0; % non-dimensional segment bending torque
        para.T = -2e-7/((0.0116/para.n)*(0.152/para.n)^2*(1/0.6)^2); % non-dimensional leg torque (T/(ml^2f^2))
        para.k = 1e-5/((0.0116/para.n)*(0.152/para.n)^2*(1/0.6)^2); % non-dimensional inter-segment stiffness (k/(ml^2f^2))
        para.d = 1e-6/((0.0116/para.n)*(0.152/para.n)^2*(1/0.6)); % non-dimensional inter-segment damping (d/(ml^2f))
        
        %para.T = -2;
        %para.k = 50;
        %para.d = 0;
        
    case 'NewtBend'
        para.n = 2; % number of segments
        para.dim = 2+para.n; % dimension of generalized coordinates
        para.g0 = 1/2; % leg length to body length ratio
        para.b0 = 1/10; % non-dimensional segment moment of inertia
        para.alpha0 = 0.53*pi; % Initial leg angle
        
        % initial conditions
        q0 = zeros(para.dim,1); % generalized coordinates
        %q0(4) = -pi/8;
        %q0(5) = -pi/8;
        %q0(7) = pi/8;
        %q0(8) = pi/8;
        
        u0 = zeros(para.dim,1); % generalized velocities
        
        % control table
        para.tSpan = [0 10]; % ingegration time span
        ph = 0.9238; % segment phase (relative to period)
        dc = 0.15; % leg duty cycle (relative to period)
        ph = 0.6; % segment phase (relative to period)
        dc = 0.5; % leg duty cycle (relative to period)
        
        para.ctrlTable = legSequencer(dc,ph,para);
        %para.ctrlTable = [0 1 -4 7 0];
        %para.ctrlTable = [0 -3 6 0];
        
        % forces
        para.T = -1*0; % non-dimensional torque (T/(ml^2f^2))
        para.k = 5.e0; % non-dimensional inter-segment stiffness (k/(ml^2f^2))
        %         para.d = 1e-4; % non-dimensional inter-segment damping (d/(ml^2f))
        para.bT = -0.5; % non-dimensional bending torque (T/(ml^2f^2))
        
                
    otherwise
        disp(['case ',caseName,' not found']);
end
