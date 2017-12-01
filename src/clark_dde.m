%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     Function that implements the Delayed Differential Equations             %%%
%%%         for model in Clark et al. [11]                                      %%%
%%% 		                                                                    %%%
%%%	INPUT PARAMETERS:											                %%%
%%%     t = time instant                                                        %%%
%%%     y = 13 dimensional vector with function values at time instant t        %%%
%%%     y_lagged = 13 dimensional vector with function values at t-tau          %%%
%%%     params = structure with all parameters required for the DDE             %%%
%%% 		                                                                    %%%
%%%	OUTPUT PARAMETERS:											                %%%
%%%     dydt = 13 dimensional vector of dde value                               %%%
%%% 		                                                                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function dydt=clark_dde(t,y,y_lagged,params)    
    % Auxialiary input functions
    % Ovarian hormone E2
    E2=params.e0+params.e1*y(6)+params.e2*y(7)+params.e3*y(13);
    % Ovarian hormone P4
    P4=params.p0+params.p1*y(12)+params.p2*y(13);
    % Ovarian hormone P4 delayed
    P4_dP=params.p0+params.p1*y_lagged(12,params.dP_idx)+params.p2*y_lagged(13,params.dP_idx);
    % Ovarian hormone Ih
    Ih=params.h0+params.h1*y(7)+params.h2*y(12)+params.h3*y(13);
    % Ovarian hormone Ih
    Ih_dIh=params.h0+params.h1*y_lagged(7,params.dIh_idx)+params.h2*y_lagged(12,params.dIh_idx)+params.h3*y_lagged(13,params.dIh_idx);
    
    % States
    % Allocation
    dydt=zeros(size(y));
    % y(1)=RP_LH
    dydt(1)=(params.V0LH+params.V1LH*E2^8/(params.KmLH^8+E2^8))/(1+P4_dP/params.KiLHP)-params.kLH*(1+params.cLHP*P4)*y(1)/(1+params.cLHE*E2);
    % y(2)=LH
    dydt(2)=params.kLH*(1+params.cLHP*P4)*y(1)/(params.nu*(1+params.cLHE*E2)) - params.aLH*y(2);
    % y(3)=RP_FSH
    dydt(3)=params.VFSH/(1+Ih_dIh/params.KiFSHIh)-params.kFSH*(1+params.cFSHP*P4)*y(3)/(1+params.cFSHE*E2^2);
    % y(4)=FSH
    dydt(4)=params.kFSH*(1+params.cFSHP*P4)*y(3)/(params.nu*(1+params.cFSHE*E2^2)) - params.aFSH*y(4);
    % y(5)=RcF
    dydt(5)=params.b*y(4)+(params.c1*y(4)-params.c2*y(2)^params.alpha)*y(5);
    % y(6)=SeF
    dydt(6)=params.c2*y(2)^params.alpha*y(5)+(params.c3*y(2)^params.beta-params.c4*y(2))*y(6);
    % y(7)=PrF
    dydt(7)=params.c4*y(2)*y(6)-params.c5*y(2)^params.gamma*y(7);
    % y(8)=Sc1
    dydt(8)=params.c5*y(2)^params.gamma*y(7)-params.d1*y(8);
    % y(9)=Sc2
    dydt(9)=params.d1*y(8)-params.d2*y(9);
    % y(10)=Lut1
    dydt(10)=params.d2*y(9)-params.k1*y(10);
    % y(11)=Lut2
    dydt(11)=params.k1*y(10)-params.k2*y(11);
    % y(12)=Lut3
    dydt(12)=params.k2*y(11)-params.k3*y(12);
    % y(13)=Lut4
    dydt(13)=params.k3*y(12)-params.k4*y(13);

