%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     Script to simulate mechanistic models                                   %%%
%%% 	    over different scaling of initial values                            %%%
%%% 	    and different alpha and KmLH value range                            %%%
%%% 		                                                                    %%%
%%%	INPUT PARAMETERS:											                %%%
%%%     model = String to model dde matlab function                             %%%
%%%     params_file = String to file with all parameters required for the model %%%
%%%     y0_file = String to file with initial 13 dimensional vector             %%%
%%%     options_file = String to file with options for matlab DDE solver by Shampine and Thompson [33].        %%%
%%%     t_max = Time to run the DDE for                                         %%%
%%%     n_t = Number of equally spaced time-points to evaluate the dde at       %%%
%%% 		                                                                    %%%
%%%	OUTPUT PARAMETERS:											                %%%
%%%     NONE                                                                    %%%
%%%     Simulated data is saved in '../data/y_alpha_KmLH/                       %%%
%%% 		                                                                    %%%
%%%	Example call:											                    %%%
%%%     simulate_dde_y_alpha_KmLH('clark', '../src/input/clark_params', '../src/input/clark_y_init_normal', '../src/input/options_file', 150, 150)  %%%
%%% 		                                                                    %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function simulate_dde_y_alpha_KmLH(model, params_file, y0_file, options_file, t_max, n_t)
    % Model to simulate
    model_dde=eval(['@' model '_dde'])

    % Load params
    params_file_id = fopen(params_file);
    C = textscan(params_file_id,'%s %n', 'Delimiter', '=');
    fclose(params_file_id);
    param_names=C{1};
    param_values=C{2};
    params=cell2struct(num2cell(param_values), param_names);
    % Lags
    if(strcmp(model, 'clark'))
        d_lags=[params.dP, params.dIh];
    elseif(strcmp(model, 'keener'))
        d_lags=[params.tau];
    else
        error(['Unknown model type: ' model]);
    end

    % Load Initial/history conditions
    y0_file_id = fopen(y0_file);
    C = textscan(y0_file_id,'%n', 'Delimiter', '=');
    fclose(y0_file_id);
    y0=C{1};

    % Options
    options_file_id = fopen(options_file);
    C = textscan(options_file_id,'%s');
    fclose(y0_file_id);
    options = eval(['ddeset(' C{1}{:} ')']);

    % Directory for data
    mkdir('../data', 'y_alpha_KmLH');
    %y_scales=linspace(0.7, 1.3, 101);
    y_scales=1.0
    alpha_ranges=linspace(0.7, 0.8, 21);
    KmLH_ranges=linspace(500, 800, 151);
    for y_scale=y_scales
        for KmLH=KmLH_ranges
            for alpha=alpha_ranges
                % Scale all
                disp(['y_scale=' num2str(y_scale) ' KmLH=' num2str(KmLH) ' alpha=' num2str(alpha)])
                % Parameters with new alpha and KmLH
                params.alpha=alpha;
                params.KmLH=KmLH;
        
                % Call dde23 to solve
                sol = dde23(model_dde, d_lags, y0*y_scale, [0,t_max], options, params);

                % Evaluate in points
                x = deval(sol,linspace(0,t_max,n_t));

                % Save state
                model_init=strsplit(y0_file, '/');
                dlmwrite(['../data/y_alpha_KmLH/x_' model_init{end} '_t' num2str(t_max) '_yscale_' num2str(y_scale) '_alpha_' num2str(alpha) '_KmLH_' num2str(KmLH)], x, 'precision', 10)
                
                % Observations
                y=zeros(5, n_t);
                % y(1)=LH=x(2)
                y(1,:)=x(1,:);
                % y(2)=FSH=x(4)
                y(2,:)=x(2,:);
                % y(3)=E2
                y(3,:)=params.e0+params.e1*x(6,:)+params.e2*x(7,:)+params.e3*x(13,:);
                % y(4)=P4
                y(4,:)=params.p0+params.p1*x(12,:)+params.p2*x(13,:);
                % y(5,:)=Ih
                y(5,:)=params.h0+params.h1*x(7,:)+params.h2*x(12,:)+params.h3*x(13,:);
                
                % Save observations
                model_init=strsplit(y0_file, '/');
                dlmwrite(['../data/y_alpha_KmLH/y_' model_init{end} '_t' num2str(t_max) '_yscale_' num2str(y_scale) '_alpha_' num2str(alpha) '_KmLH_' num2str(KmLH)], y, 'precision', 10)
            end
        end
    end
