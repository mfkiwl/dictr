classdef paramnoise
    % noise parameters
    % type   noise type ('G' for Gaussian white noise, or 'S' for signal-dependent noise)
    % nu     standard deviation of Gaussian noise if 'G', or readout noise if 'SD')
    % g      gain (for 'S')
    % intercept    intercept (for 'S')
    properties 
        type  
        nu     
        g     
        intercept 
    end
    methods
        function obj = paramnoise(type,a,b)
        obj.type=type;
        if (nargin==2)
            obj.nu=a;
            obj.g=0;
            obj.intercept=0;
        else
            obj.nu=0;
            obj.g=a;
            obj.intercept=b;
        end
        end
    end
end
