classdef RBM < matlab.mixin.Heterogeneous & handle
    % RBM: Abstract base class for Restricted Boltzmann Machines.
    % Defines common methods and a static sampling function.

    properties
    end

    methods(Abstract)
        % Compute visible probabilities from hidden states
        rbmdown(rbm, x)
        % Compute hidden probabilities from visible states
        rbmup(rbm, x)
        % Train the RBM
        train(rbm, x, opts)
    end

    methods(Static)
        function x = sample(probabilities)
            % Sample binary states from probabilities using sigmoid.
            % Input: probabilities - Sigmoid outputs [nSamples, nUnits]
            % Output: x - Binary samples [nSamples, nUnits]
            x = double(logsig(probabilities) > rand(size(probabilities)));
        end
    end
end
