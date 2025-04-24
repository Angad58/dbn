classdef DBN < handle
    % DBN: Deep Belief Network for MNIST classification/generation.
    % Pre-trains RBMs layer-wise using Contrastive Divergence.

    properties
        sizes    % Layer sizes [nVis, hidden1, hidden2, ...]
        rbm      % Array of RBMs (BernoulliRBM and SoftmaxRBM)
    end
    
    methods
        function dbn = DBN(x, y, sizes, opts)
            % Constructor: Initialize DBN with layer sizes and options.
            % Inputs: x - Features [nSamples, nFeatures]
            %         y - Labels [nSamples, nClasses]
            %         sizes - Hidden layer sizes
            %         opts - Training options
            n_features = size(x, 2);
            n_classes = size(y, 2);
            dbn.sizes = [n_features, sizes]; % Full architecture
            n_layers = numel(dbn.sizes) - 1;

            % Initialize lower RBMs (BernoulliRBM)
            for layer_idx = 1:n_layers-1
                dbn.rbm(layer_idx) = BernoulliRBM(dbn.sizes(layer_idx), ...
                                                 dbn.sizes(layer_idx + 1), opts);
            end

            % Initialize top RBM (SoftmaxRBM) with labels
            dbn.rbm(n_layers) = SoftmaxRBM(dbn.sizes(n_layers), ...
                                          dbn.sizes(n_layers + 1), opts, n_classes);
        end

        function train(dbn, x, y)
            % Train the DBN using greedy layer-wise pre-training.
            % Inputs: x - Features [nSamples, nFeatures]
            %         y - Labels [nSamples, nClasses]

            n_layers = numel(dbn.rbm);
            current_data = x;

            % Train each lower RBM
            for layer_idx = 1:n_layers-1
                % Train current RBM on input data
                fprintf('Training RBM %d (%d -> %d)...\n', ...
                        layer_idx, dbn.sizes(layer_idx), dbn.sizes(layer_idx + 1));
                train(dbn.rbm(layer_idx), current_data);

                % Compute hidden activations for next layer
                current_data = rbmup(dbn.rbm(layer_idx), current_data);
                % current_data: [nSamples, nHidden] for next RBM
            end

            % Train top RBM with features and labels
            fprintf('Training top RBM (%d -> %d, with %d classes)...\n', ...
                    dbn.sizes(n_layers), dbn.sizes(n_layers + 1), size(y, 2));
            train(dbn.rbm(n_layers), current_data, y);
        end
        
        function probs = predict(dbn, x, y)
            % Predict class probabilities for test data.
            % Inputs: x - Test features [nSamples, nFeatures]
            %         y - Test labels [nSamples, nClasses]
            % Output: probs - Class probabilities [nSamples, nClasses]

            n_layers = numel(dbn.rbm);
            n_samples = size(y, 1);
            n_classes = size(y, 2);
            current_data = x;

            % Propagate through lower layers to get penultimate activations
            for layer_idx = 2:n_layers
                current_data = rbmup(dbn.rbm(layer_idx - 1), current_data);
            end

            % Compute unnormalized probabilities for each class
            precom = repmat(dbn.rbm(n_layers).c', n_samples, 1) + ...
                    current_data * dbn.rbm(n_layers).W';
            probs = zeros(n_samples, n_classes, 'gpuArray');
            for class_idx = 1:n_classes
                probs(:, class_idx) = exp(dbn.rbm(n_layers).d(class_idx)) * ...
                    prod(1 + exp(precom + repmat(dbn.rbm(n_layers).U(:, class_idx)', n_samples, 1)), 2);
            end
        end

        function x = generate(dbn, class, c, nGibbSteps)
            % randomly initialize a single input
            x = rand(1, dbn.sizes(1));
            n = numel(dbn.rbm);

            % clamp softmax to this label
            y = zeros(1, c);
            y(class) = 1;

            % do an upward pass through the network for the test examples
            % to compute the feature activations in the penultimate layer
            for i = 1 : n - 1
                x = rbmup(dbn.rbm(i), x);
            end

            % do nGibbSteps iterations of gibbs sampling at the top layer
            for i = 1:nGibbSteps - 1
                h = RBM.sample(dbn.rbm(n).c' + x * dbn.rbm(n).W' + y * dbn.rbm(n).U');
                x = RBM.sample(dbn.rbm(n).b' + h * dbn.rbm(n).W);
            end
            h = RBM.sample(dbn.rbm(n).c' + x * dbn.rbm(n).W' + y * dbn.rbm(n).U');
            x = logsig(dbn.rbm(n).b' + h * dbn.rbm(n).W);

            % do a downward pass to generate sample
            for i = n-1:-1:1
                x = rbmdown(dbn.rbm(i), x);
            end
            x = reshape(x, 28, 28)';
        end
        
        function x = generate2(dbn, class, c, nGibbSteps)
            % randomly initialize the visbile units of the jointly trained layer
            x = rand(1, dbn.sizes(end - 1));
            n = numel(dbn.rbm);
            
            % clamp softmax to this label
            y = zeros(1, c);
            y(class) = 1;

            % do nGibbSteps iterations of gibbs sampling at the top layer
            for i = 1:nGibbSteps
                h = RBM.sample(dbn.rbm(n).c' + x * dbn.rbm(n).W' + y * dbn.rbm(n).U');
                x = RBM.sample(dbn.rbm(n).b' + h * dbn.rbm(n).W);
            end
            
            % do a downward pass to generate sample
            for i = n-1:-1:1
                x = rbmdown(dbn.rbm(i), x);
            end
            
        end
        
        function x = imageseq(dbn, class, c, nGibbSteps)
            % randomly initialize the visbile units of the jointly trained layer
            x = rand(1, dbn.sizes(end - 1));
            n = numel(dbn.rbm);
            
            % clamp softmax to this label
            y = zeros(1, c);
            y(class) = 1;

            % do nGibbSteps iterations of gibbs sampling at the top layer
            for i = 1:nGibbSteps
                h = RBM.sample(dbn.rbm(n).c' + x * dbn.rbm(n).W' + y * dbn.rbm(n).U');
                x = RBM.sample(dbn.rbm(n).b' + h * dbn.rbm(n).W);
                saveimg(dbn, x, n, class, i);
            end
%             h = RBM.sample(dbn.rbm(n).c' + x * dbn.rbm(n).W' + y * dbn.rbm(n).U');
%             x = logistic(dbn.rbm(n).b' + h * dbn.rbm(n).W);
% 
            
        end
        
        function saveimg(dbn, x, n, class, iter)
            % do a downward pass to generate sample
            for i = n-1:-1:1
                x = rbmdown(dbn.rbm(i), x);
            end
            imwrite(reshape(gather(x), 28, 28)', sprintf('figures/%d/%03d.png', class - 1, iter));
        end
    end
    
end

