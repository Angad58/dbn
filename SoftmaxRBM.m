classdef SoftmaxRBM < BernoulliRBM & handle
    % SoftmaxRBM: RBM with binary hidden units, binary visible units, and softmax label units.
    % Models joint distribution of features and labels (e.g., MNIST digits).

    properties
        U        % Softmax-to-hidden weights [nHidden, nClasses]
        vU       % Momentum for U
        d        % Softmax bias [nClasses, 1]
        vd       % Momentum for d
    end

    methods
        function rbm = SoftmaxRBM(nVis, nHidden, opts, nClasses)
            % Constructor: Initialize RBM with softmax units for labels.
            % Inputs: nVis - Visible units (features)
            %         nHidden - Hidden units
            %         opts - Training options
            %         nClasses - Number of label classes
            rbm@BernoulliRBM(nVis, nHidden, opts);
            rbm.d = zeros(nClasses, 1, 'gpuArray');    % Softmax bias
            rbm.vd = zeros(nClasses, 1, 'gpuArray');   % Bias momentum
            rbm.U = zeros(nHidden, nClasses, 'gpuArray'); % Softmax weights
            rbm.vU = zeros(nHidden, nClasses, 'gpuArray'); % Weights momentum
        end

        function rbm = train(rbm, x, y)
            % Train the RBM to model joint distribution of features and labels.
            % Inputs: x - Features [nSamples, nVis], values in [0,1]
            %         y - One-hot labels [nSamples, nClasses]

            % Validate inputs
            assert(isfloat(x), 'Feature data must be a float.');
            assert(all(x(:) >= 0) && all(x(:) <= 1), 'Feature data must be in [0,1].');
            n_samples = size(x, 1);

            % Get training parameters
            batch_size = rbm.opts.batchsize;
            n_epochs = rbm.opts.numepochs;
            learning_rate = rbm.opts.alpha;
            momentum = rbm.opts.momentum;
            weight_decay = rbm.opts.decay;
            cd_steps = rbm.opts.k;
            n_batches = n_samples / batch_size;

            % Ensure batch size divides data evenly
            assert(rem(n_batches, 1) == 0, 'Number of batches must be an integer.');

            % Move data to GPU
            x = gpuArray(x);
            y = gpuArray(y);

            % Loop over epochs
            for epoch_idx = 1:n_epochs
                % Shuffle data indices for random mini-batches
                shuffled_indices = randperm(n_samples);

                % Process each mini-batch
                for batch_idx = 1:n_batches
                    % Extract mini-batch
                    batch_start = (batch_idx - 1) * batch_size + 1;
                    batch_end = batch_idx * batch_size;
                    batch_indices = shuffled_indices(batch_start:batch_end);
                    visible_pos = x(batch_indices, :); % Feature batch
                    labels_pos = y(batch_indices, :);  % Label batch

                    % Positive phase: Compute hidden states from features and labels
                    hidden_pos = RBM.sample(repmat(rbm.c', batch_size, 1) + ...
                                           visible_pos * rbm.W' + labels_pos * rbm.U');
                    % hidden_pos: [batch_size, nHidden], binary samples

                    % Initialize persistent chain
                    if epoch_idx == 1 && batch_idx == 1
                        hidden_neg = hidden_pos;
                    end

                    % Negative phase: Perform k steps of Gibbs sampling
                    for step = 1:cd_steps
                        % Sample labels using softmax
                        labels_neg = softmax(repmat(rbm.d', batch_size, 1) + ...
                                           hidden_neg * rbm.U);
                        % Reconstruct features
                        visible_neg = RBM.sample(repmat(rbm.b', batch_size, 1) + ...
                                                hidden_neg * rbm.W);
                        % Recompute hidden states
                        hidden_neg = RBM.sample(repmat(rbm.c', batch_size, 1) + ...
                                               visible_neg * rbm.W' + labels_neg * rbm.U');
                    end

                    % Compute gradients
                    pos_stats_features = hidden_pos' * visible_pos; % [nHidden, nVis]
                    neg_stats_features = hidden_neg' * visible_neg;
                    pos_stats_labels = hidden_pos' * labels_pos; % [nHidden, nClasses]
                    neg_stats_labels = hidden_neg' * labels_neg;

                    % Update weights and biases with momentum
                    rbm.vW = momentum * rbm.vW + ...
                             learning_rate * (pos_stats_features - neg_stats_features - ...
                                             weight_decay * rbm.W) / batch_size;
                    rbm.vU = momentum * rbm.vU + ...
                             learning_rate * (pos_stats_labels - neg_stats_labels - ...
                                             weight_decay * rbm.U) / batch_size;
                    rbm.vb = momentum * rbm.vb + ...
                             learning_rate * (sum(visible_pos - visible_neg)' - ...
                                             weight_decay * rbm.b) / batch_size;
                    rbm.vc = momentum * rbm.vc + ...
                             learning_rate * (sum(hidden_pos - hidden_neg)' - ...
                                             weight_decay * rbm.c) / batch_size;
                    rbm.vd = momentum * rbm.vd + ...
                             learning_rate * (sum(labels_pos - labels_neg)' - ...
                                             weight_decay * rbm.d) / batch_size;

                    % Apply updates
                    rbm.W = rbm.W + rbm.vW;
                    rbm.U = rbm.U + rbm.vU;
                    rbm.b = rbm.b + rbm.vb;
                    rbm.c = rbm.c + rbm.vc;
                    rbm.d = rbm.d + rbm.vd;
                end

                % Show progress
                fprintf('Epoch %d of %d completed for SoftmaxRBM.\n', epoch_idx, n_epochs);
            end
        end
    end
end
