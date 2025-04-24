classdef SoftmaxRBM < BernoulliRBM & handle
    % SoftmaxRBM: RBM for modeling features and labels together.
    % Uses binary hidden units, binary visible units, and softmax labels.

    properties
        U        % Softmax-to-hidden weights [nHidden, nClasses]
        d        % Softmax bias [nClasses, 1]
    end

    methods
        function rbm = SoftmaxRBM(nVis, nHidden, opts, nClasses)
            % Initialize RBM with softmax units for labels.
            % Inputs: nVis - Visible units (features)
            %         nHidden - Hidden units
            %         opts - Training options
            %         nClasses - Number of classes (e.g., 10 for MNIST)
            rbm@BernoulliRBM(nVis, nHidden, opts);
            rbm.d = zeros(nClasses, 1, 'gpuArray');    % Softmax bias
            rbm.U = zeros(nHidden, nClasses, 'gpuArray'); % Softmax weights
        end

        function rbm = train(rbm, x, y)
            % Train RBM to learn joint distribution of features and labels.
            % Inputs: x - Features [nSamples, nVis]
            %         y - One-hot labels [nSamples, nClasses]

            % Check input data
            assert(isfloat(x), 'Features must be a float.');
            assert(all(x(:) >= 0) && all(x(:) <= 1), 'Features must be in [0,1].');
            n_samples = size(x, 1);

            % Set up training parameters
            batch_size = rbm.opts.batchsize;
            n_epochs = rbm.opts.numepochs;
            learning_rate = rbm.opts.alpha;
            weight_decay = rbm.opts.decay;
            cd_steps = rbm.opts.k;
            n_batches = n_samples / batch_size;

            % Ensure batch size works
            assert(rem(n_batches, 1) == 0, 'Batch size must divide data evenly.');

            % Move data to GPU
            x = gpuArray(x);
            y = gpuArray(y);

            % Loop through epochs
            for epoch_idx = 1:n_epochs
                % Shuffle data for random mini-batches
                shuffled_indices = randperm(n_samples);

                % Process each mini-batch
                for batch_idx = 1:n_batches
                    % Grab a mini-batch
                    batch_start = (batch_idx - 1) * batch_size + 1;
                    batch_end = batch_idx * batch_size;
                    batch_indices = shuffled_indices(batch_start:batch_end);
                    visible_pos = x(batch_indices, :); % Features
                    labels_pos = y(batch_indices, :);  % Labels

                    % Positive phase: Get hidden states from features and labels
                    hidden_pos = RBM.sample(repmat(rbm.c', batch_size, 1) + ...
                                           visible_pos * rbm.W' + labels_pos * rbm.U');
                    % Formula: hidden = sample(sigmoid(W'*visible + U'*labels + c))

                    % Negative phase: Start with positive hidden states
                    hidden_neg = hidden_pos;

                    % Do one Gibbs sampling step
                    for step = 1:cd_steps
                        % Sample labels with softmax
                        labels_neg = softmax(repmat(rbm.d', batch_size, 1) + ...
                                           hidden_neg * rbm.U);
                        % Sample visible units
                        visible_neg = RBM.sample(repmat(rbm.b', batch_size, 1) + ...
                                                hidden_neg * rbm.W);
                        % Sample hidden units again
                        hidden_neg = RBM.sample(repmat(rbm.c', batch_size, 1) + ...
                                               visible_neg * rbm.W' + labels_neg * rbm.U');
                    end

                    % Calculate updates
                    pos_stats_features = hidden_pos' * visible_pos; % Feature correlations
                    neg_stats_features = hidden_neg' * visible_neg;
                    pos_stats_labels = hidden_pos' * labels_pos; % Label correlations
                    neg_stats_labels = hidden_neg' * labels_neg;

                    % Update weights and biases
                    rbm.W = rbm.W + learning_rate * (pos_stats_features - neg_stats_features - ...
                                                    weight_decay * rbm.W) / batch_size;
                    rbm.U = rbm.U + learning_rate * (pos_stats_labels - neg_stats_labels - ...
                                                    weight_decay * rbm.U) / batch_size;
                    rbm.b = rbm.b + learning_rate * (sum(visible_pos - visible_neg)' - ...
                                                    weight_decay * rbm.b) / batch_size;
                    rbm.c = rbm.c + learning_rate * (sum(hidden_pos - hidden_neg)' - ...
                                                    weight_decay * rbm.c) / batch_size;
                    rbm.d = rbm.d + learning_rate * (sum(labels_pos - labels_neg)' - ...
                                                    weight_decay * rbm.d) / batch_size;
                end

                % Show progress
                fprintf('Finished epoch %d of %d for top RBM.\n', epoch_idx, n_epochs);
            end
        end
    end
end
