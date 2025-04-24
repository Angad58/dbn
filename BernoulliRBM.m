classdef BernoulliRBM < RBM & handle
    % BernoulliRBM: RBM for binary data (e.g., MNIST pixels).
    % Trains with Contrastive Divergence to learn patterns.

    properties
        W        % Weights [nHidden, nVis]
        b        % Visible bias [nVis, 1]
        c        % Hidden bias [nHidden, 1]
        opts     % Training options (batchsize, numepochs, etc.)
    end

    methods
        function rbm = BernoulliRBM(nVis, nHidden, opts)
            % Initialize RBM with zeros for weights and biases.
            % Inputs: nVis - Visible units (e.g., 784 for MNIST)
            %         nHidden - Hidden units (e.g., 500)
            %         opts - Training options
            if nargin > 0
                rbm.W  = zeros(nHidden, nVis, 'gpuArray');  % Weight matrix
                rbm.b  = zeros(nVis, 1, 'gpuArray');        % Visible bias
                rbm.c  = zeros(nHidden, 1, 'gpuArray');     % Hidden bias
                rbm.opts = opts;                            % Store options
            end
        end

        function x = rbmdown(rbm, x)
            % Get visible probabilities from hidden states.
            % Input: x - Hidden states [nSamples, nHidden]
            % Output: x - Visible probabilities [nSamples, nVis]
            x = logsig(repmat(rbm.b', size(x, 1), 1) + x * rbm.W);
        end

        function x = rbmup(rbm, x)
            % Get hidden probabilities from visible states.
            % Input: x - Visible states [nSamples, nVis]
            % Output: x - Hidden probabilities [nSamples, nHidden]
            x = logsig(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
        end

        function rbm = train(rbm, x)
            % Train RBM with Contrastive Divergence (CD-1).
            % Input: x - Data [nSamples, nVis], values in [0,1] (e.g., MNIST pixels)

            % Check input data is valid
            assert(isfloat(x), 'Data must be a float.');
            assert(all(x(:) >= 0) && all(x(:) <= 1), 'Data must be in [0,1].');

            % Set up training parameters
            n_samples = size(x, 1);
            batch_size = rbm.opts.batchsize;
            n_epochs = rbm.opts.numepochs;
            learning_rate = rbm.opts.alpha;
            weight_decay = rbm.opts.decay;
            cd_steps = rbm.opts.k; % Should be 1 for CD-1
            n_batches = n_samples / batch_size;

            % Make sure batch size works
            assert(rem(n_batches, 1) == 0, 'Batch size must divide data evenly.');

            % Move data to GPU for speed
            x = gpuArray(x);

            % Loop through training epochs
            for epoch_idx = 1:n_epochs
                % Shuffle data indices for random mini-batches
                shuffled_indices = randperm(n_samples);

                % Process each mini-batch
                for batch_idx = 1:n_batches
                    % Grab a mini-batch
                    batch_start = (batch_idx - 1) * batch_size + 1;
                    batch_end = batch_idx * batch_size;
                    batch_indices = shuffled_indices(batch_start:batch_end);
                    visible_pos = x(batch_indices, :); % Real data

                    % Positive phase: Get hidden states from data
                    hidden_pos = RBM.sample(repmat(rbm.c', batch_size, 1) + ...
                                           visible_pos * rbm.W');
                    % Formula: hidden = sample(sigmoid(W'*visible + c))

                    % Negative phase: Initialize with positive hidden states
                    hidden_neg = hidden_pos;

                    % Do one Gibbs sampling step (CD-1)
                    for step = 1:cd_steps
                        % Sample visible units
                        visible_neg = RBM.sample(repmat(rbm.b', batch_size, 1) + ...
                                                hidden_neg * rbm.W);
                        % Sample hidden units again
                        hidden_neg = RBM.sample(repmat(rbm.c', batch_size, 1) + ...
                                               visible_neg * rbm.W');
                    end

                    % Calculate updates
                    pos_stats = hidden_pos' * visible_pos; % Data correlations
                    neg_stats = hidden_neg' * visible_neg; % Model correlations

                    % Update weights and biases
                    rbm.W = rbm.W + learning_rate * (pos_stats - neg_stats - ...
                                                    weight_decay * rbm.W) / batch_size;
                    rbm.b = rbm.b + learning_rate * (sum(visible_pos - visible_neg)' - ...
                                                    weight_decay * rbm.b) / batch_size;
                    rbm.c = rbm.c + learning_rate * (sum(hidden_pos - hidden_neg)' - ...
                                                    weight_decay * rbm.c) / batch_size;
                end

                % Show progress
                fprintf('Finished epoch %d of %d.\n', epoch_idx, n_epochs);
            end
        end
    end
end
