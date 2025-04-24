classdef BernoulliRBM < RBM & handle
    % BernoulliRBM: RBM with binary visible and hidden units for modeling binary data (e.g., MNIST).
    % Trains using Contrastive Divergence (CD-k).

    properties
        W        % Weight matrix [nHidden, nVis]
        vW       % Momentum for weights
        b        % Visible bias [nVis, 1]
        vb       % Momentum for visible bias
        c        % Hidden bias [nHidden, 1]
        vc       % Momentum for hidden bias
        opts     % Training options (batchsize, numepochs, alpha, etc.)
    end

    methods
        function rbm = BernoulliRBM(nVis, nHidden, opts)
            % Constructor: Initialize RBM with given sizes and options.
            % Inputs: nVis - Number of visible units
            %         nHidden - Number of hidden units
            %         opts - Training options
            if nargin > 0
                rbm.W  = zeros(nHidden, nVis, 'gpuArray');  % Weights
                rbm.vW = zeros(nHidden, nVis, 'gpuArray');  % Momentum
                rbm.b  = zeros(nVis, 1, 'gpuArray');        % Visible bias
                rbm.vb = zeros(nVis, 1, 'gpuArray');        % Bias momentum
                rbm.c  = zeros(nHidden, 1, 'gpuArray');     % Hidden bias
                rbm.vc = zeros(nHidden, 1, 'gpuArray');     % Bias momentum
                rbm.opts = opts;                            % Store options
            end
        end

        function x = rbmdown(rbm, x)
            % Compute visible probabilities from hidden states.
            % Input: x - Hidden states [nSamples, nHidden]
            % Output: x - Visible probabilities [nSamples, nVis]
            x = logsig(repmat(rbm.b', size(x, 1), 1) + x * rbm.W);
        end

        function x = rbmup(rbm, x)
            % Compute hidden probabilities from visible states.
            % Input: x - Visible states [nSamples, nVis]
            % Output: x - Hidden probabilities [nSamples, nHidden]
            x = logsig(repmat(rbm.c', size(x, 1), 1) + x * rbm.W');
        end

        function rbm = train(rbm, x)
            % Train the RBM using Contrastive Divergence (CD-k).
            % Input: x - Training data [nSamples, nVis], values in [0,1].

            % Validate input data
            assert(isfloat(x), 'Input data must be a float.');
            assert(all(x(:) >= 0) && all(x(:) <= 1), 'Data must be in [0,1].');

            % Get training parameters
            n_samples = size(x, 1);
            batch_size = rbm.opts.batchsize;
            n_epochs = rbm.opts.numepochs;
            learning_rate = rbm.opts.alpha;
            momentum = rbm.opts.momentum;
            weight_decay = rbm.opts.decay;
            cd_steps = rbm.opts.k;
            n_batches = n_samples / batch_size;

            % Ensure batch size divides data evenly
            assert(rem(n_batches, 1) == 0, 'Number of batches must be an integer.');

            % Move data to GPU for faster computation
            x = gpuArray(x);

            % Loop over epochs
            for epoch_idx = 1:n_epochs
                % Shuffle data indices to create random mini-batches
                shuffled_indices = randperm(n_samples);

                % Process each mini-batch
                for batch_idx = 1:n_batches
                    % Extract mini-batch
                    batch_start = (batch_idx - 1) * batch_size + 1;
                    batch_end = batch_idx * batch_size;
                    batch_indices = shuffled_indices(batch_start:batch_end);
                    visible_pos = x(batch_indices, :); % Data for positive phase

                    % Positive phase: Compute hidden states from data
                    hidden_pos = RBM.sample(repmat(rbm.c', batch_size, 1) + ...
                                           visible_pos * rbm.W');
                    % hidden_pos: [batch_size, nHidden], binary samples

                    % Initialize persistent chain for negative phase
                    if epoch_idx == 1 && batch_idx == 1
                        hidden_neg = hidden_pos;
                    end

                    % Negative phase: Perform k steps of Gibbs sampling
                    for step = 1:cd_steps
                        % Reconstruct visible units
                        visible_neg = RBM.sample(repmat(rbm.b', batch_size, 1) + ...
                                                hidden_neg * rbm.W);
                        % Recompute hidden units
                        hidden_neg = RBM.sample(repmat(rbm.c', batch_size, 1) + ...
                                               visible_neg * rbm.W');
                    end

                    % Compute gradients (positive - negative statistics)
                    pos_stats = hidden_pos' * visible_pos; % [nHidden, nVis]
                    neg_stats = hidden_neg' * visible_neg; % [nHidden, nVis]

                    % Update weights and biases with momentum
                    rbm.vW = momentum * rbm.vW + ...
                             learning_rate * (pos_stats - neg_stats - ...
                                             weight_decay * rbm.W) / batch_size;
                    rbm.vb = momentum * rbm.vb + ...
                             learning_rate * (sum(visible_pos - visible_neg)' - ...
                                             weight_decay * rbm.b) / batch_size;
                    rbm.vc = momentum * rbm.vc + ...
                             learning_rate * (sum(hidden_pos - hidden_neg)' - ...
                                             weight_decay * rbm.c) / batch_size;

                    % Apply updates to weights and biases
                    rbm.W = rbm.W + rbm.vW;
                    rbm.b = rbm.b + rbm.vb;
                    rbm.c = rbm.c + rbm.vc;
                end

                % Show progress
                fprintf('Epoch %d of %d completed for RBM.\n', epoch_idx, n_epochs);
            end
        end
    end
end
