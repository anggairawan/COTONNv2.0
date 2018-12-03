% function that describes the neural network
function input = neuralNetworkSoftmax(state, W, b)
    exp_results = exp(state*W+b');
    input = exp_results/sum(exp_results);
end