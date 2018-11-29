% function that describes the neural network
function input = neuralNetworkSigmoid(state, W, b)
    input = sigmoid(state*W+b');
end