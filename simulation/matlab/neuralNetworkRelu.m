% function that describes the neural network
function input = neuralNetworkRelu(state, W, b)
    input = max(0,state*W+b');
end
