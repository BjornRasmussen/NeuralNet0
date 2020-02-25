package net;

public class NeuralNetTrainer {
    private NeuralNet _net;
    private final static double DESCENT_RATE_WEIGHTS = 0.1;
    private final static double DESCENT_RATE_BIASES = 0.1;
    private final static double PROPAGATION_WEIGHT = 1;

    public NeuralNetTrainer(NeuralNet net, boolean startRandom) {
        _net = net;
        if (startRandom) {
            generateRandomNet(_net);
        }
    }

    private void generateRandomNet(NeuralNet net) {
        // Dig into the network, and set every little setting to something completely random.
        for (int i = 0; i < net.getWeights().length; i++) {
            for (int j = 0; j < net.getWeights()[i].length; j++) {
                for (int k = 0; k < net.getWeights()[i][j].length; k++) {
                    net.getWeights()[i][j][k] = (Math.random()-0.5) * 2; // -1 to 1
                }
            }
        }

        for (int i = 0; i < net.getBiases().length; i++) {
            for (int j = 0; j < net.getBiases()[i].length; j++) {
                net.getBiases()[i][j] = (Math.random()-0.5) * 2; // -1 to 1
            }
        }
    }

    public void trainNetwork(double[][] inputs, double[][] targets, int iterations) {
        // Use this to create the "mini-batches", and call improveNetwork on each.
        // For now, just treat everything as a single mini-batch.
        for (int i = 0; i < iterations; i++) {
            // Each iteration takes a step down towards a valley.
            stepDeeper(inputs, targets);
            System.out.println(getAveragePoints(inputs, targets, _net));
        }
    }

    public void stepDeeper(double[][] inputs, double[][] targets) {
        // Use back propagation for each input in the step, and then add up the suggested changes.
        double[][][] weightChanges = new double[_net.getWeights().length][][];
        for (int i = 0; i < weightChanges.length; i++) {
            weightChanges[i] = new double[_net.getWeights()[i].length][];
            for (int j = 0; j < _net.getWeights()[i].length; j++) {
                weightChanges[i][j] = new double[_net.getWeights()[i][j].length];
            }
        }

        double[][] biasChanges = new double[_net.getBiases().length][];
        for (int i = 0; i < _net.getBiases().length; i++) {
            biasChanges[i] = new double[_net.getWeights()[i].length];
        }

        // Now, back propagate through, updating modifications along the way.
        for (int i = 0; i < inputs.length; i++) {
            backPropagate(inputs[i], targets[i], weightChanges, biasChanges);
        }

        // Now, apply the step added in weightChanges and biasChanges:
        for (int i = 0; i < _net.getWeights().length; i++) {
            for (int j = 0; j < _net.getWeights()[i].length; j++) {
                for (int k = 0; k < _net.getWeights()[i][j].length; k++) {
//                    System.out.println("[" + i + "][" + j + "][" + k + "] " + weightChanges[i][j][k]);
                    _net.getWeights()[i][j][k] += weightChanges[i][j][k];

                    if (_net.getWeights()[i][j][k] > 100.0) {
                        _net.getWeights()[i][j][k] = 100.0;
                    } else if (_net.getWeights()[i][j][k] < -100.0) {
                        _net.getWeights()[i][j][k] = -100.0;
                    }

                }
            }
        }

//        System.out.println();
        for (int i = 0; i < _net.getBiases().length; i++) {
            for (int j = 0; j < _net.getBiases()[i].length; j++) {
//                System.out.println("[" + i + "][" + j + "] " + biasChanges[i][j]);
                _net.getBiases()[i][j] += biasChanges[i][j];
                if (_net.getBiases()[i][j] > 100.0) {
                    _net.getBiases()[i][j] = 100.0;
                } else if (_net.getBiases()[i][j] < -100.0) {
                    _net.getBiases()[i][j] = -100.0;
                }
            }
        }
//        System.out.println("========");
    }

    private void backPropagate(double[] input, double[] target, double[][][] weightChanges, double[][] biasChanges) {
        // This method finds and applies the negative gradient of a single training example.
        // The results of this single example are then added to weightChanges and biasChanges, where they
        double[] output = _net.execute(input);

        // Create "offsets", which stores the values that we would LIKE to add to the result through
        // changes to the net.
        double[] offsets = new double[output.length];
        for (int i = 0; i < output.length; i++) {
            offsets[i] = target[i] - output[i];
        }


        // Now, back-propagate the values up, modifying as we go.
        for (int i = _net.getLayers().length - 2; i >= 0; i--) {
            // i is the layer that we are trying to get the required offsets for.
            // as we get those connections, the weights and biases will be updated.
            // Updating the weights and biases simply means changing those values a tiny bit,
            // not actually replacing it.

            // Update biases:
//            System.out.println("i = " + i + ", biasChanges[i].length = " + biasChanges[i].length);
            for (int j = 0; j < biasChanges[i].length; j++) {
                biasChanges[i][j] = offsets[j] * DESCENT_RATE_BIASES;
//                System.out.println("biasChanges[" + (i) + "][" + j + "] = " + offsets[j]);
            }

            // Update weights:
            for (int j = 0; j < _net.getWeights()[i].length; j++) {
                    // This runs for each item in the i+1th column of layers, aka the ith column of weights.
                    for (int k = 0; k < _net.getWeights()[i][j].length; k++) {
                        // This runs for each individual connection, updating it to descend slightly.
                        weightChanges[i][j][k] += _net.getLayers()[i][k] * offsets[j] * DESCENT_RATE_WEIGHTS;
                    }
            }

            // Now, propagate upwards, and replace offsets with the next layer.
            double[] newOffsets = new double[_net.getLayers()[i].length];
            for (int j = 0; j < newOffsets.length; j++) {
                double jthOffset = 0.0;
                for (int k = 0; k < offsets.length; k++) {
                    jthOffset += offsets[k] * _net.getWeights()[i][k][j];
                }
                jthOffset *= PROPAGATION_WEIGHT;
                jthOffset /= offsets.length;
                newOffsets[j] = jthOffset;
            }
            offsets = newOffsets;
        }

    }

    private double getAveragePoints(double[][] inputs, double[][] targets, NeuralNet net) {
        double total = 0;
        for (int i = 0; i < inputs.length; i++) {
            total += getPoints(inputs[i], targets[i], net);
        }
        return total / inputs.length;
    }

    private double getPoints(double[] input, double[] target, NeuralNet net) {
        // More points = worse neural net.
        double[] result = net.execute(input);
        double total = 0;
        for (int i = 0; i < result.length; i++) {
            total += (result[i] - target[i])*(result[i] - target[i]);
        }
        return total / input.length;
    }
}
