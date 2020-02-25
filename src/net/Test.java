package net;

public class Test {
    public static void main(String[] args) {
        double[][] inputs = new double[50][2];
        double[][] outputs = new double[inputs.length][1];
        for (int i = 0; i < outputs.length; i++) {
            inputs[i][0] = Math.random() > 0.5 ? 1.0 : 0.0;
            inputs[i][1] = Math.random() > 0.5 ? 1.0 : 0.0;
            outputs[i][0] = inputs[i][0] != inputs[i][1] ? 1.0 : 0.0;
            System.out.println(inputs[i][0] + " " + inputs[i][1] + " " + outputs[i][0]);
        }
        NeuralNet net = new NeuralNet(new int[] {2, 4, 4, 1});
        NeuralNetTrainer trainer = new NeuralNetTrainer(net, true);
        printVals(net);
        System.out.println();
        trainer.trainNetwork(inputs, outputs, 12);
        printVals(net);
    }

    private static void printVals(NeuralNet net) {
        System.out.println("0.0, 0.0 -> " + net.execute(new double[]{0.0, 0.0})[0]);
        System.out.println("0.5, 1.0 -> " + net.execute(new double[]{0.5, 1.0})[0]);
        System.out.println("1.0, 0.0 -> " + net.execute(new double[]{1.0, 0.0})[0]);
        System.out.println("1.0, 1.0 -> " + net.execute(new double[]{1.0, 1.0})[0]);
    }
}
