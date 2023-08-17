/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package neural_network_project;

import neural_network_project.Network;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

public class App {

    public static void main(String[] args) {
        int[] size = {1,2};
        Network network = new Network(size);
        System.out.println(network.weights);
        INDArray a = Nd4j.create(new float[] {999});
        System.out.println(network.feedforward(a));
    }
}