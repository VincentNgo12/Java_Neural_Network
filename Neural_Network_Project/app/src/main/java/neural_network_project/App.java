/*
 * This Java source file was generated by the Gradle 'init' task.
 */
package neural_network_project;

import neural_network_project.Network;
import neural_network_project.Network1;
import neural_network_project.Layers.FullyConnectedLayer;
import neural_network_project.ActivationFuncs.Sigmoid;
import neural_network_project.ActivationFuncs.Tanh;
import neural_network_project.MnistCSVLoader;
import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

public class App {

    public static void main(String[] args) {
        String trainingFile = "/workspaces/Java_Neural_Network/Neural_Network_Project/resources/MNIST/mnist_train.csv";
        String testFile = "/workspaces/Java_Neural_Network/Neural_Network_Project/resources/MNIST/mnist_test.csv";

        List<List<INDArray>> training_datas = MnistCSVLoader.LoadData(trainingFile);
        List<List<INDArray>> testing_datas = MnistCSVLoader.LoadData(testFile);


        // Network1 net = new Network1(new int[]{784 , 30, 10});
        // net.stochasticGradientDescent(training_datas, 5, 10, 3.0f, testing_datas);
        Network net = new Network(new FullyConnectedLayer(784, 30),
                                    new Tanh(),
                                    new FullyConnectedLayer(30, 10),
                                    new Tanh());

        net.stochastic_gradient_descent(training_datas, 30, 10, 0.1f, testing_datas);
    }
}
