package neural_network_project;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import neural_network_project.Helper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;


public class Network{
    int num_layers;
    int[] networkSize;
    List<INDArray> weights = new ArrayList<>();
    List<INDArray> biases = new ArrayList<>();


    public Network(int[] size){
        this.num_layers = size.length;
        this.networkSize = size;
        for(int i=1;i < size.length;i++){
            // Number of neurons for current layer
            int x = size[i];
            INDArray bias = Nd4j.randn(x, 1);
            this.biases.add(bias);
        }

        for(int i=0; i < size.length -1; i++){
            // Number of neurons for current layer
            int x = size[i];
            // Number of neurons for next layer
            int y = size[i+1];
            INDArray weight = Nd4j.randn(y,x);
            this.weights.add(weight);
        }
    }


    public INDArray feedforward(INDArray a){
        for(int i = 0; i < this.weights.size(); i++){
            INDArray weightsMatrix = this.weights.get(i);
            INDArray biasesMatrix = this.biases.get(i);
            a = weightsMatrix.mmul(a).add(biasesMatrix);
            a = Transforms.sigmoid(a);
        }

        return a;
    }

}