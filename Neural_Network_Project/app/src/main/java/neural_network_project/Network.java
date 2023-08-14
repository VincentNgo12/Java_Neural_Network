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
        for(int i = 0; i < this.biases.size(); i++){
            INDArray weightsMatrix = this.weights.get(i);
            INDArray biasesMatrix = this.biases.get(i);
            a = Nd4j.add(weightsMatrix.mmul(a),biasesMatrix);
            a = Transforms.sigmoid(a);
        }

        return a;
    }


    public List<List<INDArray>> backpropagation(INDArray input, INDArray desiredOutput){
        List<INDArray> gradient_biases = new ArrayList<>();
        List<INDArray> gradient_weights = new ArrayList<>();
        
        for(INDArray weight:this.weights){
            gradient_weights.add(Nd4j.zerosLike(weight));
        }
        for(INDArray bias:this.biases){
            gradient_weights.add(Nd4j.zerosLike(bias));
        }

        INDArray activation = input;
        List<INDArray> activations = new ArrayList<>();
        activations.add(input);
        List<INDArray> z_vectors = new ArrayList<>();

        for(int i=0; i<this.biases.size(); i++){
            INDArray weightsMatrix = this.weights.get(i);
            INDArray biasesMatrix = this.biases.get(i);
            INDArray z = Nd4j.add(weightsMatrix.mmul(activation),biasesMatrix);
            z_vectors.add(z);
            activation = Transforms.sigmoid(z);
            activations.add(activation);
        }

        INDArray last_activation_layer = activations.get(activations.size()-1);
        INDArray last_z_vector_layer = z_vectors.get(z_vectors.size()-1);
        // Backward pass
        INDArray delta_vector = this.cost_derivative(last_activation_layer, desiredOutput).mul(Transforms.sigmoidDerivative(last_z_vector_layer));
        gradient_biases.set(gradient_biases.size()-1,delta_vector);
        gradient_weights.set(gradient_weights.size()-1,delta_vector.mmul(activations.get(activations.size()-2).transpose()));


        for(int layer=this.num_layers-2; layer > 0; layer--){
            INDArray z = z_vectors.get(layer);
            INDArray sigmoid_prime = Transforms.sigmoidDerivative(z);
            delta_vector = this.weights.get(layer+1).transpose().mmul(delta_vector).mul(sigmoid_prime);

            gradient_biases.set(layer, delta_vector);
            gradient_weights.set(layer, delta_vector.mmul(activations.get(layer-1).transpose()));
        }

        List<List<INDArray>> gradients = new ArrayList<>();
        gradients.add(gradient_biases);
        gradients.add(gradient_weights);
        return gradients;
    }


    public INDArray cost_derivative(INDArray output_activations, INDArray desiredOutput){
        return Nd4j.sub(output_activations,desiredOutput);
    }
}