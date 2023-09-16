package neural_network_project.Layers;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Layer{

    public abstract INDArray forward(INDArray input);
    public abstract INDArray backward(INDArray output_gradient);
    public abstract boolean is_trainable();
    // Methods for tranable layers
    public void update_mini_batch(INDArray weights_gradient, INDArray biases_gradient, float learning_rate, int mini_batch_size){
        throw new UnsupportedOperationException("Layer not trainable.");
    }
    public INDArray get_weights(){
        throw new UnsupportedOperationException("Layer not trainable.");
    }
    public INDArray get_biases(){
        throw new UnsupportedOperationException("Layer not trainable.");
    }
    public INDArray get_biases_gradients(){
        throw new UnsupportedOperationException("Layer not trainable.");
    }
    public INDArray get_weights_gradients(){
        throw new UnsupportedOperationException("Layer not trainable.");
    }
}
