package neural_network_project;

import java.util.Collections;
import java.util.ArrayList;
import java.util.List;

import javax.sound.midi.SysexMessage;

import java.util.Arrays;
import neural_network_project.Layers.Layer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
// Java serialization 
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.Serializable;


public class Network implements Serializable{
    List<Layer> layers;

    // Initialize the object
    public Network(Layer... layers){
        this.layers = layers;
    }


    // This methods feed forward throught the layers of the network
    public INDArray(INDArray input){
        for(Layer layer : layers){
            input = layer.forward(input);
        }
        return input;
    }



    // Comput the mean-square cost derivative
    public INDArray cost_derivative(INDArray output_activations, INDArray desiredOutput){
        return output_activations.sub(desiredOutput);
    }


    // Evaluate the accuracy of the network on test datas
    public int evaluate(List<List<INDArray>> test_datas){
        int correct_predictions = 0;
        List<List<Integer>> test_results = new ArrayList<>();

        for(List<INDArray> test_data:test_datas){
            INDArray raw_input = test_data.get(0);
            INDArray desired_output = test_data.get(1);

            INDArray finale_output_layer = this.feedforward(raw_input);
            Integer class_index = Nd4j.argMax(finale_output_layer).getInt(0);
            Integer class_result = Nd4j.argMax(desired_output).getInt(0);

            List<Integer> test_result = new ArrayList<>();
            test_result.add(class_index);
            test_result.add(class_result);

            test_results.add(test_result);
        }

        for(List<Integer> result:test_results){
            if(result.get(0) == result.get(1)) correct_predictions++;
        }

        return correct_predictions;
    }
}