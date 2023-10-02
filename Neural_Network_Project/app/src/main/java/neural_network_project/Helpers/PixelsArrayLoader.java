package neural_network_project.Helpers;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.List;


public class PixelsArrayLoader{
    public static INDArray getData(int[] pixels){
        // Create an INDArray from the pixel values
        INDArray pixels_values = Nd4j.create(pixels).reshape(pixels.length, 1).divi(255.0f);
        
        return pixels_values;
    }
}