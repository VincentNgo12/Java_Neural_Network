package neural_network_project;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class MnistCSVLoader{
    public static List<List<INDArray>> LoadData(String filePath ){
        List<List<INDArray>> datas = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))){
            String line;

            while ((line = reader.readLine()) != null){
                // Split the line into array of String
                String[] values = line.split(",");

                // Skip the first column as it is the labeling
                float[] pixelsValue = new float[values.length - 1];
                for (int i = 1; i < values.length; i++) {
                    pixelsValue[i - 1] = Float.parseFloat(values[i]);
                }

                // Create an INDArray from the pixel values
                INDArray pixels = Nd4j.create(pixelsValue).reshape(pixelsValue.length, 1);

                // get the label of the current image
                int label = Integer.parseInt(values[0]);
                // Create a 10-dimensional zero vector
                INDArray result = Nd4j.zeros(10, 1);

                // Set current label element to 1.0 in the result vector
                result.putScalar(label, 1.0);

                List<INDArray> data = new ArrayList<>();
                data.add(pixels);
                data.add(result);

                datas.add(data);
            }
            
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Data Loaded Successfully");
        return datas;
    }
}