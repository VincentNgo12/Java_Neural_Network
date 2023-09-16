package neural_network_project.Layers;

import java.util.ArrayList;
import java.util.List;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

public abstract class Layer{

    public abstract INDArray forward(INDArray input);
    public abstract INDArray backward(INDArray output_gradient);
}
