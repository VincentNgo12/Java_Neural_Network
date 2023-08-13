package neural_network_project;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;

public class Helper{
    public static double sigmoid(double x){
        return 1 / (1 + Math.exp(-x));
    }


    @SuppressWarnings("unchecked")
    public static <T> List<List<T>> zip(List<T>... lists){
        List<List<T>> zipped = new ArrayList<List<T>>();

        for(List<T> list:lists){
            for(int i=0, listSize=list.size(); i<listSize; i++){
                List<T> currentList;
                if(i>=zipped.size()){
                    zipped.add(currentList = new ArrayList<T>());
                }else{
                    currentList = zipped.get(i);
                }

                currentList.add(list.get(i));
            }
        }

        return zipped;
    }
}