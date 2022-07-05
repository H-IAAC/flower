package flwr.android_client;

import java.util.ArrayList;
import java.util.List;

public class ValueDataset {
    float line;
    String nameClass;
    String idClass;
    int quantity;
    List<Float> floatList = new ArrayList<>();

    public ValueDataset(float line, String idClass, String nameClass, List<Float> floatList) {
        this.line = line;
        this.nameClass = nameClass;
        this.floatList=floatList;
        this.idClass =idClass;
    }
}

