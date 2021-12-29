//package flwr.android_client;
//
//import android.content.Context;
//import android.os.ConditionVariable;
//
//import org.tensorflow.lite.examples.transfer.api.TransferLearningModel;
//
//public class HarModelWrapper extends HarModel implements Cloneable {
//
//    private final HarModel model;
//
//    private final ConditionVariable shouldTrain = new ConditionVariable();
//    private volatile TransferLearningModel.LossConsumer lossConsumer;
//    private Context context;
//
//    public HarModelWrapper(Context context) {
//        this.context = context;
//        this.model = new HarModel();
//    }
//
//}
