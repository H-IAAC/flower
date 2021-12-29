//package flwr.android_client;
//
//import android.content.Context;
//import android.os.ConditionVariable;
//import android.util.Log;
//import android.util.Pair;
//
//import androidx.lifecycle.MutableLiveData;
//
//import java.nio.ByteBuffer;
//
//public class FlowerHarClient {
//    private HarModelWrapper hModel;
//    private static final int LOWER_BYTE_MASK = 0xFF;
//    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
//    private Context context;
//    private final ConditionVariable isTraining = new ConditionVariable();
//    private static String TAG = "Flower";
//    private int local_epochs = 1;
//
//    public FlowerHarClient(Context context) {
//        this.hModel = new HarModelWrapper(context);
//        this.context = context;
//    }
//
//    public ByteBuffer[] getWeights() {
//        return hModel.getParameters();
//    }
//
//    public Pair<ByteBuffer[], Integer> fit(ByteBuffer[] weights, int epochs) {
//
//        this.local_epochs = epochs;
//        hModel.updateParameters(weights);
//        isTraining.close();
//        hModel.train(this.local_epochs);
//        hModel.enableTraining((epoch, loss) -> setLastLoss(epoch, loss));
//        Log.e(TAG ,  "Training enabled. Local Epochs = " + this.local_epochs);
//        isTraining.block();
//        return Pair.create(getWeights(), tlModel.getSize_Training());
//    }
//
//    public Pair<Pair<Float, Float>, Integer> evaluate(ByteBuffer[] weights) {
//        hModel.updateParameters(weights);
//        hModel.disableTraining();
//        return Pair.create(hModel.calculateTestStatistics(), hModel.getSize_Testing());
//    }
//
//
//
//}
