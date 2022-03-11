package flwr.android_client;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.ConditionVariable;
import android.util.Log;
import android.util.Pair;

import androidx.lifecycle.MutableLiveData;

import org.apache.commons.lang3.ObjectUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class FlowerClient {

    private TransferLearningModelWrapper tlModel;
    private static final int LOWER_BYTE_MASK = 0xFF;
    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
    private Context context;
    private final ConditionVariable isTraining = new ConditionVariable();
    private static String TAG = "Flower";
    private int local_epochs = 1;

    private String trainInputFile = "data/X_train.csv";
    private String testInputFile = "data/X_test.csv";
    private String trainLabelFile = "data/label_train.csv";
    private String testLabelFile = "data/label_test.csv";

    public static final int WINDOW_SIZE = 400;
    public static final int N_SAMPLES_TRAIN = 1000;
    public static final int N_SAMPLES_TEST = 1000;
    public static final int WINDOWED_N_SAMPLES_TRAIN = WINDOW_SIZE * N_SAMPLES_TRAIN;
    public static final int WINDOWED_N_SAMPLES_TEST = WINDOW_SIZE * N_SAMPLES_TEST;

    public FlowerClient(Context context) {
        this.tlModel = new TransferLearningModelWrapper(context);
        this.context = context;
    }

    public ByteBuffer[] getWeights() {
        return tlModel.getParameters();
    }

    public Pair<ByteBuffer[], Integer> fit(ByteBuffer[] weights, int epochs) {

        this.local_epochs = epochs;
        tlModel.updateParameters(weights);
        isTraining.close();
        tlModel.train(this.local_epochs);
        tlModel.enableTraining((epoch, loss) -> setLastLoss(epoch, loss));
        Log.e(TAG ,  "Training enabled. Local Epochs = " + this.local_epochs);
        isTraining.block();
        return Pair.create(getWeights(), tlModel.getSize_Training());
    }

    public Pair<Pair<Float, Float>, Integer> evaluate(ByteBuffer[] weights) {
        tlModel.updateParameters(weights);
        tlModel.disableTraining();
        return Pair.create(tlModel.calculateTestStatistics(), tlModel.getSize_Testing());
    }

    public void setLastLoss(int epoch, float newLoss) {
        if (epoch == this.local_epochs - 1) {
            Log.e(TAG, "Training finished after epoch = " + epoch);
            lastLoss.postValue(newLoss);
            tlModel.disableTraining();
            isTraining.open();
        }
    }

    private float[] toFloatArray(List<Float> list)
    {
        int i = 0;
        float[] array = new float[list.size()];

        for (Float f : list) {
            array[i++] = (f != null ? f : Float.NaN);
        }
        return array;
    }

    private void csvReadInput(String csvInputFilePath, String csvLabelFilePath, int n_samples, int win_sz, Boolean isTrain) throws IOException {
        BufferedReader readerInput = new BufferedReader(new InputStreamReader(this.context.getAssets().open(csvInputFilePath)));
        BufferedReader readerLabel = new BufferedReader(new InputStreamReader(this.context.getAssets().open(csvLabelFilePath)));

        String nextLine;
        // read the first line (column labels)
        String[] csvTitles = readerInput.readLine().split(",");

        ArrayList<Float> csvDataList = new ArrayList<>();

        int count = 0;
        float[] val = null;

        while ((nextLine = readerInput.readLine()) != null && count <= n_samples) {

            String[] curr_line = nextLine.split(",");

            if (val == null) {
                int lin_sz = curr_line.length;
                val = new float[lin_sz*n_samples];
            }

            for (String word: curr_line) {
                val[count] = Float.parseFloat(word);
            }

            /*
             * read the label for each window
             */
            if (0 == (count % win_sz)) {
                Log.e(TAG,  (count/win_sz)+ "th test window loaded");
                addSample(val, readerLabel.readLine(), isTrain);
            }

            count++;
        }

        readerInput.close();
        readerLabel.close();
    }

    public void loadData(int device_id) {
        try {
            // load train data
            csvReadInput(trainInputFile, trainLabelFile, WINDOWED_N_SAMPLES_TRAIN, WINDOW_SIZE, true);
            // load test data
            csvReadInput(testInputFile, testLabelFile, WINDOWED_N_SAMPLES_TRAIN, WINDOW_SIZE, false);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    private void addSample(float[] window, String sampleClass, Boolean isTraining) throws IOException {
        // add to the list.
        try {
            this.tlModel.addSample(window, sampleClass, isTraining).get();
        } catch (ExecutionException e) {
            throw new RuntimeException("Failed to add sample to model", e.getCause());
        } catch (InterruptedException e) {
            // no-op
        }
    }
}
