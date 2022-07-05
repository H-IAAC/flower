package flwr.android_client;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.os.ConditionVariable;
import android.util.Log;
import android.util.Pair;

import androidx.lifecycle.MutableLiveData;


import org.apache.commons.lang3.ArrayUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.Arrays;
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
        Log.e(TAG, "Training enabled. Local Epochs = " + this.local_epochs);
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
/*
    public void loadData(int device_id) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/partition_" + (device_id - 1) + "_train.txt")));
            String line;
            int i = 0;
            while ((line = reader.readLine()) != null) {
                i++;
                Log.e(TAG, i + "th training image loaded");
                addSample("data/" + line, true);
            }
            reader.close();

            i = 0;
            reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/partition_" +  (device_id - 1)  + "_test.txt")));
            while ((line = reader.readLine()) != null) {
                i++;
                Log.e(TAG, i + "th test image loaded");
                addSample("data/" + line, false);
            }
            reader.close();

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }*/


    public void loadDataExtrasensory(int device_id, String experimentid) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/extrasensory/fold_" + experimentid + "/extrasensory_partition_" + (device_id - 1) + ".txt")));
            String line;
            int i = 0;
            while ((line = reader.readLine()) != null) {
                i++;


                loadDataExtrasensoryByClient("data/extrasensory" +  "/" + line, true);

                Log.d("TOTAL AMOSTRAS", i + " training vector loaded "+String.valueOf(this.tlModel.getSize_Training()));
            }
            reader.close();

            i = 0;
            reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/extrasensory/fold_" + experimentid + "/extrasensory_partition_" + (device_id - 1) + ".txt")));
            while ((line = reader.readLine()) != null) {
                i++;
                loadDataExtrasensoryByClient("data/extrasensory"  + "/" + line, false);

                //    Log.d(TAG, i + " test vector loaded");
                Log.d("TOTAL AMOSTRAS", i + " TEST vector loaded "+String.valueOf(this.tlModel.getSize_Testing()));
            }
            reader.close();

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
 /*   private void loadDataExtrasensoryByClient(String file, Boolean isTraining) throws IOException {

        ExtrasensoryDataset extrasensoryDataset = new ExtrasensoryDataset(this.context, file, isTraining);
        List<List<Float>> inputXValues = extrasensoryDataset.getAllXFloatArray();
        List<List<String>> inputYValues = extrasensoryDataset.getAllYStringArray();

        for (int y = 0; y < inputXValues.size(); y++) {
            float[] floatArray1 = ArrayUtils.toPrimitive(inputXValues.get(y).subList(1, inputXValues.get(y).size()).toArray(new Float[0]), 0.0F);

            addSample_extrasensory(floatArray1, "Lying down", isTraining);
            Log.d(TAG, "AMOSTRA  "+String.valueOf(inputXValues.get(y).get(0)) +" output:"+ String.valueOf(inputYValues.get(y).get(0)));
        }

        Log.d(TAG, "Total Amostras: "+String.valueOf(inputXValues.size()) +" train:"+ isTraining );
    }


    private void addSample_extrasensory( float[]  floatList, String sampleClass, Boolean isTraining) throws IOException {

        // get rgb equivalent and class


        // add to the list.
        try {
            this.tlModel.addSample(floatList, sampleClass, isTraining).get();
        } catch (ExecutionException e) {
            throw new RuntimeException("Failed to add sample to model", e.getCause());
        } catch (InterruptedException e) {
            // no-op
        }

    }*/
    private void loadDataExtrasensoryByClient(String file, Boolean isTraining) throws IOException {
        List<String> labels=  Arrays.asList( "label:LYING_DOWN","label:SITTING","label:FIX_walking","label:FIX_running","label:BICYCLING","label:SLEEPING");

        ExtrasensoryDataset extrasensoryDataset = new ExtrasensoryDataset(this.context, file, isTraining,labels);
        List<ValueDataset> inputXValues = extrasensoryDataset.getListAllXY();


        for (String label :labels) {
            List<ValueDataset> valuesCat = extrasensoryDataset.getDataByCategory(label);
            for (ValueDataset value :valuesCat){

            float[] floatArray1 = ArrayUtils.toPrimitive(value.floatList.toArray(new Float[0]), 0.0F);

            //  addSample_extrasensory(floatArray1, "Lying down", isTraining);
            Bitmap  bitmap= getarray(floatArray1);
            float[] rgbImage = prepareImage(bitmap);
            try {
                this.tlModel.addSample(rgbImage, value.idClass, isTraining).get();
            } catch (ExecutionException e) {
                throw new RuntimeException("Failed to add sample to model", e.getCause());
            } catch (InterruptedException e) {
                // no-op
            }}
            // Log.d(TAG, "AMOSTRA  "+String.valueOf(inputXValues.get(y).get(0)) +" output:"+ String.valueOf(inputYValues.get(y).get(0)));
        }

        //   Log.d(TAG, "Total Amostras: "+String.valueOf(inputXValues.size()) +" train:"+ isTraining );
    }


    private void addSample_extrasensory( float[]  floatList, String sampleClass, Boolean isTraining) throws IOException {

        // get rgb equivalent and class


        // add to the list.
        try {
            this.tlModel.addSample(floatList, sampleClass, isTraining).get();
        } catch (ExecutionException e) {
            throw new RuntimeException("Failed to add sample to model", e.getCause());
        } catch (InterruptedException e) {
            // no-op
        }

    }

   /* private void addSample(String photoPath, Boolean isTraining) throws IOException {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap =  BitmapFactory.decodeStream(this.context.getAssets().open(photoPath), null, options);
        String sampleClass = get_class(photoPath);

        // get rgb equivalent and class
        float[] rgbImage = prepareImage(bitmap);

        // add to the list.
        try {
            this.tlModel.addSample(rgbImage, sampleClass, isTraining).get();
        } catch (ExecutionException e) {
            throw new RuntimeException("Failed to add sample to model", e.getCause());
        } catch (InterruptedException e) {
            // no-op
        }
    }*/

    private Bitmap getarray(float[] rows) throws IOException {

        int w = 10;
        int h= 10;
        int[]rawData = new int[w * h];
int count=0;
        for (int y = 0; y < w; y++) {
            for (int x = 0; x < h; x++) {
                float hue = rows[count]* 180;
                float hsv[]=new float[3];
                hsv[0]=hue;
                hsv[1]=1f;
                hsv[2]=1f;

                int color = Color.HSVToColor(hsv);
                rawData[y + x * w]  = color;
                count=count+1;
            }

        }
        Log.d("d", String.valueOf(rawData.length));
        Bitmap bitmap = Bitmap.createBitmap(w, h, Bitmap.Config.RGB_565);

        bitmap.setPixels(rawData, 0, w, 0, 0, w, h);
        return bitmap;
    }



    public String get_class(String path) {
        String label = path.split("/")[2];
        return label;
    }

    /**
     * Normalizes a camera image to [0; 1], cropping it
     * to size expected by the model and adjusting for camera rotation.
     */
    private static float[] prepareImage(Bitmap bitmap)  {
        int modelImageSize = TransferLearningModelWrapper.IMAGE_SIZE;
        Bitmap paddedBitmap = padToSquare(bitmap);
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(
                paddedBitmap, modelImageSize, modelImageSize, true);
        float[] normalizedRgb = new float[modelImageSize * modelImageSize * 3];
        int nextIdx = 0;
        for (int y = 0; y < modelImageSize; y++) {
            for (int x = 0; x < modelImageSize; x++) {
                int rgb = scaledBitmap.getPixel(x, y);
                float r = ((rgb >> 16) & LOWER_BYTE_MASK) * (1 / 255.0f);
                float g = ((rgb >> 8) & LOWER_BYTE_MASK) * (1 / 255.0f);
                float b = (rgb & LOWER_BYTE_MASK) * (1 / 255.0f);
                normalizedRgb[nextIdx++] = r;
                normalizedRgb[nextIdx++] = g;
                normalizedRgb[nextIdx++] = b;
            }
        }

        return normalizedRgb;
    }

    private static Bitmap padToSquare(Bitmap source) {
        int width = source.getWidth();
        int height = source.getHeight();

        int paddingX = width < height ? (height - width) / 2 : 0;
        int paddingY = height < width ? (width - height) / 2 : 0;
        Bitmap paddedBitmap = Bitmap.createBitmap(
                width + 2 * paddingX, height + 2 * paddingY, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(paddedBitmap);
        canvas.drawARGB(0xFF, 0xFF, 0xFF, 0xFF);
        canvas.drawBitmap(source, paddingX, paddingY, null);
        return paddedBitmap;
    }

}