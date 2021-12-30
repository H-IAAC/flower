package flwr.android_client;

import android.content.Context;
import android.util.Log;


import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;

import flwr.android_client.ml.Harmodel;

public class HarModelWrapper {

    private Harmodel model;
    private static final String TAG = "HarModel";
    private final Context context;
    private String inputline;

    public HarModelWrapper(Context context) {
        this.context = context;
        try {
            Harmodel model = Harmodel.newInstance(context);
            this.model = model;
        } catch (IOException e) {
            Log.e(TAG,"Erro ao inicializar o modelo", e);
            // TODO Handle the exception
        }
    }
    public void loadData(){

        BufferedReader reader = null;
        try {
            reader = new BufferedReader(
                    new InputStreamReader(this.context.getAssets().open("data/test.csv")));

            // do reading, usually loop until end of file reading
            String mLine;
            while ((mLine = reader.readLine()) != null) {
                //process line
                //Log.d(TAG,mLine);
                this.inputline = mLine;
            }
        } catch (IOException e) {
            Log.e(TAG,"erro ao carregar csv", e);
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e) {
                    //log the exception
                }
            }
        }
        this.fit();
    }
    public void fit(){
        String[] csv_separated = this.inputline.split(",");
        //Log.e(TAG,csv_separated[0]);

        float[] numbers = new float[csv_separated.length];
        for (int i = 0; i < csv_separated.length; ++i) {
            //TODO Handle invalid values in csv
            try{
                float csv_value = Float.parseFloat(csv_separated[i]);
                numbers[i] = csv_value;
            }
            catch(NumberFormatException e){
                numbers[i] = 0;
            }

        }

        ByteBuffer.allocate(4).putFloat(numbers[0]).array();
        byte[] byteArray= FloatArray2ByteArray(numbers);
        ByteBuffer byteBuffer = ByteBuffer.wrap(byteArray);

        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 561}, DataType.FLOAT32);
        int[] shape = inputFeature0.getShape();
        Log.d(TAG,"size of bytebuffer" + byteBuffer.toString());
        Log.d(TAG,"size of Tensor Buffer: "+ shape[1]);
        inputFeature0.loadBuffer(byteBuffer);

        // Runs model inference and gets result.
        Harmodel.Outputs outputs = this.model.process(inputFeature0);
        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
        //String converted = new String(buffer.array(), "UTF-8");

        // Releases model resources if no longer used.
        //model.close();
    }
    public byte[] FloatArray2ByteArray(float[] values){
        ByteBuffer buffer = ByteBuffer.allocate(4000 * values.length );

        for (float value : values){
            buffer.putFloat(value);
        }

        return buffer.array();
    }
}
