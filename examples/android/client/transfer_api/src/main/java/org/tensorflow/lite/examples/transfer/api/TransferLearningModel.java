/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.transfer.api;

import android.nfc.Tag;
import android.util.Log;
import android.util.Pair;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.GatheringByteChannel;
import java.nio.channels.ScatteringByteChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Represents a "partially" trainable model that is based on some other,
 * base model.
 */
public final class TransferLearningModel implements Closeable {

  /**
   * Prediction for a single class produced by the model.
   */
  public static class Prediction {
    private final float[] label;
    private final float confidence;

    public Prediction(float[] label, float confidence) {
      this.label = label;
      this.confidence = confidence;
    }

    public float[] getLabel() {
      return label;
    }

    public float getConfidence() {
      return confidence;
    }
  }

  private static class TrainingSample {
    float[] bottleneck;
    float[] label;

    TrainingSample(float[] bottleneck, float[] label) {
      this.bottleneck = bottleneck;
      this.label = label;
    }
  }


  private static class TestingSample {
    float[][][] image;
    float[] label;

    TestingSample(float[][][] image, float[] label) {
      this.image = image;
      this.label = label;
    }
  }

  /**
   * Consumer interface for training loss.
   */
  public interface LossConsumer {
    void onLoss(int epoch, float loss);
  }

  private static final int FLOAT_BYTES = 4;

  // Setting this to a higher value allows to calculate bottlenecks for more samples while
  // adding them to the bottleneck collection is blocked by an active training thread.
  private static final int NUM_THREADS =
      Math.max(1, Runtime.getRuntime().availableProcessors() - 1);

  private final int[] bottleneckShape;

  private final Map<String, Integer> classes;
  private final String[] classesByIdx;
  private final Map<String, float[]> oneHotEncodedClass;

  private LiteMultipleSignatureModel model;

  private final List<TrainingSample> trainingSamples = new ArrayList<>();
  private final List<TestingSample> testingSamples = new ArrayList<>();

  private ByteBuffer[] modelParameters;

  // Where to store the optimizer outputs.
  private ByteBuffer[] nextModelParameters;

  private ByteBuffer[] optimizerState;

  // Where to store the updated optimizer state.
  private ByteBuffer[] nextOptimizerState;

  // Where to store training inputs.
  private float[][] trainingBatchBottlenecks;
  private float[][] trainingBatchLabels;

  // A zero-filled buffer of the same size as `trainingBatchClasses`.
  private float[][] zeroBatchClasses;

  // Where to store calculated gradients.
  private float[][][] modelGradients;

  // Where to store bottlenecks produced during inference.
  private float[][] inferenceBottleneck;

  // Used to spawn background threads.
  private final ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);

  // This lock guarantees that only one thread is performing training at any point in time.
  // It also protects the sample collection from being modified while in use by a training
  // thread.
  private final Lock trainingLock = new ReentrantLock();

  // This lock guarantees that only one thread is performing training and inference at
  // any point in time. It also protects the sample collection from being modified while
  // in use by a training thread.
  private final Lock trainingInferenceLock = new ReentrantLock();

  // This lock guards access to trainable parameters.
  private final ReadWriteLock parameterLock = new ReentrantReadWriteLock();

  // This lock allows [close] method to assure that no threads are performing inference.
  private final Lock inferenceLock = new ReentrantLock();

  // Set to true when [close] has been called.
  private volatile boolean isTerminating = false;

  public TransferLearningModel(ModelLoader modelLoader, Collection<String> classes) {
    classesByIdx = classes.toArray(new String[0]);
    this.classes = new TreeMap<>();
    oneHotEncodedClass = new HashMap<>();
    for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
      String className = classesByIdx[classIdx];
      this.classes.put(className, classIdx);
      oneHotEncodedClass.put(className, oneHotEncoding(classIdx));
    }

    try {
      this.model =
          new LiteMultipleSignatureModel(
              modelLoader.loadMappedFile("model.tflite"), classes.size());
    } catch (IOException e) {
      throw new RuntimeException("Couldn't read underlying model for TransferLearningModel", e);
    }

    this.bottleneckShape = this.model.getBottleneckShape();
  }


  public int[] getBottleneckShape(){
    return this.bottleneckShape;
  }

  /**
   * Adds a new sample for training/testing.
   *
   * Sample bottleneck is generated in a background thread, which resolves the returned Future
   * when the bottleneck is added to training/testing samples.
   *
   * @param image image RGB data.
   * @param className ground truth label for image.
   */
  public Future<Void> addSample(float[][][] image, String className, Boolean isTraining) {
    checkNotTerminating();

    if (!classes.containsKey(className)) {
      throw new IllegalArgumentException(String.format(
          "Class \"%s\" is not one of the classes recognized by the model", className));
    }

    return executor.submit(
        () -> {
          if (Thread.interrupted()) {
            return null;
          }

          trainingInferenceLock.lockInterruptibly();
          try {
            float[] bottleneck = model.loadBottleneck(image);
            if (isTraining)
               trainingSamples.add(new TrainingSample(bottleneck, oneHotEncodedClass.get(className)));
            else
               testingSamples.add(new TestingSample(image, oneHotEncodedClass.get(className)));
          } finally {
            trainingInferenceLock.unlock();
          }

          return null;
        });
  }

  /**
   * Trains the model on the previously added data samples.
   *
   * @param numEpochs number of epochs to train for.
   * @param lossConsumer callback to receive loss values, may be null.
   * @return future that is resolved when training is finished.
   */
  public Future<Void> train(int numEpochs, LossConsumer lossConsumer) {
    checkNotTerminating();
    int trainBatchSize = getTrainBatchSize();

    if (trainingSamples.size() < trainBatchSize) {
      throw new RuntimeException(
          String.format(
              "Too few samples to start training: need %d, got %d",
              trainBatchSize, trainingSamples.size()));
    }

    trainingBatchBottlenecks = new float[trainBatchSize][numBottleneckFeatures()];
    trainingBatchLabels = new float[trainBatchSize][this.classes.size()];

    return executor.submit(
        () -> {
          trainingInferenceLock.lock();
          try {
            epochLoop:
            for (int epoch = 0; epoch < numEpochs; epoch++) {
              float totalLoss = 0;
              int numBatchesProcessed = 0;

              for (List<TrainingSample> batch : trainingBatches(trainBatchSize)) {
                if (Thread.interrupted()) {
                  break epochLoop;
                }

                for (int sampleIdx = 0; sampleIdx < batch.size(); sampleIdx++) {
                  TrainingSample sample = batch.get(sampleIdx);
                  trainingBatchBottlenecks[sampleIdx] = sample.bottleneck;
                  trainingBatchLabels[sampleIdx] = sample.label;
                }

                float loss = this.model.runTraining(trainingBatchBottlenecks, trainingBatchLabels);
                totalLoss += loss;
                numBatchesProcessed++;

                ByteBuffer[] swapBufferArray;

                // Swap optimizer state with its next version.
                swapBufferArray = optimizerState;
                optimizerState = nextOptimizerState;
                nextOptimizerState = swapBufferArray;

                // Swap model parameters with their next versions.
                parameterLock.writeLock().lock();
                try {
                  swapBufferArray = modelParameters;
                  modelParameters = nextModelParameters;
                  nextModelParameters = swapBufferArray;
                } finally {
                  parameterLock.writeLock().unlock();
                }
              }
              float avgLoss = totalLoss / numBatchesProcessed;
              Log.e("Avg Loss", avgLoss +"");
              if (lossConsumer != null) {
                lossConsumer.onLoss(epoch, avgLoss);
              }
            }

            return null;
          } finally {
            trainingInferenceLock.unlock();
          }
        });
  }

  public Boolean trainingInProgress(){

    if (trainingLock.tryLock()){
      trainingLock.unlock();
      return false;
    }
    else {
      return true;
    }
  }


  public Pair<Float, Float> getTestStatistics() {
    float[] confidences;
    Prediction[] predictions = new Prediction[classes.size()];
    parameterLock.readLock().lock();
    float loss = 0.0f;
    int correct = 0;
    try {
      for (int sampleIdx = 0; sampleIdx < testingSamples.size(); sampleIdx++) {
        TestingSample sample = testingSamples.get(sampleIdx);
        confidences = this.model.runInference(sample.image);

        for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
          predictions[classIdx] = new Prediction(oneHotEncoding(classIdx), confidences[classIdx]);
        }
        Arrays.sort(predictions, (a, b) -> -Float.compare(a.confidence, b.confidence));
        if (predictions[0].getLabel() == sample.label) correct++;
        loss += getLLLoss(predictions, sample.label);
      }
    } finally {
      parameterLock.readLock().unlock();
    }

    Log.e("Accuracy", (float) correct/testingSamples.size() + "--" + loss / testingSamples.size() );
    return Pair.create(loss/testingSamples.size(), (float) correct /testingSamples.size());
  }

  private float getLLLoss(Prediction[] predictions, float[] gt){
    for (int i = 0; i < predictions.length; i++){
      if (predictions[i].label == gt){
        return (float) (-1.0 * Math.log(predictions[i].confidence));
      }
    }
    return 0.0f;
  }

  /**
   * Runs model inference on a given image.
   *
   * @param image image RGB data.
   * @return predictions sorted by confidence decreasing. Can be null if model is terminating.
   */
  public Prediction[] predict(float[][][] image) {
    checkNotTerminating();
    trainingInferenceLock.lock();

    try {
      if (isTerminating) {
        return null;
      }

      float[] confidences;
      parameterLock.readLock().lock();
      try {
        confidences = this.model.runInference(image);
      } finally {
        parameterLock.readLock().unlock();
      }

      Prediction[] predictions = new Prediction[classes.size()];
      for (int classIdx = 0; classIdx < classes.size(); classIdx++) {
        predictions[classIdx] = new Prediction(oneHotEncoding(classIdx), confidences[classIdx]);
      }

      Arrays.sort(predictions, (a, b) -> -Float.compare(a.confidence, b.confidence));
      return predictions;
    } finally {
      trainingInferenceLock.unlock();
    }
  }

  private float[] oneHotEncoding(int classIdx) {
    float[] oneHot = new float[4];
    oneHot[classIdx] = 1;
    return oneHot;
  }

  /**
   * Writes the current values of the model parameters to a writable channel.
   *
   * The written values can be restored later using {@link #loadParameters(ScatteringByteChannel)},
   * under condition that the same underlying model is used.
   *
   * @param outputChannel where to write the parameters.
   * @throws IOException if an I/O error occurs.
   */
  public void saveParameters(GatheringByteChannel outputChannel) throws IOException {
    parameterLock.readLock().lock();
    try {
      outputChannel.write(modelParameters);
      for (ByteBuffer buffer : modelParameters) {
        buffer.rewind();
      }
    } finally {
      parameterLock.readLock().unlock();
    }
  }


  public ByteBuffer[] getParameters()  {
      return modelParameters;
  }

  /**
   * Overwrites the current model parameter values with the values read from a channel.
   *
   * The channel should contain values previously written by
   * {@link #saveParameters(GatheringByteChannel)} for the same underlying model.
   *
   * @param inputChannel where to read the parameters from.
   * @throws IOException if an I/O error occurs.
   */
  public void loadParameters(ScatteringByteChannel inputChannel) throws IOException {
    parameterLock.writeLock().lock();
    try {
      inputChannel.read(modelParameters);
      for (ByteBuffer buffer : modelParameters) {
        buffer.rewind();
      }
    } finally {
      parameterLock.writeLock().unlock();
    }
  }

  /** Training model expected batch size. */
  public int getTrainBatchSize() {
    return Math.min(
        Math.max(/* at least one sample needed */ 1, trainingSamples.size()),
        model.getExpectedBatchSize());
  }

  /**
   * Constructs an iterator that iterates over training sample batches.
   *
   * @param trainBatchSize batch size for training.
   * @return iterator over batches.
   */
  private Iterable<List<TrainingSample>> trainingBatches(int trainBatchSize) {
    if (!trainingInferenceLock.tryLock()) {
      throw new RuntimeException("Thread calling trainingBatches() must hold the training lock");
    }
    trainingInferenceLock.unlock();

    Collections.shuffle(trainingSamples);
    return () ->
        new Iterator<List<TrainingSample>>() {
          private int nextIndex = 0;

          @Override
          public boolean hasNext() {
            return nextIndex < trainingSamples.size();
          }

          @Override
          public List<TrainingSample> next() {
            int fromIndex = nextIndex;
            int toIndex = nextIndex + trainBatchSize;
            nextIndex = toIndex;
            if (toIndex >= trainingSamples.size()) {
              // To keep batch size consistent, last batch may include some elements from the
              // next-to-last batch.
              return trainingSamples.subList(
                  trainingSamples.size() - trainBatchSize, trainingSamples.size());
            } else {
              return trainingSamples.subList(fromIndex, toIndex);
            }
          }
        };
  }

  private int numBottleneckFeatures() {
    return model.getNumBottleneckFeatures();
  }

  private void checkNotTerminating() {
    if (isTerminating) {
      throw new IllegalStateException("Cannot operate on terminating model");
    }
  }

  /**
   * Terminates all model operation safely. Will block until current inference request is finished
   * (if any).
   *
   * <p>Calling any other method on this object after [close] is not allowed.
   */
  @Override
  public void close() {
    isTerminating = true;
    executor.shutdownNow();

    // Make sure that all threads doing inference are finished.
    trainingInferenceLock.lock();

    try {
      boolean ok = executor.awaitTermination(5, TimeUnit.SECONDS);
      if (!ok) {
        throw new RuntimeException("Model thread pool failed to terminate");
      }

      this.model.close();
    } catch (InterruptedException e) {
      // no-op
    } finally {
      trainingInferenceLock.unlock();
    }
  }

  private static ByteBuffer allocateBuffer(int capacity) {
    ByteBuffer buffer = ByteBuffer.allocateDirect(capacity);
    buffer.order(ByteOrder.nativeOrder());
    return buffer;
  }

  public int getSize_Training() {
    return trainingSamples.size();
  }

  public int getSize_Testing() {
    return testingSamples.size();
  }


  public void updateParameters(ByteBuffer[] newParams){
    parameterLock.writeLock().lock();
    try {
      modelParameters = newParams;
    } finally {
      parameterLock.writeLock().unlock();
    }
  }
}
