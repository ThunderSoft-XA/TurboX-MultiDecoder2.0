package com.thundercomm.eBox.AI;

import android.app.Application;
import android.graphics.Bitmap;
import android.media.Image;
import android.util.Pair;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.SNPE;
import com.qualcomm.qti.snpe.Tensor;
import com.thundercomm.eBox.Config.GlobalConfig;
import com.thundercomm.eBox.Constants.Constants;
import com.thundercomm.eBox.Jni;
import com.thundercomm.eBox.Model.RtspItemCollection;
import com.thundercomm.eBox.Utils.LogUtil;
import com.thundercomm.eBox.VIew.ObjectDetectFragment;
import com.thundercomm.eBox.VIew.PlayFragment;
import com.thundercomm.gateway.data.DeviceData;
import com.thundercomm.eBox.Data.ObjectData;

import org.jetbrains.annotations.NotNull;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

import java.util.Map;
import java.util.Set;
import java.util.Vector;

import lombok.SneakyThrows;

/**
 * Object Detector Task
 *
 * @Describe
 */
public class ObjectDectorTask {

    private static String TAG = "ObjectDectorTask";

    private HashMap<Integer, NeuralNetwork> mapObjectDetector = new HashMap<Integer, NeuralNetwork>();

    private HashMap<Integer, DataInputFrame> inputFrameMap = new HashMap<Integer, DataInputFrame>();
    private Vector<ObjectTaskThread> mObjectPerTaskThreads = new Vector<ObjectTaskThread>();

    private boolean istarting = false;
    private boolean isInit = false;
    private Application mContext;
    private ArrayList<PlayFragment> playFragments;

    private int frameWidth;
    private int frameHeight;

    private static volatile ObjectDectorTask _instance;

    private ObjectDectorTask() {
    }

    public static ObjectDectorTask getObjectInputTask() {
        if (_instance == null) {
            synchronized (ObjectDectorTask.class) {
                if (_instance == null) {
                    _instance = new ObjectDectorTask();
                }
            }
        }
        return _instance;
    }

    public void init( Application context, Vector<Integer> idlist, ArrayList<PlayFragment> playFragments, int width, int height) {
        frameWidth = width;
        frameHeight = height;
        interrupThread();
        for (int i = 0; i < idlist.size(); i++) {
            if (getObjectAlgorithmType(idlist.elementAt(i))) {
                DataInputFrame data = new DataInputFrame(idlist.elementAt(i));
                inputFrameMap.put(idlist.elementAt(i), data);
            }
        }
        mContext = context;
        istarting = true;
        isInit = true;
        this.playFragments = playFragments;
        for (int i = 0; i < idlist.size(); i++) {
            if (getObjectAlgorithmType(idlist.elementAt(i))) {
                ObjectTaskThread objTaskThread = new ObjectTaskThread(idlist.elementAt(i));
                objTaskThread.start();
                mObjectPerTaskThreads.add(objTaskThread);
            }
        }
    }

    private boolean getObjectAlgorithmType(int id) {
        DeviceData deviceData = RtspItemCollection.getInstance().getDeviceList().get(id);
        boolean enable_object = Boolean.parseBoolean(RtspItemCollection.getInstance().getAttributesValue(deviceData, Constants.ENABLE_OBJECT_STR));
        return enable_object;
    }

    public void addImgById(int id, final Image img) {
        if (!inputFrameMap.containsKey(id)) {
            return;
        }

        DataInputFrame data = inputFrameMap.get(id);
        data.addImgById(img);
    }

    public void addBitmapById(int id, final Bitmap bmp, int w, int h) {
        if (!inputFrameMap.containsKey(id)) {
            return;
        }

        DataInputFrame data = inputFrameMap.get(id);
        data.org_w = w;
        data.org_h = h;
        data.addBitMapById(bmp);
    }

    public void addMatById(int id, final Mat img, int w, int h) {
        if (!inputFrameMap.containsKey(id)) {
            return;
        }

        DataInputFrame data = inputFrameMap.get(id);
        data.org_w = w;
        data.org_h = h;
        data.addMatById(img);
    }

    class ObjectCombineTaskThread extends Thread {

    }

    class ObjectTaskThread extends Thread {
        private ObjectDetectFragment objectDectorTask = null;
        private NeuralNetwork network = null;
        Map<String, FloatTensor> outputs = null;
        Map<String, FloatTensor> inputs = null;
        int alg_camid = -1;

        ObjectTaskThread(int id) {
            alg_camid = id;
            if (!mapObjectDetector.containsKey(alg_camid)) {
                try {
                    final SNPE.NeuralNetworkBuilder builder = new SNPE.NeuralNetworkBuilder(mContext)
                            .setRuntimeOrder(NeuralNetwork.Runtime.GPU)
                            .setModel(new File( GlobalConfig.SAVE_PATH + "inception_v3.dlc"));
                    network = builder.build();
                } catch (IOException e) {
                    e.printStackTrace();
                }
                mapObjectDetector.put(alg_camid, network);
            } else {
                network = mapObjectDetector.get(alg_camid);
            }
        }

        @SneakyThrows
        @Override
        public void run() {
            super.run();
            Jni.Affinity.bindToCpu(alg_camid % 4 + 4);
            objectDectorTask = (ObjectDetectFragment) playFragments.get(alg_camid);
            DataInputFrame inputFrame = inputFrameMap.get(alg_camid);
            Mat rotateimage = new Mat(frameHeight, frameWidth, CvType.CV_8UC4);
            Mat resizeimage = new Mat(frameHeight, frameWidth, CvType.CV_8UC4);
            Mat frameBgrMat = new Mat(frameHeight, frameWidth, CvType.CV_8UC3);
            float[] matData = new float[frameHeight * frameWidth * 3];

            LogUtil.d("", "debug test start Object camid  " + alg_camid);

            Set<String> inputNames = network.getInputTensorsNames();
            Set<String> outputNames = network.getOutputTensorsNames();

            final String inputLayer = inputNames.iterator().next();
            final String outputLayer = outputNames.iterator().next();

            final FloatTensor tensor = network.createFloatTensor(
                    network.getInputTensorsShapes().get(inputLayer));

            while (istarting) {
                try {
                    inputFrame.updateFaceRectCache();
                    Mat mat = inputFrame.getMat();

                    if (!OPencvInit.isLoaderOpenCV() || mat == null) {
                        if (mat != null) mat.release();
                        try {
                            Thread.sleep(50);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                        continue;
                    }

                    Core.flip(mat, rotateimage, 0);
                    Imgproc.resize(rotateimage, resizeimage, new Size(299, 299));
                    Imgproc.cvtColor(resizeimage, frameBgrMat, Imgproc.COLOR_RGBA2RGB);
                    frameBgrMat.convertTo(frameBgrMat, CvType.CV_32FC3);

                    Mat input_norm = new Mat(299, 299, CvType.CV_32FC3);
                    Core.normalize(frameBgrMat, input_norm, -1.0f, 1.0f, Core.NORM_MINMAX);
                    input_norm.get(0,0, matData);
                    if (mat != null) mat.release();

                    tensor.write(matData, 0, matData.length);
                    inputs = new HashMap<>();
                    inputs.put(inputLayer, tensor);

                    ObjectData result = null;
                    outputs = network.execute(inputs);
                    for (Map.Entry<String, FloatTensor> output : outputs.entrySet()) {
                        if (output.getKey().equals(outputLayer)) {
                            FloatTensor outputTensor = output.getValue();
                            final float[] array = new float[outputTensor.getSize()];
                            outputTensor.read(array, 0, array.length);
                            for (Pair<Integer, Float> pair : topK(1, array)) {
                                if(pair.second > 0.7f) {
                                    result = new ObjectData(pair.first, pair.second);
                                }
                            }
                        }
                    }

                    if (null != result) {
                        Vector<ObjectData> objVec = getObjectData(inputFrame, result);
                        postObjectDetectResult(alg_camid, objVec);
                    } else {
                        postNoResult(alg_camid);
                    }
                } catch (final Exception e) {
                    e.printStackTrace();
                    LogUtil.e(TAG, "Exception!");
                }
            }
            releaseTensors(inputs, outputs);
        }
        private final void releaseTensors(Map<String, ? extends Tensor>... tensorMaps) {
            for (Map<String, ? extends Tensor> tensorMap: tensorMaps) {
                for (Tensor tensor: tensorMap.values()) {
                    tensor.release();
                }
            }
        }

        Pair<Integer, Float>[] topK(int k, final float[] tensor) {
            final boolean[] selected = new boolean[tensor.length];
            final Pair<Integer, Float> topK[] = new Pair[k];
            int count = 0;
            while (count < k) {
                final int index = top(tensor, selected);
                selected[index] = true;
                topK[count] = new Pair<>(index, tensor[index]);
                count++;
            }
            return topK;
        }

        private int top(final float[] array, boolean[] selected) {
            int index = 0;
            float max = -1.f;
            for (int i = 0; i < array.length; i++) {
                if (selected[i]) {
                    continue;
                }
                if (array[i] > max) {
                    max = array[i];
                    index = i;
                }
            }
            return index;
        }

        private void postNoResult(int id) {
            if (objectDectorTask != null) {
                objectDectorTask.OnClean();
            }
        }

        private void postObjectDetectResult(int id, Vector<ObjectData> mObjectVec) {
            if (objectDectorTask != null) {
                objectDectorTask.OndrawObject(mObjectVec);
            }
        }
    }

    @NotNull
    private Vector<ObjectData> getObjectData(DataInputFrame inputFrame, ObjectData result) {
        Vector<ObjectData> mObjectVec = new Vector<>();

        LogUtil.i(TAG, "check object ", result);

        result.rect[0] = 150;
        result.rect[1] = 100;
        result.rect[2] = inputFrame.org_w - 300;
        result.rect[3] = inputFrame.org_h - 200;
        mObjectVec.add(result);

        return mObjectVec;
    }

    public void closeService() {

        isInit = false;
        istarting = false;

        System.gc();
        System.gc();
    }

    private void interrupThread() {
        for (ObjectTaskThread objectPerTaskThread : this.mObjectPerTaskThreads) {
            if (objectPerTaskThread != null && !objectPerTaskThread.isInterrupted()) {
                objectPerTaskThread.interrupt();
            }
        }
        mapObjectDetector.clear();
    }

    public boolean isIstarting() {
        return isInit;
    }
}
