package com.example.FittnessAppWithPuppy;

import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaPlayer;
import android.os.*;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;
import com.github.mikephil.charting.charts.PieChart;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.data.Entry;
import com.github.mikephil.charting.data.PieData;
import com.github.mikephil.charting.data.PieDataSet;
import com.github.mikephil.charting.formatter.PercentFormatter;
import com.github.mikephil.charting.utils.ColorTemplate;
import weka.classifiers.Classifier;
import weka.core.*;

import java.io.*;
import java.util.*;

public class MainActivity extends Activity implements SensorEventListener {
    private Sensor mAccelerometer;
    private SensorManager mSensorManager;
    private TextView showPredictionView;
    private long WINDOW_SIZE_IN_MILISECONDS = 2000;
    private long OVERLAP_IN_PERCENT = 50;
    private PowerManager.WakeLock wakeLock;
    private Timer mTimer;
    private ArrayList<float[]> eventArrayList;
    private Classifier classifier;
    private TextView printView;
    private ArrayList<Double> stateHistory;
    private ArrayList<Double[]> savedDistributions;
    private HashMap<String, Integer> stateCounter;
    private Button startButton;
    private boolean checkingForTap;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mAccelerometer = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        showPredictionView = (TextView) findViewById(R.id.textviewShowPrediction);
        printView = (TextView) findViewById(R.id.println);
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "pwl");
        eventArrayList = new ArrayList<>();
        stateHistory = new ArrayList<>();
        savedDistributions = new ArrayList<>();
        stateCounter = new HashMap<>();
        startButton = (Button) findViewById(R.id.buttonStart);
        checkingForTap = false;
        //TODO: this takes a long time. Loading screen?
        try {
            InputStream classifierStream = getAssets().open("rfWindowv2.model");
            classifier = (Classifier) weka.core.SerializationHelper.read(classifierStream);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void startPrediction(View v){
        //hide the startbutton
        startButton.setVisibility(View.INVISIBLE);
        //create the timer
        mTimer = new Timer();
        //get the lock
        wakeLock.acquire();
        //register the sensing
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        //start the timertask
        mTimer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                if(checkingForTap)
                    checkForTapStateCalculation();
                else
                    normalStateCalculation();
            }

            private void checkForTapStateCalculation(){
                //calculate the values
                ArrayList<float[]> copiedTap = new ArrayList<>(getEvents());
                ArrayList<float[]> copied = new ArrayList<>(getEvents());
                copied = removeValues(8, copied);
                //copiedTap = removeValues(#ofmeasurementswhith high sampling frequency, copiedTap);
                List<Double[]> windowForTapClassification = convertToWindow(copiedTap);
                ResVal resValForTapClassification = new ResVal(windowForTapClassification).invoke();
                Instances unlabeledTap = getInstances();
                unlabeledTap.add(buildDataToBeClassified(resValForTapClassification));
                Instances labeledTap = new Instances(unlabeledTap);

                //Classify whether tap data into {Tap, NoTap}
                try {
                    Instance ins = unlabeledTap.instance(0);
                    double label = classifier.classifyInstance(ins);
                    //TODO: Figure out how big the threshhold should be
                    /*double[] dist = classifier.distributionForInstance(ins);
                    saveDistribution(dist);
                    double highestProb = 0;
                    int indexOfHighest = -1;
                    for (int i = 0; i < dist.length; i++) {
                        if (highestProb < dist[i]) {
                            highestProb = dist[i];
                            indexOfHighest = i;
                        }
                    }*/

                    labeledTap.instance(0).setClassValue(label);

                } catch (Exception e) {
                    e.printStackTrace();
                }

                //TODO: only used some of the data

                //do the state calculation
                List<Double[]> windows = convertToWindow(copied);

                ResVal resVal = new ResVal(windows).invoke();

                //Setup the classification
                Instances unlabeled = getInstances();
                unlabeled.add(buildDataToBeClassified(resVal));
                Instances labeled = new Instances(unlabeled);

                //do the classification
                labeled = doTheClassification(unlabeled, labeled);

                //update the view
                final Instances finalLabeled = labeled;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        double classification =finalLabeled.firstInstance().classValue();
                        //Not tapped
                        if(classification < 0.5){
                            updateViewWithNewPrediction(finalLabeled.firstInstance());
                        //tapped
                        } else {
                            updateViewAndHandleTap(finalLabeled.firstInstance());
                        }
                        printLabeledInstance(finalLabeled.firstInstance());
                    }
                });
            }

            /**
             * More or less done
             * */
            private void normalStateCalculation(){
                //calculate the values
                ArrayList<float[]> copied = new ArrayList<>(getEvents());
                copied = removeValues(8, copied);
                List<Double[]> windows = convertToWindow(copied);

                ResVal resVal = new ResVal(windows).invoke();

                //Setup the classification
                Instances unlabeled = getInstances();
                unlabeled.add(buildDataToBeClassified(resVal));
                Instances labeled = new Instances(unlabeled);

                //do the classification
                labeled = doTheClassification(unlabeled,labeled);

                //update the view
                final Instances finalLabeled = labeled; //need to make variable final
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        updateViewWithNewPrediction(finalLabeled.firstInstance());
                        printLabeledInstance(finalLabeled.firstInstance());
                    }
                });
            }

            /**
             * Works fine
             * */
            private Instances doTheClassification(Instances unlabeled, Instances labeled) {
                try {
                    Instance ins = unlabeled.instance(0);
                    double label = classifier.classifyInstance(ins);
                    double[] dist = classifier.distributionForInstance(ins);
                    saveDistribution(dist);
                    double highestProb = 0;
                    int indexOfHighest = -1;
                    for (int i = 0; i < dist.length; i++) {
                        if (highestProb < dist[i]) {
                            highestProb = dist[i];
                            indexOfHighest = i;
                        }
                    }
                    ArrayList<Double> stateHist = getStateHistory();
                    double lastState;
                    if(stateHist.size()<1)
                        lastState = 0;
                    else
                        lastState = stateHist.get(stateHist.size()-1);

                    //boost the rate of lowenergy
                    if(dist[1]> 0.10){
                        labeled.instance(0).setClassValue((double) 1);
                    }else if(highestProb > 0.8){
                        labeled.instance(0).setClassValue((double) indexOfHighest);
                    } else if ( dist[(int)lastState] < 0.2){
                        labeled.instance(0).setClassValue((double) indexOfHighest);
                    } else if (Math.abs(dist[(int)lastState]-
                            dist[indexOfHighest])
                            < 0.2){
                        labeled.instance(0).setClassValue(lastState);
                    } else {
                        labeled.instance(0).setClassValue(label);
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                return labeled;
            }
        }, WINDOW_SIZE_IN_MILISECONDS, WINDOW_SIZE_IN_MILISECONDS*OVERLAP_IN_PERCENT/100);
    }

    private void updateViewAndHandleTap(Instance instance) {
        String text = instance.toString(); //the prediction and the data used to make it
        String currentState = instance.stringValue(instance.classIndex());

        playNewStateToUser(currentState);
        //Increase sampling frequency
        mSensorManager.unregisterListener(this);
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        showPredictionView.setText(text);
        //remove the invalid prediction
        stateHistory.remove(stateHistory.size()-1);
        stateHistory.add(instance.classValue());


        //Save the new prediction in the statecounter
        if(stateCounter.containsKey(currentState)){
            stateCounter.put(currentState, stateCounter.get(currentState)+1);
        } else {
            stateCounter.put(currentState, 0);
        }

        if(stateHistory.size() > 5)
            stateHistory.remove(0);
    }

    private void playNewStateToUser(String currentState) {
        switch (currentState) {
            case "still":
                playSound(R.raw.still);
                break;
            case "lowenergy":
                playSound(R.raw.lowenergy);
                break;
            default:
                playSound(R.raw.highenergy);
                break;
        }
    }

    private void updateViewWithNewPrediction(Instance prediction){
        String text = prediction.toString(); //the prediction and the data used to make it
        String previousText = (String) showPredictionView.getText();
        String previousState = getState(previousText);
        String currentState = prediction.stringValue(prediction.classIndex());
        if( !currentState.equals(previousState)) {
            playNewStateToUser(currentState);
            //Increase sampling frequency
            mSensorManager.unregisterListener(this);
            mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        }
        showPredictionView.setText(text);
        stateHistory.add(prediction.classValue());

        //Save the new prediction in the statecounter
        if(stateCounter.containsKey(currentState)){
            stateCounter.put(currentState, stateCounter.get(currentState)+1);
        } else {
            stateCounter.put(currentState, 0);
        }

        if(stateHistory.size() > 5)
            stateHistory.remove(0);
    }

    private void saveDistribution(double[] dist) {
        Double[] temp = new Double[3];
        temp[0] = dist[0];
        temp[1] = dist[1];
        temp[2] = dist[2];
        savedDistributions.add(temp);
    }

    private void logDistributionToFile() {
        int time = (int) System.currentTimeMillis();
        File path = getDownloadsDir();
        File file = new File(path+"/test"+time+".arff");

        if (!file.exists()){
            try {
                file.createNewFile();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        try {
            if(isSdReadable()) {
                FileOutputStream fos = new FileOutputStream(file,true);
                OutputStreamWriter osw = new OutputStreamWriter(fos);

                for (Double[]distribution: savedDistributions){
                    osw.append(Arrays.toString(distribution)).append("\r");
                }
                osw.flush();
                osw.close();
                fos.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private boolean isSdReadable() {
        boolean mExternalStorageAvailable = false;
        try {
            String state = android.os.Environment.getExternalStorageState();
            if (android.os.Environment.MEDIA_MOUNTED.equals(state)) {
                mExternalStorageAvailable = true;
                Log.i("isSdReadable", "External storage card is readable.");
            } else if (android.os.Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
                Log.i("isSdReadable", "External storage card is readable.");
                mExternalStorageAvailable = true;
            } else {
                mExternalStorageAvailable = false;
                Log.i("isSdReadable", "External storage card is not readable");
            }
        } catch (Exception ignored) {
        }
        return mExternalStorageAvailable;
    }

    private File getDownloadsDir(){
        File file = new File(String.valueOf(android.os.Environment.getExternalStoragePublicDirectory(android.os.Environment.DIRECTORY_DCIM)));
        if (!file.mkdirs()){
            Log.i("file not present", "the file was not present");
        }
        return file;
    }

    public ArrayList<Double> getStateHistory(){
        return stateHistory;
    }

    private ArrayList<float[]> getEvents() {
        return eventArrayList;
    }

    private ArrayList<float[]> removeValues(int maxSize, ArrayList<float[]> list) {
        if (list.size() > maxSize){
            list =  new ArrayList<>(list.subList( list.size()-maxSize ,list.size()));
        }
        return list;
    }

    private void printLabeledInstance(Instance instance) {
        int attributes = instance.numAttributes();
        String print = "";
        for (int i = 0; i < attributes; i++) {
            print += instance.attribute(i).name() + " ";
        }
        printView.setText(print);
    }

    private List<Double[]> convertToWindow(ArrayList<float[]> eventList) {
        List<Double[]> ret = new ArrayList<>();
        for (float[] event : eventList) {
            Double[] temp = new Double[3];
            temp[0] = (double) event[0];
            temp[1] = (double) event[1];
            temp[2] = (double) event[2];
            ret.add(temp);
        }
        return ret;
    }

    private String getState(String previousText) {
        String reverse = new StringBuffer(previousText).reverse().toString();
        String reverseClass = reverse.substring(0,reverse.indexOf(","));
//        System.out.println("Reverseclass = " + reverseClass);
        return new StringBuffer(reverseClass).reverse().toString();
    }

    public void playSound(int uri){
        MediaPlayer mp = MediaPlayer.create(getActivityContext(), uri);
        mp.start();
        mp.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mp) {
                mp.release();
            }
        });
    }

    private Context getActivityContext() {
        return this;
    }

    private Instances getInstances(){
        Attribute atr2 = new Attribute("maximalVal");
        Attribute atr3 = new Attribute("largestChange");
        Attribute atr4 = new Attribute("averageChange");
        Attribute atr5 = new Attribute("median");
        Attribute atr6 = new Attribute("standardDev");
        ArrayList<String> labels = new ArrayList<String>();
        labels.add("still");
        labels.add("lowenergy");
        labels.add("highenergy");
        Attribute classLabel = new Attribute("class", labels);
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(atr2);
        attributes.add(atr3);
        attributes.add(atr4);
        attributes.add(atr5);
        attributes.add(atr6);
        attributes.add(classLabel);
        Instances dataset = new Instances("dataset", attributes, 10);
        dataset.setClassIndex(dataset.numAttributes()-1);
        return dataset;
    }

    private Instance buildDataToBeClassified(ResVal resVal){
        double maximalVal = resVal.getLargestVal();
        double largestChange = resVal.getLargestChangePerFrame();
        double averageChange = resVal.getAverageChangePerFrame();
        double median = resVal.getMedian();
        double standardDev = resVal.getStandardDeviation();
        double[] vals = {maximalVal,largestChange, averageChange, median, standardDev, 0};
        return new DenseInstance(1, vals);
    }

    public void stopPrediction(View v){
        //stop the sensing
        mSensorManager.unregisterListener(this);
        //stop the recurring task
        mTimer.cancel();
        mTimer.purge();
        //release the wakelock
        wakeLock.release();
        showPredictionView.setText("Stopped");
        logDistributionToFile();
        //Change layout
        setContentView(R.layout.end);
        //add the piechart
        LinearLayout endlayout = (LinearLayout) findViewById(R.id.endlayout);
        PieChart pieChart = new PieChart(this);
        endlayout.addView(pieChart);
        //create piechart
        setupPieChart(pieChart);
        addDataToPieChart(pieChart);
    }

    private void addDataToPieChart(PieChart pieChart) {
        //get the date from the statecounter
        Object[] temp = stateCounter.keySet().toArray();
        String[] xData= Arrays.copyOf(temp, temp.length, String[].class);
        float[] yData = new float[xData.length];
        for (int i = 0; i < xData.length; i++) {
            yData[i] = (float) stateCounter.get(xData[i]);
        }


        //Convert to proper formats
        System.out.println(Arrays.toString(xData));
        System.out.println(Arrays.toString(yData));
        ArrayList<Entry> yVals = new ArrayList<>();
        ArrayList<String> xVals = new ArrayList<>();
        for (int i = 0; i < yData.length; i++) {
            yVals.add(new Entry(yData[i], i));
        }
        Collections.addAll(xVals, xData);

        //convert to PieDataSet
        PieDataSet pieDataSet = new PieDataSet(yVals, "");
        pieDataSet.setSliceSpace(0);
        pieDataSet.setSelectionShift(5);

        //get colors and add to dataset
        ArrayList<Integer> colors = new ArrayList<>();
        for(int c : ColorTemplate.VORDIPLOM_COLORS)
            colors.add(c);

        pieDataSet.setColors(colors);

        //Convert to PieData and set proper values
        PieData pieData = new PieData(xVals, pieDataSet);
        pieData.setValueFormatter(new PercentFormatter());
        pieData.setValueTextColor(Color.GRAY);
        pieData.setValueTextSize(11f);
        pieChart.setData(pieData);
        pieChart.highlightValue(null);

        //trigger a redrawing?
        pieChart.invalidate();
    }

    private void setupPieChart(PieChart pieChart) {
        //size and description
        pieChart.setUsePercentValues(true);
        pieChart.setMinimumHeight(600);
        pieChart.setMinimumWidth(600);
        pieChart.setDescription("");

        //disable hole in the middle
        pieChart.setDrawHoleEnabled(false);

        //allow rotation
        pieChart.setRotationAngle(0);
        pieChart.setRotationEnabled(true);

        //add legend
        Legend legend = pieChart.getLegend();
        legend.setPosition(Legend.LegendPosition.RIGHT_OF_CHART_CENTER);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        //deepcopy... very important. Otherwise we get an arraylist of cloned objects
        float[] copy = new float[3];
        copy[0] = event.values[0];
        copy[1] = event.values[1];
        copy[2] = event.values[2];
        eventArrayList.add(copy);
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {
        //do nothing
    }

    public void goAgain(View view) {
        showPredictionView.setText("Hello world, android activity");
        eventArrayList = new ArrayList<>();
        stateHistory = new ArrayList<>();
        savedDistributions = new ArrayList<>();
        stateCounter = new HashMap<>();
        //set the layout to be the normal
        setContentView(R.layout.main);
        startPrediction(view);
        startButton.setVisibility(View.INVISIBLE);
    }

    private static class ResVal {
        private double largestChangePerFrame;
        private List<Double[]> dataAsLines;
        private double averageChangePerFrame;
        private List<Double> allValues;
        private double median;
        private double standardDeviation;
        private double totalAcc;
        private double largestVal;

        public ResVal(List<Double[]> dataAsLines) {
            this.dataAsLines = dataAsLines;
        }

        private double calculateEuclideanDistanceAndTally(Double[] s) {
            double totalA = 0;
            for (Double value : s) {
                totalA += Math.pow(value, 2);
            }
            return Math.sqrt(totalA);
        }

        public double getTotalAcc(){
            return totalAcc;
        }

        public double getLargestVal(){
            return largestVal;
        }

        public double getLargestChangePerFrame() {
            return largestChangePerFrame;
        }

        public double getAverageChangePerFrame() {
            return averageChangePerFrame;
        }

        public double getMedian(){
            return median;
        }

        public double getStandardDeviation(){
            return standardDeviation;
        }

        public ResVal invoke() {
            double totalChangePerFrame = 0;
            double total = 0;
            totalAcc = 0;
            largestVal = 0;
            allValues = new ArrayList<>();
            //check all the datalines
            for (int i = 0; i < dataAsLines.size(); i++) {
                //sum up all the individual accelerations
                for (int j = 0; j < dataAsLines.get(i).length; j++) {
                    double temp = Math.abs(dataAsLines.get(i)[j]);
                    allValues.add(temp);
                    total += temp;
                    if(largestVal < temp)
                        largestVal = temp;
                }
                //increment the total acceleration of the frame
                totalAcc += calculateEuclideanDistanceAndTally(dataAsLines.get(i));

                //calculate the change in euclidian acceleration from previous one
                double change = 0;
                if (i > 0) {
                    change = Math.abs(
                            calculateEuclideanDistanceAndTally(dataAsLines.get(i - 1)) -
                                    calculateEuclideanDistanceAndTally(dataAsLines.get(i)));
                }

                //update largest
                if(largestChangePerFrame < change) {
                    largestChangePerFrame = change;
                }

                //update total euclidian acceleration for frame
                totalChangePerFrame += change;
            }

            int indexFloorHalf = (int) Math.floor(allValues.size()/2);
            if (dataAsLines.size() % 2 == 0){
                median = (allValues.get(indexFloorHalf-1) + allValues.get(indexFloorHalf)) / 2;
            } else {
                median = allValues.get(indexFloorHalf);
            }

            double mean = total / allValues.size();
            double squaredDifferenceSum = 0;
            for (Double allValue : allValues) {
                squaredDifferenceSum += Math.pow(allValue - mean, 2);
            }
            standardDeviation = Math.sqrt(squaredDifferenceSum/allValues.size());
            averageChangePerFrame = totalChangePerFrame / dataAsLines.size();
            return this;
        }
    }
}

