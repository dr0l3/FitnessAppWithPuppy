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
import com.github.mikephil.charting.charts.LineChart;
import com.github.mikephil.charting.charts.PieChart;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.components.LimitLine;
import com.github.mikephil.charting.components.YAxis;
import com.github.mikephil.charting.data.*;
import com.github.mikephil.charting.formatter.PercentFormatter;
import com.github.mikephil.charting.formatter.YAxisValueFormatter;
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
    private Classifier stateClassifier;
    private Classifier tapClassifier;
    private TextView printView;
    private ArrayList<Double> stateHistory;
    private ArrayList<Double[]> savedDistributions;
    private HashMap<String, Integer> stateCounter;
    private Button startButton;
    private TextView countdownTextView;
    private boolean checkingForTap;
    private int predictionsSinceLastAnnouncement;
    private LineChart mLineChart;
    private int MAX_STATE_HISTORY_SIZE = 60;
    private int MAX_EVENT_HISTORY_SIZE = 10000;
    private int SECONDS_IN_WINDOW = 30;
    private LinearLayout currentLayout;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        currentLayout = (LinearLayout) findViewById(R.id.mainLayout);
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
        predictionsSinceLastAnnouncement = 0;
        checkingForTap = false;
        //TODO: this takes a long time. Loading screen?
        try {
            InputStream classifierStream = getAssets().open("rfWindowv2.model");
            stateClassifier = (Classifier) weka.core.SerializationHelper.read(classifierStream);
            classifierStream = getAssets().open("rfWindowTap.model");
            tapClassifier = (Classifier) weka.core.SerializationHelper.read(classifierStream);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void startPrediction(View v){

        //hide the startbutton
        setContentView(R.layout.during);
        currentLayout = (LinearLayout) findViewById(R.id.duringLayout);
        mLineChart = new LineChart(this);
        currentLayout.addView(mLineChart);
        setupLineChart(mLineChart);
        predictionsSinceLastAnnouncement = 0;
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
                normalStateCalculation();
                /*
                if(checkingForTap) {
                    checkForTapStateCalculation();
                }
                else {
                    normalStateCalculation();
                }*/
            }

            private void checkForTapStateCalculation(){
                //calculate the values
                ArrayList<float[]> copiedTap = new ArrayList<>(getEvents());
                ArrayList<float[]> copied = new ArrayList<>(getEvents());
                copied = removeValues(400, copied);
                copiedTap = removeValues(400, copiedTap);
                List<Double[]> windowForTapClassification = convertToWindow(copiedTap);
                TapFeatureCalculator tapFeatureCalculator = new TapFeatureCalculator(windowForTapClassification).invoke();
                Instances unlabeledTap = getInstancesTap();
                unlabeledTap.add(buildTapDataToBeClassified(tapFeatureCalculator));
                Instances labeledTap = new Instances(unlabeledTap);

                //Classify tap data into {Tap, NoTap}
                try {
                    Instance ins = unlabeledTap.instance(0);
                    double label = tapClassifier.classifyInstance(ins);
                    //TODO: Figure out how big the threshhold should be
                    double[] dist = tapClassifier.distributionForInstance(ins);
                    //saveDistribution(dist);
                    /*double highestProb = 0;
                    int indexOfHighest = -1;
                    for (int i = 0; i < dist.length; i++) {
                        if (highestProb < dist[i]) {
                            highestProb = dist[i];
                            indexOfHighest = i;
                        }
                    }*/

                    System.out.println("Tapdistribution = " + Arrays.toString(dist));

                    labeledTap.instance(0).setClassValue(label);

                } catch (Exception e) {
                    e.printStackTrace();
                }

                //extracting every ninth element to account for the extra measurements
                //needed to detect tapping
                int n = 195;
                copied = extractEveryNthMeasurement(copied, n);
                //do the state calculation
                List<Double[]> windows = convertToWindow(copied);

                StateFeatureCalculator stateFeatureCalculator = new StateFeatureCalculator(windows).invoke();

                //Setup the classification
                Instances unlabeled = getInstances();
                unlabeled.add(buildStateDataToBeClassified(stateFeatureCalculator));
                Instances labeled = new Instances(unlabeled);

                //do the classification
                labeled = doTheClassification(unlabeled, labeled);

                //update the view
                final Instances finalLabeledTap = labeledTap;
                final Instances finalLabeled = labeled;
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        double classification =finalLabeledTap.firstInstance().classValue();
                        System.out.println("Tapclassificaito = " + classification);
                        //tapped
                        if(classification < 0.5){
                            updateViewAndHandleTap(finalLabeled.firstInstance());
                        //not tapped
                        } else {
                            updateViewWithNewPrediction(finalLabeled.firstInstance());
                        }
                        decreaseSamplingFrequency();
                        printLabeledInstance(finalLabeled.firstInstance());
                    }
                });
            }

            private void normalStateCalculation(){
                //calculate the values
                ArrayList<float[]> copied = new ArrayList<>(getEvents());
                copied = removeValues(8, copied);
                List<Double[]> windows = convertToWindow(copied);

                StateFeatureCalculator stateFeatureCalculator = new StateFeatureCalculator(windows).invoke();

                //Setup the classification
                Instances unlabeled = getInstances();
                unlabeled.add(buildStateDataToBeClassified(stateFeatureCalculator));
                Instances labeled = new Instances(unlabeled);

                //do the classification
                labeled = doTheClassification(unlabeled,labeled);

                //update the view
                final Instances finalLabeled = labeled; //need to make variable final
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        updateViewWithNewPrediction(finalLabeled.firstInstance());
//                        printLabeledInstance(finalLabeled.firstInstance());
                    }
                });
            }

            private Instances doTheClassification(Instances unlabeled, Instances labeled) {
                try {
                    Instance ins = unlabeled.instance(0);
                    double label = stateClassifier.classifyInstance(ins);
                    double[] dist = stateClassifier.distributionForInstance(ins);
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
                    double lastState = stateHist.get(stateHist.size()-1);

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

        //playNewStateToUser(currentState);
        showPredictionView.setText(text);
        //remove the invalid prediction
        stateHistory.remove(stateHistory.size()-1);
        stateHistory.add(instance.classValue());

        playSound(R.raw.recalculating);

        //Save the new prediction in the statecounter
        if(stateCounter.containsKey(currentState)){
            stateCounter.put(currentState, stateCounter.get(currentState)+1);
        } else {
            stateCounter.put(currentState, 0);
        }

        if(stateHistory.size() > MAX_STATE_HISTORY_SIZE)
            stateHistory.remove(0);
    }

    private void updateViewWithNewPrediction(Instance prediction){
        String text = prediction.toString(); //the prediction and the data used to make it
        String previousText = (String) showPredictionView.getText();
        String previousState = getState(previousText);
        String currentState = prediction.stringValue(prediction.classIndex());
        stateHistory.add(prediction.classValue());
        if( predictionsSinceLastAnnouncement >= SECONDS_IN_WINDOW-1) {
            announceMostCommonState();
            predictionsSinceLastAnnouncement = 0;
        } else {
            predictionsSinceLastAnnouncement++;
        }

        updateLineChartData(mLineChart);
        showPredictionView.setText(text);


        //Save the new prediction in the statecounter
        if(stateCounter.containsKey(currentState)){
            stateCounter.put(currentState, stateCounter.get(currentState)+1);
        } else {
            stateCounter.put(currentState, 1);
        }

        if(stateHistory.size() > MAX_STATE_HISTORY_SIZE)
            stateHistory.remove(0);
    }

    private void increaseSamplingFrequency() {
        mSensorManager.unregisterListener(this);
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_FASTEST);
        checkingForTap = true;
    }

    private void decreaseSamplingFrequency() {
        mSensorManager.unregisterListener(this);
        mSensorManager.registerListener(this, mAccelerometer, SensorManager.SENSOR_DELAY_NORMAL);
        checkingForTap = false;
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
        //increaseSamplingFrequency();
    }

    private ArrayList<float[]> extractEveryNthMeasurement(ArrayList<float[]> copied, int n) {
        ArrayList<float[]> res = new ArrayList<>();
        for (int i = 0; i < copied.size(); i++) {
            if (i % n == 0)
                res.add(copied.get(i));
        }
        return res;
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

    private Instances getInstancesTap(){
        Attribute atr1 = new Attribute("maxEucChangeTwo");
        Attribute atr2 = new Attribute("maxComChangeTwo");
        Attribute atr3 = new Attribute("maxIndChangeTwo");
        Attribute atr4 = new Attribute("maxComChangeFour");
        Attribute atr5 = new Attribute("maxComChangeSix");
        Attribute atr6 = new Attribute("maxEucChangeEight");
        Attribute atr7 = new Attribute("maxIndVal");
        ArrayList<String> labels = new ArrayList<>();
        labels.add("tapping");
        labels.add("nottapping");
        Attribute classLable = new Attribute("class", labels);
        ArrayList<Attribute> attributes = new ArrayList<>();
        attributes.add(atr1);
        attributes.add(atr2);
        attributes.add(atr3);
        attributes.add(atr4);
        attributes.add(atr5);
        attributes.add(atr6);
        attributes.add(atr7);
        attributes.add(classLable);
        Instances dataset = new Instances("dataset", attributes, 400);
        dataset.setClassIndex(dataset.numAttributes()-1);
        return dataset;
    }

    private Instance buildStateDataToBeClassified(StateFeatureCalculator stateFeatureCalculator){
        double maximalVal = stateFeatureCalculator.getLargestVal();
        double largestChange = stateFeatureCalculator.getLargestChangePerFrame();
        double averageChange = stateFeatureCalculator.getAverageChangePerFrame();
        double median = stateFeatureCalculator.getMedian();
        double standardDev = stateFeatureCalculator.getStandardDeviation();
        double[] vals = {maximalVal,largestChange, averageChange, median, standardDev, 0};
        return new DenseInstance(1, vals);
    }

    private Instance buildTapDataToBeClassified(TapFeatureCalculator tapFeatureCalculator){
        double maxEucChangeTwo = tapFeatureCalculator.getMaxEucChange2Frames();
        double maxComChangeTwo = tapFeatureCalculator.getMaxComChange2Frames();
        double maxIndChangeTwo = tapFeatureCalculator.getMaxIndChange2Frames();
        double maxComChangeFour = tapFeatureCalculator.getMaxComChange4Frames();
        double maxComChangeSix = tapFeatureCalculator.getMaxComChange6Frames();
        double maxEucChangeEight = tapFeatureCalculator.getMaxEucChange8Frames();
        double maxIndVal = tapFeatureCalculator.getLargestIndividualValue();
        double[] vals = {maxEucChangeTwo,maxComChangeTwo, maxIndChangeTwo, maxComChangeFour, maxComChangeSix, maxEucChangeEight, maxIndVal, 0};
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

    private void announceMostCommonState() {
        //get state history
        ArrayList<Double> recentState = new ArrayList<>(stateHistory.subList(stateHistory.size()-SECONDS_IN_WINDOW, stateHistory.size()));
        Collections.sort(recentState);
        Double mostCommon = 0d;
        Double last = null;
        int mostCount = 0;
        int lastCount = 0;
        for (Double state : recentState) {
            if (state.equals(last)){
                lastCount++;
            } else {
                if (lastCount > mostCount) {
                    mostCount = lastCount;
                    mostCommon = last;

                }
                lastCount = 0;
            }
            last = state;
        }
        if (lastCount > mostCount)
            mostCommon = last;
        double i = mostCommon;
        if (i == 0d) {
            playSound(R.raw.still);

        } else if (i == 1d) {
            playSound(R.raw.lowenergy);

        } else {
            playSound(R.raw.highenergy);

        }
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

    private void setupLineChart(LineChart lineChart) {
        lineChart.setDescription("");
        lineChart.setAutoScaleMinMaxEnabled(false);
        lineChart.setMinimumHeight(300);
        lineChart.setMinimumWidth(700);
        lineChart.getAxisRight().setEnabled(false);
        lineChart.getXAxis().setEnabled(false);
        lineChart.getAxisLeft().setAxisMaxValue(2f);
        lineChart.getAxisLeft().setAxisMinValue(0f);
        lineChart.getAxisLeft().setLabelCount(3,true);
        lineChart.getAxisLeft().setValueFormatter(new YAxisValueFormatter() {
            @Override
            public String getFormattedValue(float value, YAxis yAxis) {
                if (value == 0.0f){
                    return "Still";
                } else if (value == 1.0f){
                    return "Low";
                } else {
                    return "High";
                }
            }
        });
        Legend legend = lineChart.getLegend();
        legend.setEnabled(false);
        legend.setPosition(Legend.LegendPosition.BELOW_CHART_CENTER);
        lineChart.invalidate();
    }

    private void updateLineChartData(LineChart lineChart) {
        //get states from state history
        //remove previous limitlines
        lineChart.getXAxis().removeAllLimitLines();
        if (stateHistory.size() >= 30) {
            LimitLine lastAnnouncement = new LimitLine(stateHistory.size() - predictionsSinceLastAnnouncement, "");
            lastAnnouncement.setLineColor(Color.BLACK);
            lastAnnouncement.setTextSize(15f);
            lineChart.getXAxis().addLimitLine(lastAnnouncement);
        }

        if (stateHistory.size() >= 60) {
            LimitLine secondLastAnnouncement = new LimitLine(stateHistory.size() - predictionsSinceLastAnnouncement - 30, "");
            secondLastAnnouncement.setLineColor(Color.BLACK);
            secondLastAnnouncement.setTextSize(15f);
            lineChart.getXAxis().addLimitLine(secondLastAnnouncement);
        }

        //create the data
        //create the dataset
        //get the xVals
        ArrayList<String> xVals = new ArrayList<>();
        for (int i = 0; i < stateHistory.size(); i++) {
            xVals.add("");
        }
        //get the yVals
        ArrayList<Entry> yVals = new ArrayList<>();
        for (int i = 0; i < stateHistory.size(); i++) {
            float val =  stateHistory.get(i).floatValue();
            yVals.add(new Entry(val,i));
        }

        LineDataSet lineDataSet = new LineDataSet(yVals, "data");
        lineDataSet.setColor(Color.RED);
        lineDataSet.setCircleColor(Color.RED);
        lineDataSet.setDrawValues(false);
        ArrayList<LineDataSet> dataSets = new ArrayList<>();
        dataSets.add(lineDataSet);
        LineData lineData = new LineData(xVals, dataSets);
        lineChart.setData(lineData);
        lineChart.invalidate();
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

    public void correctStateInference(View view) {
        //show countdown
        setContentView(R.layout.countdown);
        countdownTextView = (TextView) findViewById(R.id.textviewCountdown);
        countdownTextView.setText("Corrected measurement. Resuming in " + 5 + " seconds!");

        //cancel timers and unregister
        mTimer.cancel();
        mSensorManager.unregisterListener(this);

        ArrayList<Double> toBeDeleted=
                new ArrayList<>(stateHistory.subList(
                        stateHistory.size() - predictionsSinceLastAnnouncement,
                        stateHistory.size()));
        if (stateHistory.size() > SECONDS_IN_WINDOW){
            toBeDeleted.addAll(stateHistory.subList(
                    stateHistory.size()-predictionsSinceLastAnnouncement-SECONDS_IN_WINDOW,
                    stateHistory.size()-predictionsSinceLastAnnouncement));
        }

        stateHistory.removeAll(toBeDeleted);
        //decrement counters
        if( toBeDeleted.size() > 0) {
            for (Double measurement : toBeDeleted) {
                String key = stateDoubleToString(measurement);
                int newValue = stateCounter.get(key) - 1;
                if (newValue >= 0)
                    stateCounter.put(key, newValue);
                else
                    stateCounter.put(key, 0);
            }
        }

        //schedule restart
        Timer timer = new Timer();
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                String text = countdownTextView.getText().toString();
                int currentNumber = Integer.decode(String.valueOf(text.charAt(35)));
                if (currentNumber == 0){
                    runOnUiThread(new Runnable() {
                                      @Override
                                      public void run() {
                                          startPrediction(view);
                                          timer.cancel();
                                      }
                                  });

                } else {
                    String newText = text.replaceFirst(String.valueOf(currentNumber), String.valueOf(currentNumber-1));
                    runOnUiThread(new Runnable() {
                        @Override
                        public void run() {
                            countdownTextView.setText(newText);
                        }
                    });
                }
            }
        }, 1000,1000);
    }

    private String stateDoubleToString(Double state){
        if (state == 0.0){
            return "still";
        } else if (state == 1.0){
            return "lowenergy";
        } else {
            return "highenergy";
        }
    }

    private static class StateFeatureCalculator {
        private double largestChangePerFrame;
        private List<Double[]> dataAsLines;
        private double averageChangePerFrame;
        private List<Double> allValues;
        private double median;
        private double standardDeviation;
        private double totalAcc;
        private double largestVal;

        public StateFeatureCalculator(List<Double[]> dataAsLines) {
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

        public StateFeatureCalculator invoke() {
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

    private static class TapFeatureCalculator{
        private List<Double[]> dataAsLines;
        private double maxEucChange2Frames = 0;
        private double maxComChange2Frames = 0;
        private double maxIndChange2Frames = 0;
        private double maxEucChange4Frames = 0;
        private double maxComChange4Frames = 0;
        private double maxIndChange4Frames = 0;
        private double maxEucChange6Frames = 0;
        private double maxComChange6Frames = 0;
        private double maxIndChange6Frames = 0;
        private double maxEucChange8Frames = 0;
        private double maxComChange8Frames = 0;
        private double maxIndChange8Frames = 0;
        private double maxEucChange10Frames = 0;
        private double maxComChange10Frames = 0;
        private double maxIndChange10Frames = 0;
        private double largestEuclidianValue = 0;
        private double largestIndividualValue = 0;

        public TapFeatureCalculator(List<Double[]> dataAsLines){
            this.dataAsLines = dataAsLines;
        }

        public TapFeatureCalculator invoke(){
            ArrayList<Double> euclidianDistances = new ArrayList<>();
            //calculate euclidian distance for all lines
            for (Double[] dataAsLine : dataAsLines) {
                euclidianDistances.add(calculateEuclidianDistance(dataAsLine));
            }

            for (int i = 0; i < dataAsLines.size(); i++) {
                if (i > 0) {
                    //calculate two
                    double tempEuclidianDistance = euclidianDistances.get(i) - euclidianDistances.get(i - 1);
                    if (tempEuclidianDistance > maxEucChange2Frames)
                        maxEucChange2Frames = tempEuclidianDistance;
                    double tempCombinedChange =
                            calculateCombinedDifferenceBetweenTwoMeasurements(dataAsLines.get(i), dataAsLines.get(i - 1));
                    if (tempCombinedChange > maxComChange2Frames)
                        maxComChange2Frames = tempCombinedChange;
                    double tempLargestIndividualChange =
                            calculateLargestIndividualChange(dataAsLines.get(i), dataAsLines.get(i - 1));
                    if (maxIndChange2Frames < tempLargestIndividualChange)
                        maxIndChange2Frames = tempLargestIndividualChange;
                }
                if (i > 2) {
                    TempCalc tempCalc = new TempCalc(dataAsLines, maxEucChange4Frames, maxComChange4Frames, maxIndChange4Frames, euclidianDistances, i, 3).invoke();
                    maxEucChange4Frames = tempCalc.getMaxEucChangeBetweenFrames();
                    maxComChange4Frames = tempCalc.getMaxComChangeBetweenFrames();
                    maxIndChange4Frames = tempCalc.getMaxIndChangeBetweenFrames();
                }
                if (i > 4) {
                    TempCalc tempCalc = new TempCalc(dataAsLines, maxEucChange6Frames, maxComChange6Frames, maxIndChange6Frames, euclidianDistances, i, 5).invoke();
                    maxEucChange6Frames = tempCalc.getMaxEucChangeBetweenFrames();
                    maxComChange6Frames = tempCalc.getMaxComChangeBetweenFrames();
                    maxIndChange6Frames = tempCalc.getMaxIndChangeBetweenFrames();
                }
                if (i > 6) {
                    TempCalc tempCalc = new TempCalc(dataAsLines, maxEucChange8Frames, maxComChange8Frames, maxIndChange8Frames, euclidianDistances, i, 7).invoke();
                    maxEucChange8Frames = tempCalc.getMaxEucChangeBetweenFrames();
                    maxComChange8Frames = tempCalc.getMaxComChangeBetweenFrames();
                    maxIndChange8Frames = tempCalc.getMaxIndChangeBetweenFrames();
                }

                if (i > 8) {
                    TempCalc tempCalc = new TempCalc(dataAsLines, maxEucChange10Frames, maxComChange10Frames, maxIndChange10Frames, euclidianDistances, i, 9).invoke();
                    maxEucChange10Frames = tempCalc.getMaxEucChangeBetweenFrames();
                    maxComChange10Frames = tempCalc.getMaxComChangeBetweenFrames();
                    maxIndChange10Frames = tempCalc.getMaxIndChangeBetweenFrames();
                }

                if (largestEuclidianValue < euclidianDistances.get(i))
                    largestEuclidianValue = euclidianDistances.get(i);
                double tempMaxIndVal = calculateLargestIndividualValue(dataAsLines.get(i));
                if (largestIndividualValue < tempMaxIndVal)
                    largestIndividualValue = tempMaxIndVal;
            }
            return this;
        }

        private Double calculateEuclidianDistance(Double[] doubles) {
            double sumPoweredByTwo = 0;
            for (Double value : doubles) {
                sumPoweredByTwo += Math.pow(value, 2);
            }
            return Math.sqrt(sumPoweredByTwo);
        }

        private static double calculateLargestIndividualValue(Double[] doubles) {
            double max = 0;
            for (Double temp : doubles) {
                if (temp > max)
                    max = temp;
            }
            return max;
        }

        private static double calculateLargestIndividualChange(Double[] doubles, Double[] doubles1) {
            double maxChange = 0;
            for (int i = 0; i < doubles.length; i++) {
                double temp = doubles[i] - doubles1[i];
                if ( temp> maxChange)
                    maxChange = temp;
            }
            return maxChange;
        }

        private static double calculateCombinedDifferenceBetweenTwoMeasurements(Double[] doubles, Double[] doubles1) {
            double combinedChange = 0;
            for (int i = 0; i < doubles.length; i++) {
                combinedChange += Math.abs(doubles[i] - doubles1[i]);
            }
            return combinedChange;
        }

        public double getMaxEucChange2Frames() {
            return maxEucChange2Frames;
        }

        public double getMaxComChange2Frames() {
            return maxComChange2Frames;
        }

        public double getMaxIndChange2Frames() {
            return maxIndChange2Frames;
        }

        public double getMaxEucChange4Frames() {
            return maxEucChange4Frames;
        }

        public double getMaxComChange4Frames() {
            return maxComChange4Frames;
        }

        public double getMaxIndChange4Frames() {
            return maxIndChange4Frames;
        }

        public double getMaxEucChange6Frames() {
            return maxEucChange6Frames;
        }

        public double getMaxComChange6Frames() {
            return maxComChange6Frames;
        }

        public double getMaxIndChange6Frames() {
            return maxIndChange6Frames;
        }

        public double getMaxEucChange8Frames() {
            return maxEucChange8Frames;
        }

        public double getMaxComChange8Frames() {
            return maxComChange8Frames;
        }

        public double getMaxIndChange8Frames() {
            return maxIndChange8Frames;
        }

        public double getMaxEucChange10Frames() {
            return maxEucChange10Frames;
        }

        public double getMaxComChange10Frames() {
            return maxComChange10Frames;
        }

        public double getMaxIndChange10Frames() {
            return maxIndChange10Frames;
        }

        public double getLargestEuclidianValue() {
            return largestEuclidianValue;
        }

        public double getLargestIndividualValue() {
            return largestIndividualValue;
        }

        private static class TempCalc {
            private List<Double[]> segment;
            private double maxEucChangeXFrames;
            private double maxComChangeXFrames;
            private double maxIndChangeXFrames;
            private ArrayList<Double> euclidianDistances;
            private int i;
            private int x;

            public TempCalc(List<Double[]> segment, double maxEucChangeXFrames, double maxComChangeXFrames, double maxIndChangeXFrames, ArrayList<Double> euclidianDistances, int i, int x) {
                this.segment = segment;
                this.maxEucChangeXFrames = maxEucChangeXFrames;
                this.maxComChangeXFrames = maxComChangeXFrames;
                this.maxIndChangeXFrames = maxIndChangeXFrames;
                this.euclidianDistances = euclidianDistances;
                this.i = i;
                this.x = x;
            }

            public double getMaxEucChangeBetweenFrames() {
                return maxEucChangeXFrames;
            }

            public double getMaxComChangeBetweenFrames() {
                return maxComChangeXFrames;
            }

            public double getMaxIndChangeBetweenFrames() {
                return maxIndChangeXFrames;
            }

            public TempCalc invoke() {
                double tempEucXFrames = 0;
                double tempComXFrames = 0;
                double tempIndXFrames = 0;
                for (int j = 0; j < x; j++) {
                    tempEucXFrames += euclidianDistances.get(i-j) - euclidianDistances.get(i-j-1);
                    tempComXFrames += calculateCombinedDifferenceBetweenTwoMeasurements(
                            segment.get(i-j), segment.get(i-j-1));
                    tempIndXFrames = calculateLargestIndividualChange(
                            segment.get(i-j), segment.get(i-j-1));
                }
                if (tempEucXFrames > maxEucChangeXFrames)
                    maxEucChangeXFrames = tempEucXFrames;
                if (tempComXFrames > maxComChangeXFrames)
                    maxComChangeXFrames = tempComXFrames;
                if (tempIndXFrames > maxIndChangeXFrames)
                    maxIndChangeXFrames = tempIndXFrames;
                return this;
            }
        }
    }
}

