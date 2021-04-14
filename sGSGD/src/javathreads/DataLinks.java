/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package javathreads;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;

/**
 * Data specifics
 * @author sharma_au
 */
public class DataLinks extends Thread{
    public String grFunc;
    public String errFunc;
    public double[][] curData; //dxN
//    public double [][] validation; 
    public double[] curLabel;
//    public double [] omPlus;
//    public double [] omMinus;
    /**
     * om contains information about "data batches", not on individual data samples.
     */
    public Double [] om; //consistent data
    public boolean[] status;//good = 1; bad = 0;
    public String activation;
    
    final Main mThread;
    final int MINI_BATCH;
    final int DELAY_TOL;
    
    private DataLinks(){
        this.mThread = null;
        this.MINI_BATCH = -1;
        this.DELAY_TOL = -1;
    }
    
    public DataLinks(Main callerThread, final int MINI_BATCH, final int DELAY_TOL){
        this.mThread = callerThread;
        this.MINI_BATCH = MINI_BATCH;
        this.DELAY_TOL = DELAY_TOL;
    }
       
    public void resetOm(){
        om = new Double[DELAY_TOL]; //MINI_BATCH*DELAY_TOL
        for (int i = 0; i < om.length; i++) {
            om[i] = 0.0;
        }
    }
    public void setSlaveFunction(String activation,double[][] curData,double[] curLabel){    
        this.activation = activation;
        this.curData = curData;
        this.curLabel = curLabel;
        status = new boolean[MINI_BATCH]; //curLabel.length];
        om = new Double[DELAY_TOL]; //set of mini-batches
        for (int i = 0; i < om.length; i++) {
            om[i] = 0.0;
        }
//        omMinus = new double[MINI_BATCH];
        Arrays.fill(status, 0, status.length, false);
        this.grFunc = "getSGD";
        this.errFunc = "getError";
    }
    
    public double[][]getData(int []miniBatch) throws IndexOutOfBoundsException{
        int d = curData.length; //rows
        int N = curData[0].length; 
        
        double[][] selectedData = new double[d][miniBatch.length];
        
        for (int m = 0; m < miniBatch.length; m++) {
            for (int i = 0; i < d; i++) { 
                selectedData[i][m] = curData[i][miniBatch[m]];     
            }
        }
        return selectedData;
    }
    
    public double[]getLabel(int []miniBatch) throws IndexOutOfBoundsException{
        int N = curLabel.length;        
        double[] selectedData = new double[miniBatch.length];
        
        for (int m = 0; m < miniBatch.length; m++) { 
            selectedData[m] = curLabel[miniBatch[m]];     
        }
        return selectedData;
    }

    
    public void resetqDataSeq(MyQueue<Integer> qDataSeq, ArrayList<Integer> dataSeq) {
        LinkedList<Integer> seq = new LinkedList<>(MyRandom.randperm(0,curData[0].length));
        
        int additional = MINI_BATCH - 
            (curData[0].length - MINI_BATCH*(int)Math.floor(curData[0].length*1.0/MINI_BATCH));
        
        for (int i = 0; i < additional; i++) { //to make all batches of same size
            seq.add(seq.get(i)); //it will make the sequence circular.
        }    
        
        qDataSeq.clearAll();
        dataSeq.clear();
        
        for (Integer s : seq) {
            qDataSeq.enqueue(s);
            dataSeq.add(s);
        }
    }
    
    synchronized public int[] retrieveMiniBatch(MyQueue<Integer> qDataSeq, ArrayList<Integer> dataSeq){
        int[] miniBatch = new int[MINI_BATCH];//this is new operator called diamond operator in Java 1.7+
        
        if(qDataSeq.curSize()==0){
            mThread.prepareToIncrementEP();
            resetqDataSeq(qDataSeq, dataSeq); 
        }
        
        for (int i = 0; i < MINI_BATCH; i++) {
            miniBatch[i]=qDataSeq.dequeue();
        }

        return miniBatch;
    }
    @Override
    public void run() {
        try {
            synchronized(this){
                System.out.println("DataLink waiting...");
                this.wait();
                System.out.println("Datalink mini batch released...");
            }
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        }
    }   
}
