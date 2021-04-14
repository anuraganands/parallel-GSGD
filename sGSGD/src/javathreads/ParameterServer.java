/*
 * @author Anurag sharma 2019.
 */
package javathreads;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import com.mathworks.engine.MatlabExecutionException;
import com.mathworks.engine.MatlabSyntaxException;
import static java.lang.reflect.Array.set;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;
import java.util.concurrent.ExecutionException;

/**
 * This spawns other weight vectors, that in turn create multiple threads of 
 * size "delay tolerated" &beta;.
 * @author sharma_au
 */
public class ParameterServer extends Thread{
    public MatlabEngine eng; //ref to masterThread eng
    final Main masterThread;
    public int NFC = 0;
    
    /**
     * use {@link MyMath#getArrayDimension(java.lang.Object) }
     */
    public Object Wi; //NOTE*: it can be either double[] or double[]...[], depends on NC (multi-class problems)
    private int t;
    public Object gri;
    public double erri;
    public double prevErr = Double.MAX_VALUE;
    
    public ArrayList<History> hist;
    private double []erri_k;
    
    /**
     * its redundant now. REMOVE IT!
     */
    private MyQueue<Integer> qDataSeq;
    public ArrayList<Integer> dataSeq;

    synchronized public MyQueue<Integer> getQdataSeq() {
        return qDataSeq;
    }
    
    
    
    
    /**
     * {@link ParameterServer#DELAY_TOLERANCE} determines number of threads to
     * be created.
     */
    private final int DELAY_TOLERANCE;
    private GradientThread [] tgrad; 
//    public SlaveFunction DL;
    private SGDvariantsParameters curSGDparameters;
    private int localEp = 0;
    
    
    
    public void prepareToIncrementEP(){
        localEp++;
        
        //since total threads must be equal to DELAY_TOLERANCE
        if(localEp >= DELAY_TOLERANCE){
            masterThread.incrementEp();
        }
    }

    private ParameterServer() {
        this.DELAY_TOLERANCE = 0;
        masterThread = null;    
    }

    public ParameterServer(final Main masterThread, final int delayTolerance) {
        if(delayTolerance<=0 || masterThread == null){
            throw new IllegalArgumentException("Parameter Server incorrectly initialized");
        }
        this.DELAY_TOLERANCE = delayTolerance;
        this.masterThread = masterThread;
//        this.curSGDparameters = new SGDvariantsParameters(eng);
                //masterThread.initSGDparameters; //get()?
        double[][] tmpWi = new double[this.masterThread.d][this.masterThread.NC];
        for (int i = 0; i < this.masterThread.d; i++) {
            for (int j = 0; j < this.masterThread.NC; j++) {
                tmpWi[i][j] = Math.random();                
            }            
        }
        this.Wi = tmpWi;
        this.t = 0;
    }
    
    public void resetHistory(){
        hist = new ArrayList<>();
        final int HIST_SZ = this.DELAY_TOLERANCE;///////////////////////////////////////////////////////
        for (int i = 0; i < HIST_SZ; i++) {
            hist.add(new History());              
        }

        erri_k = new double[HIST_SZ];
        this.masterThread.DL.resetOm(); 
    }
    
    @Override
    public void run() {
        //get weight from DL
            //create threads for gradient
        try{
            this.eng = MatlabEngine.startMatlab();
//            this.eng = engFuture.get();
            eng.eval("cd '" + Main.MatlabPath + "'");
////            this.eng = this.masterThread.eng;

            this.curSGDparameters = masterThread.initSGDparameters;//get initial vals from master thread
            this.curSGDparameters.putVariables(eng);//put variable in local eng
                        
            qDataSeq = new MyQueue<Integer>();
            dataSeq = new ArrayList<>();
            
            resetHistory();
            
            startSlaves();   
            
            while(masterThread.getEp()<masterThread.getEpochs()){ //!this.isInterrupted()){//not very safe...            
                synchronized(this){
//                    System.out.println("PMS("+ this.getId() +") waiting...");
                    this.wait();
                    updateParameters(); //Wi only
//                    System.out.println("parameter updated in (" + this.getId() +")...");                   
                }
            }

            stopSlaves(); 
            
            eng.close();
            
            synchronized(masterThread){
                masterThread.notify();
            }
            
        } catch (InterruptedException ie) {
            ie.printStackTrace();
        } catch (ExecutionException ee) {
            ee.printStackTrace();
        }  
    }
    
    synchronized private void updateParameters(){
        try {
            eng.putVariable("gri",gri);           
            curSGDparameters.putVariables(eng);
            eng.putVariable("t", t);
            eng.putVariable("Wsgd", Wi);
            
            eng.eval("ASGD_updateParameters");  
            NFC++;
            
            curSGDparameters.getVariables(eng);         
            Wi = eng.getVariable("Wsgd"); //Wi
            this.t = this.t+1; //can link it with epochs as well.

            if(Main.useGSGD){
            //<editor-fold defaultstate="collapsed" desc="Code for a/s GSGD">         
            //is it good or bad? (ASGD)
            //<<
                //get verification data(small) randomly
                //verification data represents entire training data set.
                //<<
                final int VERSZ = (int)Math.min(100,this.masterThread.DL.curData[0].length*0.1); //it can be parallalized for >10 for faster process
                int n = this.masterThread.DL.curData[0].length;
                int d = this.masterThread.DL.curData.length;
                double [][]verData = new double[d][VERSZ]; //masterThread.DL.curData; //
                double[] verLabel = new double[VERSZ]; //masterThread.DL.curLabel; //
                
                Random rand = new Random();
                int []rIdx = new int[VERSZ];
                for (int i = 0; i < VERSZ; i++) {
                    rIdx[i] = rand.nextInt(n);
                    for (int j = 0; j < d; j++) {
                        verData[j][i]= this.masterThread.DL.curData[j][rIdx[i]]; //future work - make it a small part of validation data
                    }
                    
                    verLabel[i]= this.masterThread.DL.curLabel[rIdx[i]];
                } 
                //>>

                //this will cause high variation - but fast -> may also put in a separate thread
                double avErr = this.eng.feval(masterThread.DL.errFunc,
                    verData,
                    verLabel,
                    Wi,
                    this.masterThread.DL.activation);
                
////                if(avErr<masterThread.verError){
////                    masterThread.verError = avErr;
////                    masterThread.bestWsoFar = Wi;
////                }
                
                for (int i = 1; i < hist.size(); i++) {
                    erri_k[i] = eng.feval(masterThread.DL.errFunc, hist.get(i).xi, hist.get(i).yi, 
                            Wi, this.masterThread.DL.activation);                                            
                }
                                
                //may be in steps get error from complete data set. [more reliable]
                int pos = 1;
                if(avErr<this.prevErr){//good batch (if ||batch||=1; then it is SGD otherwise minibatch SGD 
                    pos = 2;
                }
                
                for (int h = 1; h < hist.size(); h++) {
//                    for (int i = 0; i < hist.get(h).batch.length; i++) { //from min:1 to max:n
//                            this.masterThread.DL.status[h.batch[i]]=true; //This MUST BE cleared after rho iterations ****NOTE
//                    System.out.println(hist.get(h).getID());
                    if(hist.get(h).getID()>=0)
                        this.masterThread.DL.om[hist.get(h).getID()]//hist.get(h).batch[i]]
                            +=Math.pow((-1),pos)*(prevErr-erri_k[h]);
//                    }
                }

                this.prevErr = avErr;
            //>>

            
            if(/*this.t%this.masterThread.DELAY_TOLERANCE == 0 && */true){
                MyMath.SortIndex<Double> s = new MyMath.SortIndex<>(this.masterThread.DL.om);
                s.sort(Collections.reverseOrder());
                int []omIdx = s.getSortedIndices();
                //only top 5
                
                int[] miniBatch = new int[hist.get(0).batch.length];
                final int minRepeat = 4;
                for (int i = 0; i < Math.min(minRepeat, DELAY_TOLERANCE/2); i++) {
//                    idx = i*DELAY_TOLERANCE;//the first index of a mini-batch
                    if(masterThread.DL.om[omIdx[i]]>0){
                        miniBatch = new int[hist.get(0).batch.length];

                        if(dataSeq.size()<miniBatch.length*omIdx[i]+miniBatch.length){
                            break; //not enough data left
                        }
                        
                        for (int j = 0; j < miniBatch.length; j++) {
                            miniBatch[j]=dataSeq.get(miniBatch.length*omIdx[i]+j).intValue();
                        }
                        Object grk=null;                                               
                        grk = this.eng.feval(masterThread.DL.grFunc,
                                        masterThread.DL.getData(miniBatch),
                                        masterThread.DL.getLabel(miniBatch),
                                        Wi, 
                                        masterThread.DL.activation); 

                        eng.putVariable("gri",grk);           
                        curSGDparameters.putVariables(eng);
                        eng.putVariable("t", t+i+1); 
                        eng.putVariable("Wsgd", Wi);

                        eng.eval("ASGD_updateParameters");  
                        NFC++;
                        
                        curSGDparameters.getVariables(eng);         
                        Wi = eng.getVariable("Wsgd"); //Wi
                    }
                }
                
                dataSeq = new ArrayList<>(dataSeq.subList(Math.min(dataSeq.size(),DELAY_TOLERANCE*miniBatch.length),dataSeq.size()));
//                System.out.println("******* " + dataSeq.size());             
                resetHistory();               
            }
            // </editor-fold>  
            }
      
        } catch (EngineException ee) {
            ee.printStackTrace();
        } catch (InterruptedException ie){
            ie.printStackTrace();
        } catch (MatlabExecutionException mee){
            mee.printStackTrace();
        } catch (MatlabSyntaxException mse){
            mse.printStackTrace();
        } catch (ExecutionException eep){
            eep.printStackTrace();
        }
    }
    
    private void startSlaves(){
        System.out.println("staring slaves for PMS [" + this.getId()+"]");
        tgrad = new GradientThread[DELAY_TOLERANCE];
        for (int i = 0; i < DELAY_TOLERANCE; i++) {
            tgrad[i] = new GradientThread(this); //can do it with one parameter, but leave it as it is.
            tgrad[i].start();
            System.out.println("slave [" + this.getId()+"].{" + tgrad[i].getId() + "} started.");
        }
    }
    
    private void interruptSlaves(){
        for (int i = 0; i < DELAY_TOLERANCE; i++) {            
            tgrad[i].interrupt();
        }
    }
    
    private void notifySlaves(){
        for (int i = 0; i < DELAY_TOLERANCE; i++) {  
            while(tgrad[i].getState() != Thread.State.WAITING);
            synchronized(tgrad[i]){ //making this the "owner". 
                tgrad[i].notify();
            }
        }
    }
    
    private void stopSlaves(){
//        for (int i = 0; i < DELAY_TOLERANCE; i++) {
//            tgrad[i].interrupt();
//        }
//        
        for (int i = 0; i < DELAY_TOLERANCE; i++) {
            try {
                tgrad[i].join();
            } catch (InterruptedException ex) {
                ex.printStackTrace();
            }
        }
    }
    
}
