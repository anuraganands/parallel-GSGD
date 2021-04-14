/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package javathreads;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import com.mathworks.engine.MatlabSyntaxException;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 *
 * @author sharma_au
 */
public class GradientThread  extends Thread{
//    public Future<Integer> thread;
//    private ExecutorService executor  = Executors.newSingleThreadExecutor();
    
    MatlabEngine eng; 
    public Object output = null;
    final ParameterServer psThread; //also associated with eng of ParameterServer
   
    private GradientThread(){
        psThread = null;
    }
    
    public GradientThread(ParameterServer callerThread){
        psThread = callerThread;    
    }
    
    @Override
    public void run() { 
//    public Future<Integer> runAsync() {  
//        thread =  executor.submit(new Callable<Integer>() {
//            public Integer call() {
//////        Future<MatlabEngine> engFuture = MatlabEngine.startMatlabAsync();
        try {
            eng = MatlabEngine.startMatlab();
//            this.eng = engFuture.get();
            eng.eval("cd '" + Main.MatlabPath +  "'");
//            this.eng = this.psThread.eng;
        } catch (InterruptedException ie) {
            ie.printStackTrace();
        } catch (ExecutionException ee) {
            ee.printStackTrace();
        }
        runThis();
        
//        return 0;
//        }});
//        
//        return thread;
    }

    String getActivation(){
        return psThread.masterThread.DL.activation;
    }
    
    void runThis(){
        try {
            while(psThread.masterThread.getEp()<psThread.masterThread.Epochs){//!this.isInterrupted()){
                //Can also generate locally for fully Async SGD
                
//                synchronized(this){
//                    if(History.batIndx == psThread.masterThread.DELAY_TOLERANCE){                   
//                        this.wait();
//                    }
//                }
                int []miniBatch = psThread.masterThread.DL.retrieveMiniBatch(psThread.getQdataSeq(), psThread.dataSeq);

                Object gri = null;
                double [][]xi = psThread.masterThread.DL.getData(miniBatch);
                double []yi = psThread.masterThread.DL.getLabel(miniBatch); 
                History current = new History();
                current.batch = miniBatch;
                current.xi = xi;
                current.yi = yi; 
                
                
//                double erri = Double.MAX_VALUE;

                try{                  
                    gri = this.eng.feval(psThread.masterThread.DL.grFunc,
                        xi,
                        yi,
                        psThread.Wi, 
                        getActivation());                    

                    //extra wait here move it to buffer region such as DB connection?
                    synchronized(psThread){ //Shalvindra to look into this.  
                        psThread.gri = gri;
                        
                        double []tmp = (double[])psThread.gri;
                        System.out.println("gri " + psThread.masterThread.getEp());
//                        for (int i = 0; i < 1; i++) {
////                            for (int j = 0; j < tmp.length; j++) {
////                                System.out.print(tmp[j] + " ");
////                            }
////                            System.out.println("");
//                        }
                        
                        
                        if(History.batIndx < psThread.masterThread.DELAY_TOLERANCE){                        
                            History.batchAdd();
                            current.generateID();
//                            System.out.println("*" + current.getID());
                            for (int i = psThread.hist.size()-1; i >=1; i--) {
                                psThread.hist.set(i, psThread.hist.get(i-1)); 
                            }
                            psThread.hist.set(0, current);

                        }else{
                            History.batchReset();
                            History.batchAdd();
                            current.generateID();
                            psThread.resetHistory();
                            psThread.hist.set(0, current);
                        }  
                           
                        Thread.sleep(1);
//                        psThread.notify(); //wakes up the thread
                        

                    }
                
                }catch(ClassCastException cce){
                    cce.printStackTrace();
                }               
            }      

//            synchronized(psThread){ //clear any suspended thread
//                psThread.notify();
//            }
            eng.close();        
        } catch (EngineException ee) {
            ee.printStackTrace();
        } catch (InterruptedException ie){
            ie.printStackTrace();
        } catch (MatlabSyntaxException mse){
            mse.printStackTrace();
        } catch (ExecutionException eep){
            eep.printStackTrace();
        }
    }
}
