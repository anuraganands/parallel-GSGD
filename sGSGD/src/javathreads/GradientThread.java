/*
 * @author Anurag sharma 2019.
 */
package javathreads;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import com.mathworks.engine.MatlabSyntaxException;
import java.util.concurrent.ExecutionException;

/**
 *
 * @author sharma_au
 */
public class GradientThread extends Thread {
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
//        Future<MatlabEngine> engFuture = MatlabEngine.startMatlabAsync();
        try {
            this.eng = MatlabEngine.startMatlab();
//            this.eng = engFuture.get();
            eng.eval("cd '"+ Main.MatlabPath+"'");
//            this.eng = this.psThread.eng;
        } catch (InterruptedException ie) {
            ie.printStackTrace();
        } catch (ExecutionException ee) {
            ee.printStackTrace();
        }
        runThis();
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
//                        psThread.erri = erri; 
                        
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

                        psThread.notify(); //wakes up the thread
                       
                    }
                
                }catch(ClassCastException cce){
                    cce.printStackTrace();
                }               
            }      

            synchronized(psThread){ //clear any suspended thread
                psThread.notify();
            }
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
