/**
 * Title: parallel ASGD
 * @author Anurag sharma 2019
 * Version 1
 * Technical Errors encountered:
 * 1 - once an array from Matlab is retrieved in Java it becomes "row-wise" data.
 *      so it becomes necessary to convert it back to "col-wise". Matlab code are
 *      updated to check it. 
 * Other observations:
 * 1 - SGD works well with |minibatch| = 1.
 */
package javathreads;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import com.mathworks.engine.MatlabSyntaxException;
import java.io.File;
import java.io.FileFilter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.concurrent.ExecutionException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author sharma_au
 */
public class Main extends Thread{
    //when MAXCPU = 1; and DELAY_TOLERANCE = 1; that means |Wi| = 1; => a simple SGD (with mini-batch)    
    final int MAXCPU = 10; //100 ; //3; //15; //can also set it up in .ini file
    final int DELAY_TOLERANCE = 10; //MAXCPU; //=MAXCPU means pure ASGD (i.e. only ONE wi). It means total gradient threads. //10; //1; //3; // [no of threads] per [weight (Wi)]
////    final int mergeAt = 5;//After how many Eps, proceed to select or refine the Wis.
    //total Wi = MAXCPU/DELAY_TOLERANCE
    final int MINI_BATCH = 4; //4;// iterature suggest to have in the order of 2. up to 32..64 etc. [1 => complete SGD; >1 => mini batch SGD]
    final int Epochs = 50;
    public static final boolean useGSGD = true;
        
    static final int totalRuns = 1;
    static int curRun = -1;
    double Eout = 0.0;
    public double verError = Double.MAX_VALUE;
    public static final String MatlabPath = "C:\\Anurag\\Deep Learning\\ASGD\\ASDG_MATLAB";
           
    public Object bestWsoFar;
    public int bestNFC=0;
    private boolean isSynchronous;
    private ParameterServer psThread;
    private double[][] curData;
    private double[] curLabel;
    private double[][] curData4Test;
    private double[] curLabel4Test;
    private double[][] curData4val;
    private double[] curLabel4val;
    private PrintWriter runHistory = null;
    private String outFile = "";
    
    private boolean bNotified = false;
    private String activation;
    public DataLinks DL;
       
    private int ep; //current epoch
    MatlabEngine eng;
    public SGDvariantsParameters initSGDparameters;
    
    public int d;
    public int N;
    public int NC;

    public int getEp() {
        return ep;
    }

    public int getEpochs() {
        return Epochs;
    }    

    public void setCurData(double[][] curData, double[][] curData4Test, double [][] curData4val) {
        this.curData = curData;
        this.curData4Test = curData4Test;
        this.curData4val = curData4val;
        DL.setSlaveFunction(activation, curData, curLabel);
    }

    public void setCurLabel(double[] curLabel, double[] curLabel4Test, double [] curLabel4val) {
        this.curLabel = curLabel;
        this.curLabel4Test = curLabel4Test;
        this.curLabel4val = curLabel4val;
        DL.setSlaveFunction(activation, curData, curLabel);
    }

    public void setActivation(String activation) {
        this.activation = activation;
        DL.setSlaveFunction(activation, curData, curLabel);
    }
    
//    private int eTmp = 0;
    public void prepareToIncrementEP(){
//        eTmp++;
//        if(eTmp%getTotalWeights() == 0){
//            eTmp = 0;
            incrementEp();
//        }
    }
    
    public void incrementEp() {
        try {
            eng.putVariable("ep", ++this.ep);
            eng.eval("plotNow = true;");           
            
            System.out.println("Ep: " + this.ep); // + " [Verification Err: " + this.verError+"]");
            double tmp=0.0;
            if(ep%1 == 0){ //%5?
                //Generally Testing is data is NEVER used here but the paper is
                //all about impact of threshold on accuracy. Therefore it is ok
                //to have test data in this context. 
                tmp = this.eng.feval("getEout",
                    psThread.Wi,
                    this.curData4Test,
                    this.curLabel4Test,
                    DL.activation);
                System.out.println("SR: " + new DecimalFormat("#.###").format(tmp)+"\n");
                if(tmp>Eout){
                    Eout = tmp;
                    bestWsoFar = psThread.Wi;
                    bestNFC = psThread.NFC;
                }    
                runHistory.println(ep +","+ bestNFC+","+ tmp+","+MAXCPU+","+DELAY_TOLERANCE+","+MINI_BATCH);
            }
            
            eng.putVariable("Wsgd", psThread.Wi);
            eng.eval("ASGD_whileEpoch");
            
////            if(this.ep%mergeAt == 0 ){ //need to dynamically determine this... based on Wi+slaves+mini_batch
////                synchronized(this){ //this step is processed after invokation only. Not through continuous loop.
////                    this.notify(); 
////                }
////            }           
        } catch (InterruptedException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IllegalStateException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    

    public Main() throws IOException {
        //Enable this if using ini file
//        Ini ini;
//        ini = new Ini(new File("parameters.ini"));
//        java.util.prefs.Preferences prefs = new IniPreferences(ini);
//        this.MAXCPU = Integer.parseInt( prefs.node("Machine").get("CPUs", Integer.toString(0)));   
    }
     
    
    private void startProcess() throws InterruptedException, ExecutionException{
            this.eng = MatlabEngine.startMatlab(); 
//            Future<MatlabEngine> engFuture = MatlabEngine.startMatlabAsync();
//            this.eng = engFuture.get();
//            double tmp=0.0;
            this.eng.eval("cd '"+MatlabPath+"'"); 
                
            for (Main.curRun = 1; Main.curRun <= Main.totalRuns; Main.curRun++) {   
//                m.isSynchronous = true;
                this.start();
                while(this.isAlive());
            }
            
            eng.close();

    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        try {
            //Initialize variables - Thread - 1
            Main m = new Main();
            m.startProcess();
            
        } catch (IOException ioe) {
            ioe.printStackTrace();
        } catch (InterruptedException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        } catch (ExecutionException ex) {
            Logger.getLogger(Main.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    synchronized public void setGradient(Object gri){
        try {
            eng.putVariable("gri", gri);
        } catch (EngineException ee) {
            ee.printStackTrace();
        } catch (InterruptedException ie){
            ie.printStackTrace();
        } catch (ExecutionException eep){
            eep.printStackTrace();
        }
    }
    
    Object getGradient(int CPUno, String jsonIn){
        Object output = null;
        try{     
            MatlabEngine eng = MatlabEngine.startMatlab();
        } catch (EngineException ee) {
            ee.printStackTrace();
        } catch (InterruptedException ie){
            ie.printStackTrace();
        }
        return output;
    }

    private void prepareHistory(final String file_used, String algo) throws FileNotFoundException{
        File directory = new File ( MatlabPath+"\\Result\\" +algo) ;
        
        String tmp = "N"; //normal
        if(Main.useGSGD){
            tmp = "G";//guided
        }
        final String prefix = tmp;
        
        File [ ] filesInDir = directory.listFiles(
            new FileFilter() {
                public boolean accept(File pathname) {
                    return (pathname.getName().startsWith(file_used+"-"+prefix) 
                            && pathname.getName().endsWith("_history.csv"));
                }
            }
        );

        int fileNo = 1;
        if(filesInDir.length>0){
            fileNo = filesInDir.length+1;
        }
         
        runHistory = new PrintWriter(directory.getAbsolutePath() + "\\"+file_used+ "-"+prefix+fileNo+"_history.csv");
        runHistory.println("Ep"+","+"NFC"+","+"SR_test"+","+"MAXCPU"+","+"DELAY_TOLERANCE"+","+"MINI_BATCH");  
        
        outFile = directory.getAbsolutePath() + "\\"+file_used+ "-"+prefix+fileNo+"_trial.fig";
    }
    
    @Override
    public void start(){
        try {  
        //////                this.eng = MatlabEngine.startMatlab(); 
////////            Future<MatlabEngine> engFuture = MatlabEngine.startMatlabAsync();
////////            this.eng = engFuture.get();
////////            double tmp=0.0;
//////                eng.eval("cd '"+MatlabPath+"'");      
            
            if(Main.curRun == 1){
                eng.putVariable("data", null);
                eng.putVariable("file_path", null);
            }  
            eng.eval("ASGD_Initialize");
            
            String[] algos = eng.getVariable("algos");
            final int totalExpFiles = ((Double)eng.getVariable("totalExpFiles")).intValue();
            
            for (String iAlgo : algos) {
                eng.putVariable("iAlgo", iAlgo);
                eng.eval("ASGD_forAlgo");
                
                if(iAlgo.compareToIgnoreCase("RMSprop")!=0) //Canonical
                    continue;
                
                for (int ff = 1; ff <= totalExpFiles; ff++) {
                    eng.putVariable("ff", ff);

                    
                    eng.eval("ASGD_forFiles");
                    String ffFile = (String)eng.getVariable("file_used");
                  
                    //remove it later
                    if(ffFile.compareToIgnoreCase("Breast Cancer Diagnostic")!=0)
                        continue;
                    ////if (ff != 1) //remove it later
                    ////    continue; Now using the above code. 
    
                    
                    prepareHistory((String)eng.getVariable("file_used"), iAlgo);
                    
                    d = (int)((double)eng.getVariable("d"));
                    N = (int)((double)eng.getVariable("N"));
                    NC = (int)((double)eng.getVariable("NC")); 
                    
                    //some checks
                    if(DELAY_TOLERANCE*MINI_BATCH>=N){
                        System.err.println("Recheck Parameter values...");
                        System.err.println("Not appropriate for parallel execution. Try reducing the large values of DELAY_TOLERANCE etc.");
                        System.exit(0);
                    }
                    
                    //initialize with initial values for SGD paramters.
                    initSGDparameters = new SGDvariantsParameters(eng);
                    
                    DL = new DataLinks(this, this.MINI_BATCH, this.DELAY_TOLERANCE);
                    DL.start();
                    //<<for slave function 
                    setActivation((String)eng.getVariable("activation"));
                    setCurData((double[][])eng.getVariable("gbX"),
                            (double[][])eng.getVariable("inputVal"),
                            (double[][])eng.getVariable("valData"));  
                    setCurLabel((double[])eng.getVariable("gbY"),
                            (double[])eng.getVariable("givenOut"),
                            (double[])eng.getVariable("valOut"));
                    
                    DL.setSlaveFunction(activation, curData, curLabel);//redundant, already doing above
                    //>>

                    eng.putVariable("Epochs", Epochs*1.0);
                    System.out.println(((Double)eng.getVariable("Epochs")).intValue());
                    System.out.println("Algo: " + iAlgo + ", file: " + ff);      
                    
                    //Training Starts Here...
                    this.ep = 0;
                    this.Eout = 0.0;
                    //create threads here...
                    System.out.println("starting parameter servers...");
                    startParameterServers();
                    System.out.println("Slaves started...");                   

//                    while(ep<Epochs){
//                        System.out.println("Merge@ep: " + ep + "/" + Epochs);                                                             
//                        eng.eval("ASGD_whileEpoch");    
         
                        synchronized(this){
                            this.wait();
                        }            
//                    }                    
                    
                    System.out.println("\nFinally,");
                    Eout = this.eng.feval(DL.errFunc,
                    this.curData4Test,
                    this.curLabel4Test,
                    bestWsoFar,
                    DL.activation);                    

                    System.out.println("Eout: " + new DecimalFormat("#.####").format(Eout));
                    
                    Eout = this.eng.feval("getEout",
                    bestWsoFar,
                    this.curData4Test,
                    this.curLabel4Test,
                    DL.activation);

                    System.out.println("SR: " + new DecimalFormat("#.###").format(Eout));
                    System.out.println("NFC: " + psThread.NFC);
                    
                    //stop slave treads by interruption
//                    notifyParameterServers();
                    stopParameterServers(); 
                    
                    //set filename for bPlot
                    
                    eng.putVariable("outFile", outFile);
                    eng.eval("ASGD_bPlot");   
                    saveFinalResults();
                    
                    synchronized(DL){
                        DL.notify();
                    }
                    runHistory.close();
                }                    
            }

//            eng.close();         
        } catch (EngineException ee) {
            ee.printStackTrace();
        } catch (InterruptedException ie){
            ie.printStackTrace();
        } catch (MatlabSyntaxException mse){
            mse.printStackTrace();
        } catch (ExecutionException eep){
            eep.printStackTrace();
        } catch (FileNotFoundException fnf){
            fnf.printStackTrace();
        }
    }
    
    private void saveFinalResults(){
        try{
            double useG = 0.0;
            if(Main.useGSGD){
                useG = 1.0;
            }
            eng.putVariable("CPU", MAXCPU*1.0);
            eng.putVariable("DELAY_TOLERANCE", DELAY_TOLERANCE*1.0);
            eng.putVariable("Guided", useG);
            eng.putVariable("MINI_BATCH", MINI_BATCH*1.0);
            eng.putVariable("Epochs", Epochs*1.0);
            eng.putVariable("Eout", Eout*1.0);
            eng.putVariable("NFC", bestNFC*1.0);
            eng.eval("ASGD_printResults");
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
    
    private void startParameterServers(){
        psThread = new ParameterServer(this, DELAY_TOLERANCE);
////        psThread.start();
        psThread.runAsync();
        System.out.println("Parameter server started.");
    }
    
    
//    private void notifyParameterServers(){
//        if(psThread.getState() == Thread.State.WAITING);
//        //synchronized locks and unlocks a thread.
//        //in this block thread is said to "own" the lock. 
//        //At the point of Thread A wanting to acquire the lock, 
//        //if Thread B already owns the it, then Thread A must wait for 
//        //Thread B to release it.
//        synchronized(psThread){
//            psThread.notify();
//        }
//    }
    
    private void stopParameterServers(){               
        try {
//            psThread.join();
            psThread.thread.get();
        } catch (InterruptedException ex) {
            ex.printStackTrace();
        } 
        catch (ExecutionException ee){
            ee.printStackTrace();
        }      
    }
}
