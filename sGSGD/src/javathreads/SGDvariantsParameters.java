/*
 * @author Anurag sharma 2019.
 */
package javathreads;

import com.mathworks.engine.EngineException;
import com.mathworks.engine.MatlabEngine;
import com.mathworks.engine.MatlabSyntaxException;
import java.util.concurrent.ExecutionException;
import org.json.JSONObject; //Json

/**
 *
 * @author sharma_au
 */
public class SGDvariantsParameters {
    MatlabEngine eng;
    
    public Object eta;
    public Object iAlgo;
    public Object sSGD;
    public Object rSGD;
    public Object gradHistSGD;
    public JSONObject WsgdHistory;
    public Object mAdamSGD;
    public Object vAdamSGD;
    
    private SGDvariantsParameters(){;}

    public SGDvariantsParameters(MatlabEngine inEng) {
        eng = inEng;
        getVariables();//set initial values
    }
    
    /**
     * get variables into Java
     */
    public void getVariables(MatlabEngine ...inEng){
        try{
            MatlabEngine localEng;
            if(inEng.length == 0){
                localEng = eng;
            }else{
                localEng = inEng[0];                
            }
            
            eta = localEng.getVariable("eta");
            iAlgo = localEng.getVariable("iAlgo");
            sSGD = localEng.getVariable("sSGD");
            rSGD = localEng.getVariable("rSGD");
            gradHistSGD = localEng.getVariable("gradHistSGD");
            
            localEng.eval("Wjson = cell2json(WsgdHistory);");
            WsgdHistory = new JSONObject((String)localEng.getVariable("Wjson"));
            
//            WsgdHistory = localEng.getVariable("WsgdHistory");
            
            mAdamSGD = localEng.getVariable("mAdamSGD");
            vAdamSGD = localEng.getVariable("vAdamSGD");       
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
    
    /**
     * put variables into Matlab space
     * Usaage: putVariables(eng); or putVariable(another_eng);<BR>
     * <B>NOTE:</B> the linked engine might change after using this method.
     */
    public void putVariables(MatlabEngine ...inEng){
        try{
            MatlabEngine localEng;
            if(inEng.length == 0){
                localEng = eng;
            }else{
                localEng = inEng[0];
                this.eng = localEng;
            }
            
            localEng.putVariable("eta", eta);
            localEng.putVariable("iAlgo", iAlgo);
            localEng.putVariable("sSGD", sSGD);
            localEng.putVariable("rSGD", rSGD);
            localEng.putVariable("gradHistSGD", gradHistSGD);
            localEng.putVariable("WsgdHistory", WsgdHistory.toString());
//            CellStr tmp = new CellStr(new String[]{"One", "Two", "Three"});?
//            localEng.putVariable("WsgdHistory", tmp);
            
            localEng.eval("WsgdHistory = json2cell(char(WsgdHistory));");
            
            localEng.putVariable("mAdamSGD", mAdamSGD);
            localEng.putVariable("vAdamSGD", vAdamSGD);       
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
