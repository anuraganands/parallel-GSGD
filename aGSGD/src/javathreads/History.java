/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package javathreads;

/**
 *
 * @author sharma_au
 */
public class History {
    public int []batch;
    public double [][]xi;
    public double []yi;
    static int batIndx = -1;
    private int ID = -1;
    public void generateID(){
        ID = batIndx;
    }
    public int getID(){
        return ID;
    }
    public static void batchAdd(){
        batIndx++;
    }
    public static void batchReset(){
        batIndx = -1;
    }
}
