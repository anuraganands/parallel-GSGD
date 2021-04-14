/*
 * @author Anurag sharma 2012.
 */
package javathreads;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.Queue;
import javax.naming.SizeLimitExceededException;
/**
 *
 * @author sharma_au
 */
public class MyQueue<T> implements Iterable<T> {
    private int maxSize;
    private Queue<T> q;

    public Iterator<T> iterator() {
        return q.iterator();
    }
    
    
    
    public MyQueue(LinkedList<T> input){
        this.maxSize = input.size();
        q = input;        
    }
    
    public MyQueue(int size){
        this.maxSize = size;
        q = new LinkedList<T>();        
    }
    
    public MyQueue(){
        this.maxSize = -1; //indicates free of size
        q = new LinkedList<T>();  
    }
    
    /**
     * applicable for fixed sized queue only
     * @param item
     * @throws SizeLimitExceededException 
     */
    public void forceEnqueueByDequeue(T item){// throws SizeLimitExceededException{
        if(capacity() == -1){
            throw new UnsupportedOperationException("Use T tryPush(T item) method");
        }
        if(this.capacity() == 0){
            return;
        }
        if(q.size()<this.capacity()){                        
            q.add(item); 
        } else {
            this.dequeue();
            q.add(item);//most likely it will not throw again
        }
    }
    
    public void enqueue(T item){
        q.add(item);
    }
    
    public T dequeue(){
        return q.poll();
    }    
    
    /**
     * gets the current size based on the number of elements in the queue.
     * @return 
     */
    public final int curSize(){
        return q.size();
    }
    
    /**
     * remove all elements permanently
     */
    public void clearAll(){
        while(dequeue()!=null){
            ;
        }
    }
    
    /**
     * The maximum elements that this queue can hold. If the returned value is
     * negative then size is <B>unlimited</B>.
     * @return 
     */
    public final int capacity(){
        return this.maxSize;
    }

    @Override
    public String toString() {
        return q.toString();
    }
    
}
