/*
 * NodeTree.java
 *
 * Created on 24 June 2008, 18:10
 *
 * To change this template, choose Tools | Template Manager
 * and open the template in the editor.
 */

package weka.classifiers.trees;

import java.io.Serializable;

import weka.core.*;


/**
 *
 * @author Andres
 */
public class NodeTree implements Serializable{
    
    /** for serialization */
    static final long serialVersionUID = 2889730616939923301L;

    public boolean[] attSelected;
    
    /** The node's successors. */ 
    private NodeTree[] m_Successors;

    /** Attribute used for splitting. */
    private Attribute m_Attribute;

    /** Class distribution if node is leaf. */
    private double[] m_Frequency;

    /** Class distribution if node is leaf. */
    private double[] m_Distribution=null;
    
    private double m_Support;
    
    /** Creates a new instance of NodeTree */
    public NodeTree() {
    }
    
    
    public void setAttribute(Attribute att){
        this.m_Attribute=att;
    }
    
    public Attribute getAttribute(){
        return this.m_Attribute;
    }

    public double getMaxDistributionIndex(){
        return Utils.maxIndex(this.m_Frequency);
    }
    /*
    protected void setClassValue(double value){
        this.m_ClassValue=value;
    }
    */
    public double getClassValue(){
        if (this.getSupport()==0)
            return Double.NaN;
        return Utils.maxIndex(this.m_Frequency);
    }
    /*
    protected void setClassAttribute(Attribute value){
        this.m_ClassAttribute=value;
    }
    
    protected Attribute getClassAttribute(){
        return this.m_ClassAttribute;
    }
*/
    public void setDistribution(double[] value){
        this.m_Distribution=value;
    }
    
    public double[] getDistribution(){
        if (this.m_Distribution==null){
            double[] distribution=new double[this.m_Frequency.length];
            if (this.getSupport()==0)
                return distribution;
            System.arraycopy(this.m_Frequency,0,distribution,0,this.m_Frequency.length);
            Utils.normalize(distribution);
            return distribution;
        }else
            return this.m_Distribution;
        
    }
     
    public void setFrequency(double[] value){
        this.m_Frequency=value;
    }
    
    public double[] getFrequency(){
        return this.m_Frequency;
    }
    
   
    public void setRealSupport(double value){
        this.m_Support=value;
    }
   
    public double getRealSupport(){
        return this.m_Support;
    }
    
    
    public double getSupport(){
        int cont=0;
        for (int i=0; i<this.m_Frequency.length; i++)
                cont+=this.m_Frequency[i];
        return cont;
    }
     
    /*
    protected void normalize(){
        Utils.normalize(this.m_Distribution);
    }
    */
    public void setSuccesors(int i, NodeTree value){
        if (this.m_Successors==null)
            this.m_Successors=new NodeTree[this.m_Attribute.numValues()];
        this.m_Successors[i]=value;
    }
    
    public NodeTree getSuccesors(int i){
        if (this.m_Successors==null)
            return null;
        return this.m_Successors[i];
    }
    
    public NodeTree getMaxFrequencySuccesors(){
        if (this.m_Successors==null)
            return null;
        double maxsupport=this.m_Successors[0].getRealSupport();
        int index=0;
        for (int i=1; i<this.m_Successors.length; i++)
            if (maxsupport<this.m_Successors[i].getRealSupport()){
                maxsupport=this.m_Successors[i].getRealSupport();
                index=i;
            }
                
        return this.m_Successors[index];
    }

}
