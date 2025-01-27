/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utils;

import java.io.Serializable;
import weka.core.E_ContingencyTables;
import static weka.core.E_ContingencyTables.NPITransformation;
import weka.core.Utils;

/**
 *
 * @author Serafin
 */
public class DiscreteEstimatorCostSensitiveNPI implements Serializable{
    
    /** for serialization */
    static final long serialVersionUID = 2889730616939923301L;
    
    /** Holds the counts */
    private double[] counts;
    
    /** Hold the sum of counts */
    private double sum_of_counts;
    

    
    /**
    * Creates a new object
    * @param num_counts number of possible values of the discrete attribute
    * 
    */
  
    public DiscreteEstimatorCostSensitiveNPI(int num_counts){
        counts = new double[num_counts];
        sum_of_counts = 0;
        
    }
    
    /**
    * Increments the count of a given component in a certain vaue
    * @param component Component to increment
     * @param value the value to increment
    */
  
    public void addComponent(int component, double value){
        counts[component]+=value;
        sum_of_counts+=value;
    }
    
      
    /**
    * Obtains the inferior probability of a given component accoding to the A-NPI-M
    * considering the weight of the class value
    * @param component component from which obtain the inferior probability
     * @param weight the weight of the class value
    * @return inferior probability
    */
  
    public double getInferiorProbability(int component){
        double count = counts[component];
        double inferior_probability;
        double arrangement_lower_probability;
        
        arrangement_lower_probability = count - 1;
        inferior_probability = arrangement_lower_probability/sum_of_counts;
        
        if(inferior_probability < 0)
            inferior_probability = 0;
      
      
        return inferior_probability;
    }
  
    /**
   * Obtains the superior probability of a given component accoding to the IDM
   * @param component component from which obtain the inferior probability
    * @param weight the weight of the class value
   * @return inferior probability
   */
  
    public double getSuperiorProbability(int component){
        double count = counts[component];
        double superior_probability;
        double arrangement_upper_probability;
               
        arrangement_upper_probability = count + 1;
        //numerator = arrangement_upper_probability*weight;
        superior_probability = arrangement_upper_probability/sum_of_counts;
        
        if(superior_probability > 1)
            superior_probability = 1;
      
        return superior_probability;
    }
    
  
    
    /**
     * It computes the distribution with maximum entropy
     * via the algorithm for computing the maximum entropy with the A-NPI-M
     * @return the array with the distribution of maximum entropy with the A-NPI-M
     */
    
    public double[] distributionMaxEntropy(){
        double[] distribution_max_entropy;
        int num_counts;
        double uniform_probability;
        
        if(sum_of_counts == 0){
            num_counts = counts.length;
            distribution_max_entropy = new double[num_counts];
            uniform_probability = 1/num_counts;
            
            for(int i = 0; i < num_counts; i++)
                distribution_max_entropy[i] = uniform_probability;
        }
        
        else{
            distribution_max_entropy = E_ContingencyTables.NPITransformation(counts);
            Utils.normalize(distribution_max_entropy);
        }
        
        return distribution_max_entropy;
    }
    
    /**
   * Get the count for a value
   *
   * @param component the value to get the count of
   * @return the count of the supplied value
   */
    public double getCount(int component) {
        double count;
      
        if (sum_of_counts == 0) 
            count = 0;
    
        else
            count = counts[component];
    
        return count;
    }
  

   /**
   *
   * @return the number of possible values of the attribute 
   */
    public int getNumSymbols() {
        int num_counts = counts.length;
    
        return num_counts; 
    }
    
    public static void main(String[] args) {
        DiscreteEstimatorCostSensitiveNPI estimator = new DiscreteEstimatorCostSensitiveNPI(4);
        estimator.addComponent(0,4);
        estimator.addComponent(1,2);
        estimator.addComponent(2,5);
        estimator.addComponent(3,5);
        estimator.distributionMaxEntropy();
        System.out.println("Fin");
    }

}
