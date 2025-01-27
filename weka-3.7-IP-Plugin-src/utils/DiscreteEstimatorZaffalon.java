/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package utils;

import java.io.Serializable;

/**
 *
 * @author Serafin
 */
public class DiscreteEstimatorZaffalon implements Serializable{
    
       /** for serialization */
    static final long serialVersionUID = 2889730616939923301L;
    
/** Holds the counts */
    private double[] counts;
    
     /** Hold the sum of counts */
  private double sum_of_counts;
  
  /** the s valus for IDM **/
  
  private double s_value;
  
  /**
   * Creates a new object
   * @param num_counts number of possible values of the discrete attribute
   * @param s The s parameter for IDM 
   */
  
  public DiscreteEstimatorZaffalon(int num_counts, double s){
      counts = new double[num_counts];
      sum_of_counts = s;
      s_value = s;
  }
  
    /**
    * Increments the count of a given component in one
    * @param component Component to increment
    */
  
    public void addComponent(int component){
        counts[component]+=1.0;
        sum_of_counts+=1.0;
    }
  
  /**
   * Obtains the inferior probability of a given component accoding to the IDM
   * @param component component from which obtain the inferior probability
   * @return inferior probability
   */
  
  public double getInferiorProbability(int component){
      double count = counts[component];
      double inferior_probability = count/sum_of_counts;
      
      return inferior_probability;
  }
  
    /**
   * Obtains the superior probability of a given component accoding to the IDM
   * @param component component from which obtain the inferior probability
   * @return inferior probability
   */
  
    public double getSuperiorProbability(int component){
      double count = counts[component];
      double numerator = count + s_value;
      double superior_probability = numerator/sum_of_counts;
      
      return superior_probability;
  }
    
   /**
   *
   * @return the number of possible values of the attribute 
   */
  public int getNumSymbols() {
      int num_counts = counts.length;
    
      return num_counts; 
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
   * Get the sum of all the counts
   *
   * @return the total sum of counts
   */
  public double getSumOfCounts() {
    
    return sum_of_counts;
  }
  
    
}
