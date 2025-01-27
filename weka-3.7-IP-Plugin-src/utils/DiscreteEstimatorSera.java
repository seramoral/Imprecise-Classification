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
public class DiscreteEstimatorSera  implements Serializable{ 
       /** Holds the counts */
    private double[] counts;
    
     /** Hold the sum of counts */
  private double sum_of_counts;
  
  /** the m valus for m-estimates **/
  
  private double m_value;
  
    /**
   * Creates a new object
   * @param num_counts number of possible values of the discrete attribute
   * @param m The m parameter for the m-estimates 
   */
  
  public DiscreteEstimatorSera(int num_counts, double m){
      counts = new double[num_counts];
      sum_of_counts = m;
      m_value = m;
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
   * Obtains the inferior or superior probability of a given componen given the inferior or superior probability according to the IDM
   * @param component component from which obtain the inferior probability
   * @return inferior probability
   */
  
  public double getProbability(int component, double prior_probability){
      double count = counts[component];
      double prior_component = m_value*prior_probability;
      double numerator = count + prior_component;
      double probability = numerator/sum_of_counts;
      
      return probability;
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
