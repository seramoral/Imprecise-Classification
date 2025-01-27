/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.credalClassifiers;

import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.Bagging;
import weka.core.AdditionalMeasureProducer;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.Utils;

/**
 *
 * @author Serafin
 */
public class CredalBaggingFlexibility extends CredalBagging{
     /**
   * It update the statistics for a given instance. It computes the non-dominated index set for the instance. 
   * It calculates how many times a state has been dominated
   * The non-dominated states are those for which the number of times that have been dominated is minimum
   * @param instance the instance
   * @throws NoSupportForMissingValuesException 
   */

    @Override
    protected void updateCredalStatistics(Instance instance) throws NoSupportForMissingValuesException {
        CredalDecisionTree2 tree;
        int num_classes = instance.numClasses();
        boolean[] non_dominated_states = new boolean[num_classes];
        boolean[] partial_non_dominated_states;
        boolean non_dominated;
        int[] non_dominated_index_set;
        int cont;
        int num_non_dominated_states;
        double[] times_dominated = new double[num_classes];
        double min_times_dominated, second_min_times_dominated, partial_times_dominated;
        double percentaje_difference;
        double max_percertange_difference = 0.1;
        boolean second_min_non_dominated;
        
        for(int i = 0; i < m_numTrees; i++){
            tree = (CredalDecisionTree2) m_bagger.getClassifier(i);
            partial_non_dominated_states = tree.computeNonDominatedStatesSet(instance);
            
            for(int j = 0; j < num_classes; j++){
                non_dominated = partial_non_dominated_states[j];
                
                if(!non_dominated)
                   times_dominated[j]++;
            }
        }

        min_times_dominated = times_dominated[0];
        
        for(int j = 1; j < num_classes; j++){
            partial_times_dominated = times_dominated[j];
            
            if(partial_times_dominated < min_times_dominated)
                min_times_dominated = partial_times_dominated;
        }
        
        second_min_times_dominated = Double.POSITIVE_INFINITY;
        
        for(int j = 0; j < num_classes; j++){
            partial_times_dominated = times_dominated[j];
            
            if(partial_times_dominated > min_times_dominated){ // Not the min
                if(partial_times_dominated < second_min_times_dominated)
                    second_min_times_dominated = partial_times_dominated;
            }
        }
        
        percentaje_difference = (second_min_times_dominated-min_times_dominated)/m_numTrees;
        second_min_non_dominated = percentaje_difference <= max_percertange_difference;
        
        for(int j = 0; j < num_classes; j++){
            partial_times_dominated = times_dominated[j];
            
            if(partial_times_dominated == min_times_dominated)
                non_dominated_states[j] = true;
            
            else if(partial_times_dominated == second_min_times_dominated){
                non_dominated_states[j] =  second_min_non_dominated;
            }
            
            else
                non_dominated_states[j] = false;  
        } 
        
        num_non_dominated_states = 0;
        
        for(int j = 0; j < num_classes; j++){
            non_dominated = non_dominated_states[j];
            
            if(non_dominated)
                num_non_dominated_states++;
        }
        
        non_dominated_index_set = new int[num_non_dominated_states];
        cont = 0;
        
        for(int j = 0; j < num_classes; j++){
            non_dominated = non_dominated_states[j];
            
            if(non_dominated){
                non_dominated_index_set[cont] = j;
                cont++;
            }
        }
        
        this.updateStatistics(non_dominated_index_set, instance);
        
    }

}
