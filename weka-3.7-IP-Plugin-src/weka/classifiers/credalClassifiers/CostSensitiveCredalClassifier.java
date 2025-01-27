/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.credalClassifiers;

import java.util.Enumeration;
import java.util.Vector;
import weka.core.E_ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

/**
 *
 * @author Serafin
 */
public abstract class CostSensitiveCredalClassifier extends CredalClassifier{
    /**
    * Option for the matrix of cost errors
    * 0 = matrix 0/1 (all the errors have the same cost)
    * 1 = Cost Matrix (I): Cost depends on the real class value. Class with the lowest frequency highest cost
    * 2 = Cost Matrix (II): Cost depends on the predicted class value. Class with the lowest frequency highest cost
    * 3 = Cost Matrix (III): Cost depends on the real class value. Class with the highest frequency highest cost
    * 4 = Cost Matrix (IV): Cost depends on the predicted class value. Class with the highest frequency highest cost
    */
  
   protected int option_matrix_errors = 1;
    
       /* Matrix of costs of errors for cost-sensitive problems
        mij = cost of predicting the i-th class value when the
        real class value is the j-th value
    */ 
    protected double[][] matrix_cost_errors;
    
      /* Vector with the weights for each class value */
    
    double[] class_weights;
    
         /**
     * computes the matrix of cost errors 
     * depending on the relative frequencies in the training set
     * @param data the training data
     */
    
    protected void computeMatrixCostErrors(Instances data){
        int num_class_values = data.numClasses();
        matrix_cost_errors = new double[num_class_values][num_class_values];
        double[] relative_frequencies = new double[num_class_values];
        Instance instance;
        int class_value, class_value_highest_frequency;
        int[] index_ordered_relative_frequencies;
        int num_instances = data.numInstances();
        
        for(int i = 0; i < num_instances; i++){
            instance = data.instance(i);
            class_value = (int) instance.classValue();
            relative_frequencies[class_value]++;
        }
        
        index_ordered_relative_frequencies = Utils.sort(relative_frequencies);
        
        switch(option_matrix_errors){
            
            // Cost Matrix 0/1
            
            case 0:
                for(int i = 0; i < num_class_values; i++){
                    for(int j = 0; j < num_class_values; j++)
                        if(i != j)
                            matrix_cost_errors[i][j] = 1;
                }
                break;
                
            /* Cost Matrix I    
                Cost depending on the real class value
                The cost is lower as higher is the relative frequency  
            */
                
            case 1:
                
                for(int i = 0; i < num_class_values; i++){
                    class_value_highest_frequency = index_ordered_relative_frequencies[i];
                    
                    for(int j = 0; j < num_class_values; j++){
                        if(j != class_value_highest_frequency){
                            matrix_cost_errors[j][class_value_highest_frequency] = num_class_values - i;     
                        }
                    }
                            
                }
                break;
                       
            /* Cost Matrix II
               Cost depending on the real class value
                The cost is lower as lower is the relative frequency  Cost depending on the real class value
                
            */
                
            case 2:
                
                for(int i = 0; i < num_class_values; i++){
                    class_value_highest_frequency = index_ordered_relative_frequencies[i];
                    
                    for(int j = 0; j < num_class_values; j++){
                        if(j != class_value_highest_frequency){
                            matrix_cost_errors[class_value_highest_frequency][j] = num_class_values - i;     
                        }
                    }
                            
                }
                break;
                
            /* Cost Matrix III
              Cost depending on the predicted value
            The cost is lower as higher is the relative frequency  
                
            */    
            case 3:
                
                for(int i = 0; i < num_class_values; i++){
                    class_value_highest_frequency = index_ordered_relative_frequencies[i];
                    
                    for(int j = 0; j < num_class_values; j++){
                        if(j != class_value_highest_frequency){
                            matrix_cost_errors[j][class_value_highest_frequency] = i+1;     
                        }
                    }
                            
                }
                break;
                
            /* Cost Matrix IV 
            Cost depending on the predicted value
            The cost is higher as higher is the relative frequency
            */   
             
            case 4:
                
                for(int i = 0; i < num_class_values; i++){
                    class_value_highest_frequency = index_ordered_relative_frequencies[i];
                    
                    for(int j = 0; j < num_class_values; j++){
                        if(j != class_value_highest_frequency){
                            matrix_cost_errors[class_value_highest_frequency][j] = i+1;     
                        }
                    }
                            
                }
                
                break;
                
            // By default, 0/1 cost matrix.     
                
            default:
                
                for(int i = 0; i < num_class_values; i++){
                    for(int j = 0; j < num_class_values; j++)
                        if(i != j)
                            matrix_cost_errors[i][j] = 1;
                }
                
                break;
             
        }
        
        this.stats.initializeCostErrors(matrix_cost_errors);
        
    }
    
    /**
    * Computes the non dominated states set from a given set of risk intervals
    * It uses the stochastic dominance criterion on the intervals: ci dominated c_j
    * if, and only if, upper risk of ci \leq upper risk of cj
    * @param lower_risks the lower risks
    * @param upper_risks the upper risks
    * @return an array indicating whether each class value is dominated
    */
  
    protected boolean[] computeNonDominatedStatesRisks(double[] lower_risks, double[] upper_risks){
        int num_classes = lower_risks.length;
        boolean[] non_dominated_set = new boolean[num_classes];
        double lower_risk, upper_risk;
        double min_upper_risk;
        boolean non_dominated;
        boolean all_dominated;
        int k;
        
        for(int i = 0; i < num_classes; i++){
            lower_risk = lower_risks[i];
            min_upper_risk = Double.POSITIVE_INFINITY;
            
            for(int j = 0; j < num_classes; j++){
                if(j != i){
                    upper_risk = upper_risks[j];
                
                    if(upper_risk < min_upper_risk)
                        min_upper_risk = upper_risk;
                }
            }
            
            non_dominated = min_upper_risk > lower_risk;
            non_dominated_set[i] = non_dominated;
        }
        
        all_dominated = true;
        k = 0;
        
        while(all_dominated && k < num_classes){
            non_dominated = non_dominated_set[k];
            
            if(non_dominated)
                all_dominated = false;
            
            else
                k++;
        }
        
        /* if all the states are dominated, then all of them are non_dominated */
        
        if(all_dominated){
            for(int i = 0; i < num_classes; i++)
                non_dominated_set[i] = true;
        }
        
        return non_dominated_set;
    }
    
    /**
     * It computes the risks from the probabilities via Bayes decision rule
     * @param probabilities the probabilities
     * @return the risks
     */
    
    protected double[] getRiskFromProbabilities(double[] probabilities){
        int num_classes = probabilities.length;
        double[] risks = new double[num_classes];
        double risk;
        double probability, cost_error;
        
         for(int i = 0; i < num_classes; i++){
            risk = 0;
            
            for(int j = 0; j < num_classes; j++){
                cost_error = matrix_cost_errors[i][j];
                probability = probabilities[j];
                risk+= cost_error*probability;
            }
            
            risks[i] = risk;
        }       
        
        return risks;
    }
    
    /**
     * Compute the cost of misclassifying each class value from the cost matrix.
     * mij = cost of predicting c_i when the real class value is c_j 
     * cost(j) = \sum_{i = 1}^{K}m_ij
     * @return {cost(1), cost(2),...,cost(K)}
     */
    
    protected double[] computeClassCosts(){
        int num_class_values = matrix_cost_errors.length;
        double[] cost_errors = new double[num_class_values];
        double cost, sum_cost;
        
        for(int j = 0; j < num_class_values; j++){
            sum_cost = 0;
            
            for(int i = 0; i < num_class_values; i++){
                cost = matrix_cost_errors[i][j];
                sum_cost+=cost;
            }
            
            cost_errors[j] = sum_cost;
        }
        
        
        return cost_errors;
    }
    
        /**
     * Compute the cost of misclassifying each class value from the cost matrix 
     * andpredicting c_i when the real class value is c_j  considering the probability of each misclassification
     * mij = cost of predicting c_i when the real class value is c_j 
     * pij = cost of predicting c_i when the real class value is c_j 
     * cost(j) = \sum_{i = 1}^{K}m_ijxp_ij
     * @param misclassification_probabilities matrix of misclassification probabilities
     * @return {cost(1), cost(2),...,cost(K)}
     */
    
    protected double[] computeClassCostsFromProbabilities(double[][] misclassification_probabilities){
        int num_class_values = matrix_cost_errors.length;
        double[] cost_errors = new double[num_class_values];
        double cost, sum_cost;
        
        for(int j = 0; j < num_class_values; j++){
            sum_cost = 0;
            
            for(int i = 0; i < num_class_values; i++){
                cost = matrix_cost_errors[i][j]*misclassification_probabilities[i][j];
                sum_cost+=cost;
            }
            
            cost_errors[j] = sum_cost;
        }
        
        
        return cost_errors;
    }
    
    
     /**
   * Computes the weights corresponding to the class values given de misclassification mosts
   * Let C_i the cost of error associated with the i-th class value
   * Let (n_1,...,n_K) be the array with the relative frequencies in the training set
   * Let ('n_1,...,n'_K) be the arrangement associated with the probability distribution 
   * that reaches the maximom of entropy with the A-NPI-M
   * wj = Nxc_j/(sum_{i = 1}^{K}n'_ixC_i)
   * @param data the training instances
   * @param misclassification_costs the misclassification costs {C_1,C_2,...,C_K}
   */
    
    protected void computeClassWeightsFromCosts(Instances data, double[] misclassification_costs){
        int num_classes = data.numClasses();
        double num_instances = data.numInstances();
        double[] class_frequencies = new double[num_classes];
        Instance instance;
        int class_value;
        double[] NPI_transformation;
        double class_frequency, weighted_frequency;
        double sum_weighted_frequencies;
        double cost;
        double numerator;
        double weight;
                
        class_weights = new double[num_classes];
        
        Enumeration instEnum = data.enumerateInstances();
       
        
        while (instEnum.hasMoreElements()) {
            instance = (Instance) instEnum.nextElement();
            class_value = (int) instance.classValue();
            class_frequencies[class_value]++;
        }
        
        /* Compute('n_1,...,n'_K) the arrangement associated with the probability distribution 
        * that reaches the maximom of entropy with the A-NPI-M */
        
        NPI_transformation = E_ContingencyTables.NPITransformation(class_frequencies);
        
        for(int j = 0; j < num_classes; j++)
            NPI_transformation[j]*= num_instances;
        
        // Compute (sum_{i = 1}^{K}n'_ixC_i
        
        sum_weighted_frequencies = 0;
        
        for(int j = 0; j < num_classes; j++){
            class_frequency = NPI_transformation[j];
            cost = misclassification_costs[j];
            weighted_frequency = class_frequency*cost;
            sum_weighted_frequencies+=weighted_frequency;
        }
        
        /* Calculate wj =  Nxc_j/(sum_{i = 1}^{K}n'_ixC_i)*/
        
        for(int j = 0; j < num_classes; j++){
            cost = misclassification_costs[j];
            numerator = cost*num_instances;
            weight = numerator/sum_weighted_frequencies;
            class_weights[j] = weight;
        }

    }
    
        /**
   * Computes the weights corresponding to the class values
   * 
   * @param data the training instances
   */
    
    protected void computeClassWeights(Instances data){
        double[] class_costs = computeClassCosts();
        
        computeClassWeightsFromCosts(data, class_costs);
            
    }
    
        /**
   * Computes the weights corresponding to the class values considering the probability of misclasifying each class value
   * Computes the class costs considering the misclassification probabilities
   * Compute the weights for the instances from such costs
   * @param data the training instances
   * @param misclassification_probabilities the probabilities of miclassifying each class value
   */
    
    protected void computeClassWeightsFromProbabilities(Instances data, double[][] misclassification_probabilities){
        double[] class_costs = computeClassCostsFromProbabilities(misclassification_probabilities);
        
        computeClassWeightsFromCosts(data, class_costs);
            
    }
    
    /**
     * It computes the weighted frequencies given the frequencies of the class values n_1, n_2,...,n_k
     * w_i \times n_i \forall i = 1,2,...,K
     * @param frequencies the frequencies of the class values n_1,n_2,...,n_k
     * @return w_i \times n_i \forall i = 1,2,...,K
     */
    
    protected double[] computeWeightedFrequencies(double[] frequencies){
        int num_class_values = frequencies.length;
        double[] weighted_frequencies = new double[num_class_values];
        double frequency;
        double weight;
        
        for(int k = 0; k < num_class_values; k++){
            frequency = frequencies[k];
            weight = class_weights[k];
            weighted_frequencies[k] = frequency*weight;
        }
        
        return weighted_frequencies;
    }
    
    /**
     Compute the non-dominated states set from the lower and upper probabilities
     * It utilizes the stochastic dominance criterion, according to which ci is non-dominated iif
     * w\\overline{p_i} > max_{j = 1,...K j \neq i} w\\underline{p_j}, \forall i = 1,...,K
     * @param lower_probabilities the lower probabilities
     * @param upper_probabilities the upper probabilities
     * @return an array indicating whether each class value is dominated
     */
    
    protected boolean[] computeNonDominatedSetFromProbabilities(double[] lower_probabilities, double[] upper_probabilities){
        int num_class_values = lower_probabilities.length;
        boolean[] non_dominated_set = new boolean[num_class_values];
        boolean non_dominated;
        double lower_probability, upper_probability, max_lower_probability;
        boolean all_dominated;
        int i;
        
        for (int k=0; k < num_class_values; k++){
            upper_probability = upper_probabilities[k];
            max_lower_probability = Double.NEGATIVE_INFINITY;
            
            for(int j = 0; j < num_class_values; j++){
                if(j!=k){
                    lower_probability = lower_probabilities[j];
                    
                    if(lower_probability > max_lower_probability)
                        max_lower_probability = lower_probability;
                }
            }
            
            non_dominated = upper_probability > max_lower_probability;
            non_dominated_set[k] = non_dominated;
        }
        
        all_dominated = true;
        i = 0;
        
        while(all_dominated && i < num_class_values){
            non_dominated = non_dominated_set[i];
            
            if(non_dominated)
                all_dominated = false;
            
            else
                i++;
        }
        
         if(all_dominated){
            for(int k = 0; k < num_class_values; k++)
                non_dominated_set[k] = true;
        }
      
        return non_dominated_set;
    }
    
    public int getOptionMatrixErrors(){
        return option_matrix_errors;
    }
    
    public void setOptionMatrixErrors(int option){
        option_matrix_errors = option;
    }
    
    /**
    * Gets the current settings of the classifier.
    *
    * @return an array of strings suitable for passing to setOptions()
    */
    
    public String[] getOptions() {
        Vector result;
        String[] options;
        int i;
    
        result = new Vector();
    
        result.add("-c");
        result.add("" + getOptionMatrixErrors());
    
        options = super.getOptions();
    
        for (i = 0; i < options.length; i++)
            result.add(options[i]);
    
    
        return (String[]) result.toArray(new String[result.size()]);
    }
    
    /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options
   */
  
    public Enumeration listOptions() {
        Vector newVector = new Vector();

        newVector.addElement(new Option(
	"\tOption for the cost matrix",
	"c", 1, "-c <option of cost matrix>"));
    
        Enumeration enu = super.listOptions();
    
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }

        return newVector.elements();
    }
    
    /**
   * Parses a given list of options. <p/>
   * 
   * 
   <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
    
    
    public void setOptions(String[] options) throws Exception{
        String tmpStr;
    
        tmpStr = Utils.getOption('c', options);
    
        if (tmpStr.length() != 0) {
            option_matrix_errors = Integer.parseInt(tmpStr);
        
        } else {
            option_matrix_errors = 1;
        }
    
        super.setOptions(options);
    
        Utils.checkForRemainingOptions(options);
    }  
    
}
