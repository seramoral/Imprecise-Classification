/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.credalClassifiers;

import java.util.Enumeration;
import java.util.Vector;
import utils.DiscreteEstimatorCostSensitiveNPI;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Utils;

/**
 *
 * @author Serafin
 */
public class CostSensitiveNaiveBayes_Iterative extends CostSensitiveNaiveBayes{
   
    /** The number of iterations for estimating misclassification probabilities */
    int num_iterations = 1; 
    
    /** The number of folds of cross validation  */
    int num_folds = 10;
    
        /**
   * Returns a string describing this classifier
   * @return a description of the classifier suitable for
   * displaying in the explorer/experimenter gui
   */
    @Override
    public String globalInfo() {
        return "Class for an imprecise Naive Bayes classifier using estimator classes. Numeric"
        +" It iteratively considers the error costs, which are computed in each iteration.\n\n"
        +"via the current miclassification probabilities "        
        +"For more information on Naive Bayes classifiers, see\n\n"
        + getTechnicalInformation().toString();
    }
    
    /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
    
    public String numIterationsTipText(){
        return "The number of iterations for computing the weights for the instances \n\n" + " via misclassification probabilities";
    }
    
    /**
     * @return the number of iterations 
     */
    
    public int getNumIterations(){
        return num_iterations;
    }
    /**
     * Sets the number of iterations
     * @param n_iterations the number of iterations
     */
    
    public void setNumIterations(int n_iterations){
        num_iterations = n_iterations;
    }
    
    /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
    
    public String numFoldsTipText(){
        return "The number of folds for cross validation in each iteration";
    }
    
    /**
     * Set the number of folds in cross validation
     * @param n_folds the number of folds
     */
    
    public void setNumFolds(int n_folds){
        num_folds = n_folds;
    }
    
    /**
     * 
     * @return the number of cross validation folds 
     */
    
    public int getNumFolds(){
        return num_folds;
    }
    
    /**
     * It makes a prediction of a given mode for a given test instance
     * It considers the maximum entropy probability distributions
     * @param model the given model
     * @param test_instance the test instance
     * @return the predicted non-dominated states set by the model for the instance
     */
    
    private int[] makePrediction(CostSensitiveNaiveBayes model, Instance test_instance){
        int num_classes = test_instance.numClasses();
        int class_index = test_instance.classIndex();
        int num_attributes = test_instance.numAttributes();
        int attribute_value;
        DiscreteEstimatorCostSensitiveNPI model_class_distribution = model.getClassDistribution();
        DiscreteEstimatorCostSensitiveNPI model_conditioned_distribution;
        double[] inferior_probabilities = new double[num_classes];
        double[] superior_probabilities = new double[num_classes];
        double class_inferior_probability, class_superior_probability;
        double inferior_probability, superior_probability;
        double partial_inferior_probability, partial_superior_probability;
        double max_inferior = -Double.MAX_VALUE;
        double max_superior = -Double.MAX_VALUE;
        boolean[] non_dominated_set;
        int[] non_dominated_index_set;
        
        for(int j = 0; j < num_classes; j++){
            class_inferior_probability = model_class_distribution.getInferiorProbability(j);
            class_superior_probability = model_class_distribution.getSuperiorProbability(j);
            inferior_probability = class_inferior_probability;
            superior_probability = class_superior_probability;
            
            for(int i = 0; i < num_attributes; i++){
                if(i!=class_index){
                    attribute_value = (int)test_instance.value(i);
                    model_conditioned_distribution = model.getConditionedDistribution(i, j);
                    partial_inferior_probability = model_conditioned_distribution.getInferiorProbability(attribute_value);
                    partial_superior_probability = model_conditioned_distribution.getSuperiorProbability(attribute_value);
                    inferior_probability*=partial_inferior_probability;
                    superior_probability*=partial_superior_probability;
                }
            }
            
            if(inferior_probability > max_inferior)
                max_inferior = inferior_probability;
          
            if(superior_probability > max_superior)
                max_superior = superior_probability;
          
            inferior_probabilities[j] = inferior_probability;
            superior_probabilities[j] = superior_probability;
            
                
        }
        
        if ((max_inferior > 0) && (max_inferior < 1e-75)) { // Danger of probability underflow
            for (int j = 0; j < num_classes; j++) 
                inferior_probabilities[j] *= 1e75;
        }
      
        if ((max_superior > 0) && (max_superior < 1e-75)) { // Danger of probability underflow
            for (int j = 0; j < num_classes; j++) 
                superior_probabilities[j] *= 1e75;
        }

      
        non_dominated_set = computeNonDominatedSetFromProbabilities(inferior_probabilities,superior_probabilities);
      
        non_dominated_index_set=getNonDominatedIndexSet(non_dominated_set); 
        
        return non_dominated_index_set;
    }
    
       /* Builds Cost sensitive Naive Bayes Credal classifier.
    *
    * @param data the training data
    * @exception Exception if classifier can't be built successfully
    */
    
        @Override
    public void buildClassifier(Instances data) throws Exception{
        int num_classes = data.numClasses();
        DiscreteEstimatorCostSensitiveNPI[] misclassification_frequencies;
        double[][] current_probabilities = new double[num_classes][];
        Instances training, test;
        CostSensitiveNaiveBayes current_model;
        Instance test_instance;
        int num_test_instances;
        int predicted_class_value, real_class_value;
        int[] non_dominated_index_set;
        int num_non_dominated_states;
        
        computeMatrixCostErrors(data);
        
        for(int i = 0; i < num_iterations; i++){
            misclassification_frequencies = new DiscreteEstimatorCostSensitiveNPI[num_classes];
            
            // Initialize frequencies for each class value
            for(int k = 0; k < num_classes; k++)
                misclassification_frequencies[k] = new DiscreteEstimatorCostSensitiveNPI(num_classes);
            
            // Cross validation to determine misclasssification frequencies
            for(int j = 0; j < num_folds; j++){
                training = data.trainCV(num_folds, j);
                test = data.testCV(num_folds, j);
                current_model = new CostSensitiveNaiveBayes();
                current_model.copyMatrixErrorCosts(matrix_cost_errors);
                
                if(i > 0)
                    current_model.computeClassWeightsFromProbabilities(training, current_probabilities);
                
                current_model.buildClassifier(training);
                num_test_instances = test.numInstances();
                
                for(int k = 0; k < num_test_instances; k++){
                    test_instance = test.instance(k);
                    real_class_value = (int)test_instance.classValue();
                    non_dominated_index_set = makePrediction(current_model, test_instance);
                    num_non_dominated_states = non_dominated_index_set.length;
                    
                    for(int l = 0; l < num_non_dominated_states; l++){                     
                        predicted_class_value = non_dominated_index_set[l];
                        misclassification_frequencies[real_class_value].addComponent(predicted_class_value, 1.0);
                    }
                }
            }
            
            current_probabilities = new double[num_classes][];
            
            /* Obtain, for each class value, the distribution of maximum entropy
               about predictiong each class value
            */ 
            
            for(int k = 0; k < num_classes; k++)
                current_probabilities[k] = misclassification_frequencies[k].distributionMaxEntropy();
        }
        
        // Now, compute the class weights with the current probabilities
        computeClassWeightsFromProbabilities(data, current_probabilities);
        
        super.buildClassifier(data);
        
    }
    
     /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options
   */
  
    public Enumeration listOptions() {
        Vector newVector = new Vector();

        newVector.addElement(new Option(
	"\tNumber of iterations for estimating weights via misclassification costs",
	"I", 1, "-it <option of cost matrix>"));
        
         newVector.addElement(new Option(
	"\tNumber of folds of cross validation in each iteration",
	"F", 1, "-nf <option of cost matrix>"));
    
        Enumeration enu = super.listOptions();
    
        while (enu.hasMoreElements()) {
            newVector.addElement(enu.nextElement());
        }

        return newVector.elements();
    }
    
        /**
    * Gets the current settings of the classifier.
    *
    * @return an array of strings suitable for passing to setOptions()
    */
    
    @Override
    public String[] getOptions() {
        Vector result;
        String[] options;
        int i;
    
        result = new Vector();
    
        result.add("-I");
        result.add("" + getNumIterations());
        
        result = new Vector();
    
        result.add("-F");
        result.add("" + getNumFolds());
    
        options = super.getOptions();
    
        for (i = 0; i < options.length; i++)
            result.add(options[i]);
    
    
        return (String[]) result.toArray(new String[result.size()]);
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
    
    
    @Override
    public void setOptions(String[] options) throws Exception{
        String tmpStr;
    
        tmpStr = Utils.getOption('I', options);
        
        setNumIterations(Integer.parseInt(tmpStr));
    
        tmpStr = Utils.getOption('I', options);
        
        setNumFolds(Integer.parseInt(tmpStr));
    
        super.setOptions(options);
    
        Utils.checkForRemainingOptions(options);
    }  
}
