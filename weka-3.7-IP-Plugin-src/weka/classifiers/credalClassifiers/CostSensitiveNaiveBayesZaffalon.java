/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.credalClassifiers;

import utils.DiscreteEstimatorZaffalon;
import weka.core.AdditionalMeasureProducer;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.OptionHandler;
import weka.core.TechnicalInformation;

/**
 *
 * @author Serafin
 */
public class CostSensitiveNaiveBayesZaffalon extends CostSensitiveCredalClassifier implements OptionHandler, AdditionalMeasureProducer{
    
    /** The attribute estimators. */
    protected DiscreteEstimatorZaffalon [][] distributions;

    /** The class estimator. */
    protected DiscreteEstimatorZaffalon class_distribution;
    
    /* Returns default capabilities of the classifier.
    *
    * @return      the capabilities of this classifier
    */
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // instances
        result.setMinimumNumberInstances(0);
    
        return result;
    }
    
      /**
   * Returns a string describing this classifier
   * @return a description of the classifier suitable for
   * displaying in the explorer/experimenter gui
   */
    public String globalInfo() {
        return "Class for an imprecise Naive Bayes classifier using estimator classes. Numeric"
        +" It considers the error costs.\n\n"
        +"For more information on Naive Bayes classifiers, see\n\n"
        + getTechnicalInformation().toString();
    }
    
        /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation 	result;
    
        result = new TechnicalInformation(TechnicalInformation.Type.INPROCEEDINGS);
        result.setValue(TechnicalInformation.Field.AUTHOR, "G. Corani and M. Zaffalon");
        result.setValue(TechnicalInformation.Field.YEAR, "2001");
        result.setValue(TechnicalInformation.Field.TITLE, "Statistical inference of the naive credal classifier");
        result.setValue(TechnicalInformation.Field.BOOKTITLE, "Proceedings of the Second International Symposium on Imprecise Probabilities and Their Applications (ISIPTA '01)");
        result.setValue(TechnicalInformation.Field.PAGES, "384--393");
    
        return result;
    }
    
     
    /* Builds Naive Bayes Zaffalon classifier.
    *
    * @param data the training data
    * @exception Exception if classifier can't be built successfully
    */
  
    @Override
    public void buildClassifier(Instances data) throws Exception{
        int num_classes = data.numClasses();
        int num_attributes = data.numAttributes() - 1;
        int class_index = data.classIndex();
        double s_value = this.getSValue();
        Attribute attribute;
        int num_values;
        int num_instances = data.numInstances();
        Instance instance;
      
        initStats();  
        // can classifier handle the data?
        getCapabilities().testWithFail(data);
        
        computeMatrixCostErrors(data);

    
        class_distribution = new DiscreteEstimatorZaffalon(num_classes, s_value);
        distributions = new DiscreteEstimatorZaffalon[num_attributes][num_classes];
    
        for(int i = 0; i<=num_attributes; i++){
            if(i!= class_index){
                attribute = data.attribute(i);
                num_values = attribute.numValues();
            
                for(int j = 0; j < num_classes; j++)
                    distributions[i][j] = new DiscreteEstimatorZaffalon(num_values,s_value);
            }
        }
    
        for(int i = 0; i < num_instances; i++){
            instance = data.instance(i);
            updateClassifier(instance);
        }
    
    }
  
      /**
   * Updates the classifier with the given instance.
   *
   * @param instance the new training instance to include in the model 
   */
    protected void updateClassifier(Instance instance){
        int num_attributes = instance.numAttributes();
        int class_index = instance.classIndex();
        int class_value = (int)instance.classValue();
        int attribute_value;
        
        class_distribution.addComponent(class_value);
        
        for(int i = 0; i < num_attributes; i++){
            if(i!=class_index){
                attribute_value = (int)instance.value(i);
                distributions[i][class_value].addComponent(attribute_value);
            }
        }
    }
    
    /* Classifies a given test instance using the Naive Bayes.
    *
    * @param instance the instance to be classified
    * @return the classification
    * @throws NoSupportForMissingValuesException if instance has missing values
    */
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("IPTree: no missing values, "
                                                   + "please.");
        }

        this.updateCredalStatistics(instance);

        return Double.NaN;
    }
  
    
    /* Computes class distribution for instance using Zaffalon Naive Bayes.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
    public double[] distributionForInstance(Instance instance) throws Exception {

        if (instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("IPTree: no missing values, "
                                                   + "please.");
        }

        this.updateCredalStatistics(instance);

        double [] probs = new double[instance.numClasses()];
        
        for (int i=0; i<probs.length; i++)
            probs[i]=Double.NaN;

        return probs;
    }
    
    /**
    * Compute Non-Dominate Index set for an instance to update credal statistics 
    * @param instance, the instance from which update statistics
    * @throws NoSupportForMissingValuesException 
    */
   
    private void updateCredalStatistics(Instance instance) throws NoSupportForMissingValuesException{
        boolean[] non_dominated_set;
        int num_classes = instance.numClasses();
        double[] inferior_probabilities = new double[num_classes];
        double[] superior_probabilities = new double[num_classes];
        double class_inferior_probability, class_superior_probability;
        double inferior_probability, superior_probability;
        double partial_inferior_probability, partial_superior_probability;
        int class_index = instance.classIndex();
        int num_attributes = instance.numAttributes();
        int attribute_value;
        double max_inferior = 0;
        double max_superior = 0;
        int[] non_dominated_index_set;
        double[] lower_risks, upper_risks;
      
        for(int j = 0; j < num_classes; j++){
            class_inferior_probability = class_distribution.getInferiorProbability(j);
            class_superior_probability = class_distribution.getSuperiorProbability(j);
            inferior_probability = class_inferior_probability;
            superior_probability = class_superior_probability;
          
            for(int i = 0; i < num_attributes; i++){
                if(i!=class_index){
                    attribute_value = (int)instance.value(i);
                    partial_inferior_probability = distributions[i][j].getInferiorProbability(attribute_value);
                    partial_superior_probability = distributions[i][j].getSuperiorProbability(attribute_value);
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
        
        lower_risks = getRiskFromProbabilities(inferior_probabilities);
        upper_risks = getRiskFromProbabilities(superior_probabilities);
      
        non_dominated_set =  computeNonDominatedStatesRisks(lower_risks,upper_risks);
      
        non_dominated_index_set=getNonDominatedIndexSet(non_dominated_set); 
      
        this.updateStatistics(non_dominated_index_set, instance);


    }
}
