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
public class Naive_Bayes_Completa_Admisible extends CredalClassifier implements OptionHandler, AdditionalMeasureProducer{
    
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
    return "Class for a Naive Bayes classifier using estimator classes. Numeric"
      +" estimator precision values are chosen based on analysis of the "
      +" training data. For this reason, the classifier is not an"
      +" UpdateableClassifier (which in typical usage are initialized with zero"
      +" training instances) -- if you need the UpdateableClassifier functionality,"
      +" use the NaiveBayesUpdateable classifier. The NaiveBayesUpdateable"
      +" classifier will  use a default precision of 0.1 for numeric attributes"
      +" when buildClassifier is called with zero training instances.\n\n"
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
   * @exception Exception if the instance could not be incorporated in
   * the model.
   */
    private void updateClassifier(Instance instance){
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
  public double classifyInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

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
  public double[] distributionForInstance(Instance instance) 
    throws Exception {

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
   * Calculates if each class value is non-dominated via the complete-admissible dominance crterion
   * @param inferior_probabilities array of inferior probabilities
   * @param superior_probabilities array of superior probabilities
   * @return an array of boolean indicating whether the corresponding state is non-dominated
   */
  
   private boolean[] computeNonDominatedStates(double[] inferior_probabilities, double[] superior_probabilities){
       int num_values = inferior_probabilities.length;
       boolean[] non_dominated_states = new boolean[num_values];
       boolean non_dominated;
       double superior_probability, inferior_probability;
       double numerator, fraction;
       double sum_inferior_probabilities = 0.0;
       double difference, rest;
       double max_value;
       double sum_differences = 0.0;
       double SValue;
       double[] SValues = new double[num_values];
       
       for(int i = 0; i < num_values; i++){
            superior_probability = superior_probabilities[i];
            inferior_probability = inferior_probabilities[i];
            difference = superior_probability - inferior_probability;
            sum_differences+=difference;
            sum_inferior_probabilities +=inferior_probability;
            
       }
       
       rest = 1 - sum_inferior_probabilities;
      
       for(int i = 0; i < num_values; i++){
            superior_probability = superior_probabilities[i];
            inferior_probability = inferior_probabilities[i];
            difference = superior_probability - inferior_probability;
            numerator = difference*rest;
            fraction = numerator/sum_differences;
            SValue = superior_probability - fraction;
            SValues[i] = SValue;
       }
       
        max_value = Double.NEGATIVE_INFINITY;
    
    for(int i = 0; i < num_values; i++){
        SValue = SValues[i];
        
        if(SValue > max_value)
            max_value = SValue;
    }
    
     for(int i = 0; i < num_values; i++){
        SValue = SValues[i];
        non_dominated = SValue == max_value;
        non_dominated_states[i] = non_dominated;
    }
       
       return non_dominated_states;
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
      int num_non_dominated_states;
      boolean non_dominated;
      int[] non_dominated_index_set;
      int cont;
      
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
      
      non_dominated_set =  computeNonDominatedStates(inferior_probabilities,superior_probabilities);
      
      num_non_dominated_states=0;
      
      for (int i=0; i<num_classes; i++){
        non_dominated = non_dominated_set[i];  
        
        if (non_dominated)
            num_non_dominated_states++;
      }
      

     non_dominated_index_set = new int[num_non_dominated_states];

      cont=0;
      
      for (int i=0; i<num_classes; i++){
         non_dominated = non_dominated_set[i];  
        
        if (non_dominated){
            non_dominated_index_set[cont]=i;
            cont++;
        }
      }

      this.updateStatistics(non_dominated_index_set, instance);


  }
  

    
}
