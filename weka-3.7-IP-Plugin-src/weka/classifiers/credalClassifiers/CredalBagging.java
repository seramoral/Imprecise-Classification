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
public class CredalBagging extends CredalClassifier implements OptionHandler, AdditionalMeasureProducer, Randomizable{
/** Number of trees in forest. */
  protected int m_numTrees = 100;
    
  /** The random seed. */
  protected int m_randomSeed = 1;  

  /** The bagger. */
  protected Bagging m_bagger = null;
  
    /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return  "Class for constructing a Bagging of Credal Decision Trees for Imprecise Classification.\n\n"
      + "For more information see: \n\n"
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
    TechnicalInformation result;

    result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
    result.setValue(TechnicalInformation.Field.AUTHOR, "Serafín Moral-García, Carlos J. Mantas, Javier G. Castellano, María D. Benítez and Joaquín Abellán");
    result.setValue(TechnicalInformation.Field.YEAR, "2019");
    result.setValue(TechnicalInformation.Field.TITLE, "Bagging of Credal Decision Trees for Imprecise classiffication");
    result.setValue(TechnicalInformation.Field.JOURNAL, "Expert Systems with Applications");
    result.setValue(TechnicalInformation.Field.VOLUME, "141");
    result.setValue(TechnicalInformation.Field.NUMBER, "1");
    result.setValue(TechnicalInformation.Field.PAGES, "112944");

    return result;
  }
  
  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
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
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String numTreesTipText() {
    return "The number of trees to be generated.";
  }

  /**
   * Get the value of numTrees.
   *
   * @return Value of numTrees.
   */
  public int getNumTrees() {   
    return m_numTrees;
  }
  
  /**
   * Set the value of numTrees.
   *
   * @param newNumTrees Value to assign to numTrees.
   */
  public void setNumTrees(int newNumTrees) {
    
    m_numTrees = newNumTrees;
  }
  
  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String seedTipText() {
    return "The random number seed to be used.";
  }

  /**
   * Set the seed for random number generation.
   *
   * @param seed the seed 
   */
  public void setSeed(int seed) {
    m_randomSeed = seed;
  }
  
  /**
   * Gets the seed for the random number generations
   *
   * @return the seed for the random number generation
   */
  public int getSeed() {
    return m_randomSeed;
  }
  
  /**
   * Returns an enumeration describing the available options.
   *
   * @return an enumeration of all the available options
   */
  public Enumeration listOptions() {
    
    Vector newVector = new Vector();

    newVector.addElement(new Option(
	"\tNumber of trees to build.",
	"I", 1, "-I <number of trees>"));
    
    newVector.addElement(new Option(
	"\tSeed for random number generator.\n"
	+ "\t(default 1)",
	"s", 1, "-s"));

    Enumeration enu = super.listOptions();
    
    while (enu.hasMoreElements()) {
      newVector.addElement(enu.nextElement());
    }

    return newVector.elements();
  }
  
  /**
   * Gets the current settings of the forest.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String[] getOptions() {
    Vector        result;
    String[]      options;
    int           i;
    
    result = new Vector();
    
    result.add("-I");
    result.add("" + getNumTrees());
    
    result.add("-s");
    result.add("" + getSeed());
    
    options = super.getOptions();
    for (i = 0; i < options.length; i++)
      result.add(options[i]);
    
    return (String[]) result.toArray(new String[result.size()]);
  }
  
  /**
   * Parses a given list of options. <p/>
   * 
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -I &lt;number of trees&gt;
   *  Number of trees to build.</pre>
   * 
   * <pre> -S
   *  Seed for random number generator.
   *  (default 1)</pre>
   * 
   * <pre> -depth &lt;num&gt;
   *  The maximum depth of the trees, 0 for unlimited.
   *  (default 0)</pre>
   * 
   * <pre> -D
   *  If set, classifier is run in debug mode and
   *  may output additional info to the console</pre>
   * 
   <!-- options-end -->
   * 
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions(String[] options) throws Exception{
    String tmpStr;
    
    tmpStr = Utils.getOption('I', options);
    if (tmpStr.length() != 0) {
      m_numTrees = Integer.parseInt(tmpStr);
    } else {
      m_numTrees = 100;
    }
    
    tmpStr = Utils.getOption('s', options);
    if (tmpStr.length() != 0) {
      setSeed(Integer.parseInt(tmpStr));
    } else {
      setSeed(1);
    }
    
    super.setOptions(options);
    
    Utils.checkForRemainingOptions(options);
  }  
  
  /**
   * Builds the base classifiers 
   * @param data the original training set
   * @throws Exception 
   */

  @Override
    public void buildClassifier(Instances data) throws Exception {
        
        initStats();
        
    // can classifier handle the data?
        getCapabilities().testWithFail(data);

    // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
    
        m_bagger = new Bagging();
        CredalDecisionTree2 rTree = new CredalDecisionTree2();

   // rTree.setKValue(m_KValue);

    // set up the bagger and build the forest
        m_bagger.setClassifier(rTree);
        m_bagger.setSeed(m_randomSeed);
        m_bagger.setNumIterations(m_numTrees);
        m_bagger.setCalcOutOfBag(false);
        m_bagger.buildClassifier(data);
  }
    
     /**
   * Computes class distribution for instance using decision tree.
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
   * It update the statistics for a given instance. It computes the non-dominated index set for the instance. 
   * It calculates how many times a state has been dominated
   * The non-dominated states are those for which the number of times that have been dominated is minimum
   * @param instance the instance
   * @throws NoSupportForMissingValuesException 
   */

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
        double min_times_dominated, partial_times_dominated;
        
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
        
        for(int j = 0; j < num_classes; j++){
            partial_times_dominated = times_dominated[j];
            non_dominated_states[j] = partial_times_dominated == min_times_dominated;
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
    
  @Override
    public double classifyInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("IPTree: no missing values, "
                                                   + "please.");
    }

    this.updateCredalStatistics(instance);

    return Double.NaN;
  }
    
    /**
   * Main method.
   *
   * @param args the options for the classifier
   */
  public static void main(String[] args) {

    try {
      System.out.println(Evaluation.evaluateModel(new CredalBagging(), args));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }
    
}
