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
 * @author EquipoAsus
 */
public class CredalRandomForest extends CredalClassifier implements OptionHandler, AdditionalMeasureProducer, Randomizable{
  /** Number of trees in forest. */
  protected int m_numTrees = 100;
    
    /** Number of features to consider in random feature selection.
      If less than 1 will use int(logM+1) ) */
  protected int m_numFeatures = 0;

  /** The random seed. */
  protected int m_randomSeed = 1;  

  /** Final number of features that were considered in last build. */
  protected int m_KValue = 0;

  /** The bagger. */
  protected Bagging m_bagger = null;
  
    /**
   * Returns a string describing classifier
   * @return a description suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {

    return  "Class for constructing a forest of Credal Random Trees.\n\n"
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
    result.setValue(TechnicalInformation.Field.AUTHOR, "Abellan, J. and Moral, S");
    result.setValue(TechnicalInformation.Field.YEAR, "2003");
    result.setValue(TechnicalInformation.Field.TITLE, "Building classiffication trees using the total uncertainty criterion");
    result.setValue(TechnicalInformation.Field.JOURNAL, "International Journal of Intelligent Systems");
    result.setValue(TechnicalInformation.Field.VOLUME, "18");
    result.setValue(TechnicalInformation.Field.NUMBER, "12");
    result.setValue(TechnicalInformation.Field.PAGES, "1215-1225");

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
  public String numFeaturesTipText() {
    return "The number of attributes to be used in random selection (see CredalRandomTree).";
  }

  /**
   * Get the number of features used in random selection.
   *
   * @return Value of numFeatures.
   */
  public int getNumFeatures() {
    
    return m_numFeatures;
  }
  
  /**
   * Set the number of features to use in random selection.
   *
   * @param newNumFeatures Value to assign to numFeatures.
   */
  public void setNumFeatures(int newNumFeatures) {
    
    m_numFeatures = newNumFeatures;
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
	"\tNumber of features to consider (<1=int(logM+1)).",
	"K", 1, "-K <number of features>"));
    
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
    
    result.add("-K");
    result.add("" + getNumFeatures());
    
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
   * <pre> -K &lt;number of features&gt;
   *  Number of features to consider (&lt;1=int(logM+1)).</pre>
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
    
    tmpStr = Utils.getOption('K', options);
    if (tmpStr.length() != 0) {
      m_numFeatures = Integer.parseInt(tmpStr);
    } else {
      m_numFeatures = 0;
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

  @Override
    public void buildClassifier(Instances data) throws Exception {
        int num_attributes = data.numAttributes();
                
    // can classifier handle the data?
        getCapabilities().testWithFail(data);

    // remove instances with missing class
        data = new Instances(data);
        data.deleteWithMissingClass();
    
        m_bagger = new Bagging();
        CredalRandomTree rTree = new CredalRandomTree();

    // set up the random tree options
    m_KValue = m_numFeatures;
    
    if (m_KValue < 1) 
        m_KValue = (int) Utils.log2(num_attributes)+1;
    
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

    private void updateCredalStatistics(Instance instance) throws NoSupportForMissingValuesException {
        CredalRandomTree tree;
        int num_classes = instance.numClasses();
        boolean[] non_dominated_states = new boolean[num_classes];
        boolean[] partial_non_dominated_states;
        double[] frequency;
        boolean all_dominated;
        boolean non_dominated;
        boolean partial_non_dominated;
        int[] nonDominatedIndexSet;
        int times, min_times;
        int[] num_times_dominated = new int[num_classes]; 
        /*
        for(int j = 0; j < num_classes; j++)
            non_dominated_states[j] = true;
       */
        for(int i = 0; i < m_numTrees; i++){
            tree = (CredalRandomTree) m_bagger.getClassifier(i);
            frequency = tree.frequencyForInstance(instance);
            partial_non_dominated_states = tree.computeNonDominatedSet(frequency);
            
            for(int j = 0; j < num_classes; j++){
                partial_non_dominated = partial_non_dominated_states[j];
                                    
                if(!partial_non_dominated){
          //         non_dominated_states[j] = false;
                   num_times_dominated[j]++;
                }
                
            }
        }
        
        //all_dominated = checkAllDominated(non_dominated_states);
        
        //if(all_dominated){
            min_times = num_times_dominated[0];
            
            for(int j = 1; j < num_classes; j++){
               times = num_times_dominated[j];
               
               if(times < min_times)
                   min_times = times;
               
            }
            
            for(int j = 0; j < num_classes; j++){
                times = num_times_dominated[j];
                
                if(times == min_times)
                    non_dominated_states[j] = true;

            }
           /*
            for(int j = 0; j < num_classes; j++)
               non_dominated_states[j] = true;
           */
        //}   
        
        
        int num_non_dominated_states = 0;
        
        for(int j = 0; j < num_classes; j++){
            non_dominated = non_dominated_states[j];
            
            if(non_dominated)
                num_non_dominated_states++;
        }
        
        nonDominatedIndexSet = new int[num_non_dominated_states];
        int cont = 0;
        
        for(int j = 0; j < num_classes; j++){
            non_dominated = non_dominated_states[j];
            
            if(non_dominated){
                nonDominatedIndexSet[cont] = j;
                cont++;
            }
        }
        
        this.updateStatistics(nonDominatedIndexSet, instance);
        
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
      System.out.println(Evaluation.evaluateModel(new CredalRandomForest(), args));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }
  
}

