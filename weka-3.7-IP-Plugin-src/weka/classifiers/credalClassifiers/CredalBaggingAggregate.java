/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package weka.classifiers.credalClassifiers;

import java.util.Enumeration;
import java.util.Vector;
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
public class CredalBaggingAggregate extends CredalClassifier implements OptionHandler, AdditionalMeasureProducer, Randomizable{
/** Number of trees in forest. */
  protected int m_numTrees = 100;
    
  /** The random seed. */
  protected int m_randomSeed = 1;  

  /** The bagger. */
  protected Bagging m_bagger = null;
  
  /** The matrix of possibilities degrees */
  protected double[][] matrix_degrees;
  
  /** Preference scores */
  protected double[] preference_scores;
  
  /** No preference scores */
  protected double[] non_preference_scores;
  
  
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
   * Combines two belief intervals for each class value
   * @param inf_prob1 Inferior probabilities of the first intervals
   * @param sup_prob1 Superior probababilities of the first intervals
   * @param inf_prob2 Inferior probabilities of the first intervals
   * @param sup_prob2 Superior probabilities of the first intervals
   * @return Array 2xK with the inferior and superior probabilities of each possible value of the class variable. 
   */
  
    private double[][] combineTwoBeliefIntervals(double[] inf_prob1, double[] sup_prob1, double[] inf_prob2, double[] sup_prob2){
        int num_states = inf_prob1.length;
        double[][] combined_intervals = new double[2][num_states];
        double[] combined_inferior_probabilities = new double[num_states];
        double[] combined_superior_probabilities = new double[num_states];
        double[] aux_combined_superior_probabilities = new double[num_states];

        double partial_inf1, partial_inf2, partial_sup1, partial_sup2;
        double partial_combined_inferior, partial_combined_superior;

        for(int j = 0; j < num_states; j++){
            partial_inf1 = inf_prob1[j];
            partial_inf2 = inf_prob2[j];
            partial_sup1 = sup_prob1[j];
            partial_sup2 = sup_prob2[j];
            
            partial_combined_inferior = 1 -(1 - partial_inf1)*(1 - partial_inf2);
            partial_combined_superior = (1 - partial_sup1)*(1 - partial_sup2);
            combined_inferior_probabilities[j] = partial_combined_inferior;
            aux_combined_superior_probabilities[j] = partial_combined_superior;
        }
        
        if(Utils.sum(combined_inferior_probabilities) > 0)
            Utils.normalize(combined_inferior_probabilities);
        
        if(Utils.sum(aux_combined_superior_probabilities) > 0)
            Utils.normalize(aux_combined_superior_probabilities);
        
        for(int j = 0; j < num_states; j++){ 
            combined_superior_probabilities[j] = 1 - aux_combined_superior_probabilities[j];
            
            if(combined_superior_probabilities[j] < combined_inferior_probabilities[j])
               combined_superior_probabilities[j] = combined_inferior_probabilities[j];                      
        }
        
        combined_intervals[0] = combined_inferior_probabilities;
        combined_intervals[1] = combined_superior_probabilities;
        
        return combined_intervals;
  }
    
    /**
     * Possibility degree after the combination of interval BI_i = [Bel_i,Pl_i], BI_j = [Bel_j,Pl_j], P(BIi >= BIj)
     * @param bel_i Beli 
     * @param pl_i Pli
     * @param bel_j Belj
     * @param pl_j Plj
     * @return P(BIi >= BIj)
     */
    
    private double computePossibilityDegree(double bel_i, double pl_i, double bel_j, double pl_j){
        double numerator = pl_j - bel_i;
        double denominator = pl_j - bel_j + pl_i - bel_i;
        double possibility_degree;
        double fraction;
        
        if(denominator == 0)
            possibility_degree = pl_j-pl_i;
        
        else{
            fraction = numerator/denominator;
            
            if(fraction < 0)
                fraction = 0;
            
            possibility_degree = 1 - fraction;
            
            if(possibility_degree < 0)
                possibility_degree = 0;
        }
        
        return possibility_degree;
    }
    
    /**
     * Computes the matrix of possibility degrees after the combination. pij = P(BI_i >= BI_j)
     * @param combined_inferior_probabilities the inferior probabilities resulting from the combination
     * @param combined_superior_probabilities the superior probabilities resulting from the combination
     */
    
    private void computeMatrixDegrees(double[] combined_inferior_probabilities, double[] combined_superior_probabilities){
        int num_states = combined_inferior_probabilities.length;
        double bel_i, bel_j, pl_i, pl_j;
        double possibility_degree;
        
        matrix_degrees = new double[num_states][num_states];
        
        for(int i = 0; i < num_states; i++){
            matrix_degrees[i][i] = 0.5;
            bel_i = combined_inferior_probabilities[i];
            pl_i = combined_superior_probabilities[i];
                
            for(int j = 0; j < i; j++){
                bel_j = combined_inferior_probabilities[j];
                pl_j = combined_superior_probabilities[j];
                possibility_degree = computePossibilityDegree(bel_i, pl_i, bel_j, pl_j);
                matrix_degrees[i][j] = possibility_degree;
                matrix_degrees[j][i] = 1 - possibility_degree;
            }
        }
    }
    
    /**
     * Computes the preference and non-preference scores from the Matrix of degrees. 
     * Preference score for i = sum_{j}P(BI_i >= BI_j) 
     * Non Preference score for i = sum_{i}P(BI_j >= BI_i)
     */
    
    private void computePreferenceScores(){
        int num_classes = matrix_degrees.length;
        preference_scores = new double[num_classes];
        non_preference_scores = new double[num_classes];

        for(int i = 0; i < num_classes; i++){
            for(int j = 0; j < num_classes; j++){
                preference_scores[i]+=matrix_degrees[i][j];
                non_preference_scores[i]+=matrix_degrees[j][i];
            }
        }
        
    }
    /**
     * estimates if there are a class value i for which P(BI_i >=PI_j) >= 0.5 forall j \neq i
     * @return true if there exists such class value, false else 
     */
    
    private boolean clearDomining(){
        int num_classes = matrix_degrees.length;
        boolean clear_domining = false;
        boolean dominated;
                
        for(int i = 0; i < num_classes && !clear_domining; i++){
            dominated = false;
            
            for(int j = 0; j < num_classes && !dominated; j++){
                if(j!=i)
                    dominated = matrix_degrees[j][i] > 0.5; 
            }
            
            if(!dominated)
                clear_domining = true;
        }
        
        return clear_domining;
    }
    
    /**
     * Checks if the state c_j is non-dominated. It is non-dominated if, and only if, sum_{i \neq j}P(BI_i >= BI_j) >=  sum_{i \neq j}P(BI_j >= BI_i)
     * @param j the state j
     * @return true if cj is dominated, false otherwise
     */
    
    private boolean nonDominated(int j, boolean determinely){
        int num_classes = matrix_degrees.length;
        boolean non_dominated;
        double max_score = 0;
        
        if(determinely){
            for(int i = 0; i < num_classes; i++){
                if(preference_scores[i] > max_score)
                    max_score=preference_scores[i];
            }
            
            non_dominated = preference_scores[j] == max_score;
        }
   
        else
            non_dominated = preference_scores[j] > non_preference_scores[j];
       
       return non_dominated;
    }
    
    /**
     * Computes the non-dominated states set checking if each class value is non-dominated.
     * @return 
     */
    
    private boolean[] computeNonDominatedStatesSet(boolean determinely){
        int num_classes = matrix_degrees.length;
        boolean[] non_dominated_states = new boolean[num_classes];
        
        boolean non_dominated;
        
        for(int i = 0; i < num_classes; i++){
            non_dominated = nonDominated(i, determinely);
            non_dominated_states[i] = non_dominated;
        }
        
        return non_dominated_states;
    }
    
    /**
     * Update the credal statistics for an instance. Combines the belief intervals. Compute the matrix of degrees and, then,
     * the non-dominated states set
     * @param instance The instance
     * @throws NoSupportForMissingValuesException 
     */
    
    private void updateCredalStatistics(Instance instance) throws NoSupportForMissingValuesException {
        CredalDecisionTree2 tree;
        int num_classes = instance.numClasses();
        boolean[] non_dominated_states;
        boolean non_dominated;
        int[] non_dominated_index_set;
        int cont;
        int num_non_dominated_states;
        double[][] partial_extreme_probabilities, combined_probabilities;
        double[] partial_inferior_probabilities, partial_superior_probabilities;
        double[] combined_inferior_probabilities, combined_superior_probabilities;
        boolean determinely;
        
        tree = (CredalDecisionTree2) m_bagger.getClassifier(0);
        
        combined_probabilities = tree.getExtremeProbabilities(instance);
        combined_inferior_probabilities = combined_probabilities[0];
        combined_superior_probabilities = combined_probabilities[1];
        
        for(int i = 1; i < m_numTrees; i++){
            tree = (CredalDecisionTree2) m_bagger.getClassifier(i);
            partial_extreme_probabilities = tree.getExtremeProbabilities(instance);
            partial_inferior_probabilities = partial_extreme_probabilities[0];
            partial_superior_probabilities = partial_extreme_probabilities[1];
            combined_probabilities = combineTwoBeliefIntervals(partial_inferior_probabilities, partial_superior_probabilities, combined_inferior_probabilities, combined_superior_probabilities);
            combined_inferior_probabilities = combined_probabilities[0];
            combined_superior_probabilities = combined_probabilities[1];  
        }

        computeMatrixDegrees(combined_inferior_probabilities, combined_superior_probabilities);
        computePreferenceScores();
        determinely = clearDomining();
        
        non_dominated_states = computeNonDominatedStatesSet(determinely);
        
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
    /** The main method for chekings*/
    
    public static void main(String[] args) {
        CredalBaggingAggregate bagging = new CredalBaggingAggregate();
        double[] inferior_probabilities1 = new double[3];
        double[] inferior_probabilities2 = new double[3];
        double[] inferior_probabilities3 = new double[3];
        double[] superior_probabilities1 = new double[3];
        double[] superior_probabilities2 = new double[3];
        double[] superior_probabilities3 = new double[3];
        double[] result_inferior, result_superior;
        
        inferior_probabilities1[0] = 1;
        inferior_probabilities1[1] = 0;
        inferior_probabilities1[2] = 0;
        inferior_probabilities2[0] = 0;
        inferior_probabilities2[1] = 0.6;
        inferior_probabilities2[2] = 0.2;
        inferior_probabilities3[0] = 0;
        inferior_probabilities3[1] = 0.7;
        inferior_probabilities3[2] = 0;
        superior_probabilities1[0] = 1;
        superior_probabilities1[1] = 0;
        superior_probabilities1[2] = 0;
        superior_probabilities2[0] = 0;
        superior_probabilities2[1] = 0.8;
        superior_probabilities2[2] = 0.4;
        superior_probabilities3[0] = 0.1;
        superior_probabilities3[1] = 0.9;
        superior_probabilities3[2] = 0.3;
        
        double[][] first_combination = bagging.combineTwoBeliefIntervals(inferior_probabilities1, superior_probabilities1, inferior_probabilities2, superior_probabilities2);
        result_inferior = first_combination[0];
        result_superior = first_combination[1];
        double[][] second_combination = bagging.combineTwoBeliefIntervals(result_inferior, result_superior, inferior_probabilities3, superior_probabilities3);

        System.out.println("Combination");
        
    }
    
}
