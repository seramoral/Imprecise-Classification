/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    ImpreciseImpreciseInfoGainAttributeEval.java
 *   
 *
 */

package weka.attributeSelection;

import weka.core.Capabilities;
import weka.core.ContingencyTables;
import weka.core.E_ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NumericToBinary;

import java.util.Enumeration;
import java.util.Vector;

/** 
 <!-- globalinfo-start -->
 * ImpreciseInfoGainAttributeEval :<br/>
 * <br/>
 * Evaluates the worth of an attribute by measuring the information gain with respect to the class.<br/>
 * <br/>
 * ImpreciseInfoGain(Class,Attribute) = H(Class) - H(Class | Attribute).<br/>
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -M
 *  treat missing values as a seperate value.</pre>
 * 
 * <pre> -B
 *  just binarize numeric attributes instead
 *   of properly discretizing them.</pre>
 * 
 <!-- options-end -->
 *
 * @author Mark Hall (mhall@cs.waikato.ac.nz)
 * @version $Revision: 1.18 $
 * @see Discretize
 * @see NumericToBinary
 */
public class ImpreciseInfoGainAttributeEval
  extends ASEvaluation
  implements AttributeEvaluator, OptionHandler {
  
  /** for serialization */
  static final long serialVersionUID = -1949849512589218930L;

  /** Treat missing values as a seperate value */
  private boolean m_missing_merge;

  /** Just binarize numeric attributes */
  private boolean m_Binarize;

  /** The info gain for each attribute */
  private double[] m_ImpreciseInfoGains;

  /** s value of the Imprecise Dirichlet Model that it is used for build the model*/
  double m_SValue=1.0;
  
  /**
   * Returns a string describing this attribute evaluator
   * @return a description of the evaluator suitable for
   * displaying in the explorer/experimenter gui
   */
  public String globalInfo() {
    return "ImpreciseInfoGainAttributeEval :\n\nEvaluates the worth of an attribute "
      +"by measuring the information gain with respect to the class.\n\n"
      +"ImpreciseInfoGain(Class,Attribute) = H*(Class) - H*(Class | Attribute).\n";
  }

  /**
   * Constructor
   */
  public ImpreciseInfoGainAttributeEval () {
    resetOptions();
  }

  /**
   * Returns an enumeration describing the available options.
   * @return an enumeration of all the available options.
   **/
  public Enumeration listOptions () {
    Vector newVector = new Vector(3);
    newVector.addElement(new Option("\ttreat missing values as a seperate " 
				    + "value.", "M", 0, "-M"));
    newVector.addElement(new Option("\tjust binarize numeric attributes instead\n " 
				    +"\tof properly discretizing them.", "B", 0, 
				    "-B"));
    newVector.addElement(new Option(
              "\tSpecifies the s value",
              "S", 1, "-S svalue"));
    return  newVector.elements();
  }


  /**
   * Parses a given list of options. <p/>
   *
   <!-- options-start -->
   * Valid options are: <p/>
   * 
   * <pre> -M
   *  treat missing values as a seperate value.</pre>
   * 
   * <pre> -B
   *  just binarize numeric attributes instead
   *   of properly discretizing them.</pre>
   * 
   <!-- options-end -->
   *
   * @param options the list of options as an array of strings
   * @throws Exception if an option is not supported
   */
  public void setOptions (String[] options)
    throws Exception {

    String convertList = Utils.getOption("S",options);
        if (convertList.length() != 0) {
            this.setSValue(Double.parseDouble(convertList));
        } else {
            this.setSValue(1);
        }
    resetOptions();
    setMissingMerge(!(Utils.getFlag('M', options)));
    setBinarizeNumericAttributes(Utils.getFlag('B', options));
  }


  /**
   * Gets the current settings of WrapperSubsetEval.
   *
   * @return an array of strings suitable for passing to setOptions()
   */
  public String[] getOptions () {
    String[] options = new String[10];
    int current = 0;

    if (!getMissingMerge()) {
      options[current++] = "-M";
    }
    if (getBinarizeNumericAttributes()) {
      options[current++] = "-B";
    }
    options[current++] = "-S"; options[current++] = ""+getSValue();
   
    while (current < options.length) {
      options[current++] = "";
    }

    return  options;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String binarizeNumericAttributesTipText() {
    return "Just binarize numeric attributes instead of properly discretizing them.";
  }

  /**
   * Binarize numeric attributes.
   *
   * @param b true=binarize numeric attributes
   */
  public void setBinarizeNumericAttributes (boolean b) {
    m_Binarize = b;
  }
  
public double getSValue(){
        return this.m_SValue;
    }

public void setSValue(double value){
        this.m_SValue=value;
    }

/**
   * get whether numeric attributes are just being binarized.
   *
   * @return true if missing values are being distributed.
   */
  public boolean getBinarizeNumericAttributes () {
    return  m_Binarize;
  }

  /**
   * Returns the tip text for this property
   * @return tip text for this property suitable for
   * displaying in the explorer/experimenter gui
   */
  public String missingMergeTipText() {
    return "Distribute counts for missing values. Counts are distributed "
      +"across other values in proportion to their frequency. Otherwise, "
      +"missing is treated as a separate value.";
  }

  /**
   * distribute the counts for missing values across observed values
   *
   * @param b true=distribute missing values.
   */
  public void setMissingMerge (boolean b) {
    m_missing_merge = b;
  }


  /**
   * get whether missing values are being distributed or not
   *
   * @return true if missing values are being distributed.
   */
  public boolean getMissingMerge () {
    return  m_missing_merge;
  }

  /**
   * Returns the capabilities of this evaluator.
   *
   * @return            the capabilities of this evaluator
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    
    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);
    result.enable(Capability.DATE_ATTRIBUTES);
    result.enable(Capability.MISSING_VALUES);
    
    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);
    
    return result;
  }

  /**
   * Initializes an information gain attribute evaluator.
   * Discretizes all attributes that are numeric.
   *
   * @param data set of instances serving as training data 
   * @throws Exception if the evaluator has not been 
   * generated successfully
   */
  public void buildEvaluator (Instances data)
    throws Exception {
    
    // can evaluator handle data?
    getCapabilities().testWithFail(data);
    
    int classIndex = data.classIndex();
    int numInstances = data.numInstances();
    
    int numClasses = data.attribute(classIndex).numValues();
    /*
    // Reserve space and initialize counters
    double[][][] counts = new double[data.numAttributes()][][];
    for (int k = 0; k < data.numAttributes(); k++) {
      if (k != classIndex) {
	int numValues = data.attribute(k).numValues();
	counts[k] = new double[numValues + 1][numClasses + 1];
      }
    }

    // Initialize counters
    double[] temp = new double[numClasses + 1];
    for (int k = 0; k < numInstances; k++) {
      Instance inst = data.instance(k);
      if (inst.classIsMissing()) {
	temp[numClasses] += inst.weight();
      } else {
	temp[(int)inst.classValue()] += inst.weight();
      }
    }
    for (int k = 0; k < counts.length; k++) {
      if (k != classIndex) {
	for (int i = 0; i < temp.length; i++) {
	  counts[k][0][i] = temp[i];
	}
      }
    }

    // Get counts
    for (int k = 0; k < numInstances; k++) {
      Instance inst = data.instance(k);
      for (int i = 0; i < inst.numValues(); i++) {
	if (inst.index(i) != classIndex) {
	  if (inst.isMissingSparse(i) || inst.classIsMissing()) {
	    if (!inst.isMissingSparse(i)) {
	      counts[inst.index(i)][(int)inst.valueSparse(i)][numClasses] += 
		inst.weight();
	      counts[inst.index(i)][0][numClasses] -= inst.weight();
	    } else if (!inst.classIsMissing()) {
	      counts[inst.index(i)][data.attribute(inst.index(i)).numValues()]
		[(int)inst.classValue()] += inst.weight();
	      counts[inst.index(i)][0][(int)inst.classValue()] -= 
		inst.weight();
	    } else {
	      counts[inst.index(i)][data.attribute(inst.index(i)).numValues()]
		[numClasses] += inst.weight();
	      counts[inst.index(i)][0][numClasses] -= inst.weight();
	    }
	  } else {
	    counts[inst.index(i)][(int)inst.valueSparse(i)]
	      [(int)inst.classValue()] += inst.weight();
	    counts[inst.index(i)][0][(int)inst.classValue()] -= inst.weight();
	  }
	}
      }
    }
  
    // Compute imprecise info gains
      for (int k=0; k<data.numAttributes();k++ ){
      if (k != classIndex){
        for (int i=0; i<data.attribute(k).numValues(); i++){
          
            counts[k][i]=reps(counts[k][i],counts[k][i].length,this.getSValue());
              System.out.println(this.getSValue());
        }
      } 
    }
    */
    weka.classifiers.trees.IPTree iptree = new weka.classifiers.trees.IPTree();
    iptree.setSValue(this.getSValue());
    m_ImpreciseInfoGains = new double[data.numAttributes()];
    for (int i = 0; i < data.numAttributes(); i++) {
      if (i != classIndex) {
        Attribute att = data.attribute(i);
	m_ImpreciseInfoGains[i] = iptree.computeInfoGain(data, att);
        /*
	  (ContingencyTables.entropyOverColumns(counts[i]) 
	   - ContingencyTables.entropyConditionedOnRows(counts[i]));
         */
      }
    }
  }
 public static double[] reps(double[] array,int n, double svalue){
       
        double masa=1;

        double[] l=new double[n];
        System.arraycopy(array,0,l,0,n);
        double min=l[0];
        int cont=0;
        for (int i=0;i<n;i++){
            if (min>l[i]) min=l[i];
        }
        for (int i=0;i<n;i++){
            if (min==l[i]) cont++;
        }
        if (svalue<=cont)
            masa=svalue;
        else
            masa=cont;
        for (int i=0;i<n;i++){
            if (min==l[i]) l[i]=l[i]+(masa/cont);
        }
        masa=masa-cont;
        if (masa>0.0001)
            return reps(l,n,masa);
        else
            return l;

    }
  /**
   * Reset options to their default values
   */
  protected void resetOptions () {
    m_ImpreciseInfoGains = null;
    m_missing_merge = true;
    m_Binarize = false;
  }
 
  /**
   * evaluates an individual attribute by measuring the amount
   * of information gained about the class given the attribute.
   *
   * @param attribute the index of the attribute to be evaluated
   * @return the info gain
   * @throws Exception if the attribute could not be evaluated
   */
  public double evaluateAttribute (int attribute)
    throws Exception {

    return m_ImpreciseInfoGains[attribute];
  }

  /**
   * Describe the attribute evaluator
   * @return a description of the attribute evaluator as a string
   */
  public String toString () {
    StringBuffer text = new StringBuffer();

    if (m_ImpreciseInfoGains == null) {
      text.append("Information Gain attribute evaluator has not been built");
    }
    else {
      text.append("\tInformation Gain Ranking Filter");
      if (!m_missing_merge) {
	text.append("\n\tMissing values treated as seperate");
      }
      if (m_Binarize) {
	text.append("\n\tNumeric attributes are just binarized");
      }
    }
    
    text.append("\n");
    return  text.toString();
  }

  
  // ============
  // Test method.
  // ============
  /**
   * Main method for testing this class.
   *
   * @param args the options
   */
  public static void main (String[] args) {
    runEvaluator(new ImpreciseInfoGainAttributeEval(), args);
  }
}
