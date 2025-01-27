/*
 * E_Filter.java
 *
 * Created on 19 de abril de 2006, 15:05
 */

package weka.filters.supervised.attribute;

import weka.filters.*;
import weka.core.*;
import java.io.*;
import java.util.*;
import weka.core.Capabilities.Capability;


/**
 * This class inherits from weka.filters.Filter class.
 * @author Andres
 */

public class E_Filter extends Filter implements SupervisedFilter, AdditionalMeasureProducer {

    /** for serialization */
    static final long serialVersionUID =  -1018753272542481711L;
    
    /**
     * Creates a new instance of E_Filter
     */
    public E_Filter() {
    }

    /**
    * Returns a string describing this filter
    *
    * @return a description of the filter suitable for
    * displaying in the explorer/experimenter gui
    */
    public String globalInfo() {
        return "";
    }

    /**
    * Sets the format of the input instances.
    *
    * @param instanceInfo an Instances object containing the input instance
    * structure (any instances contained in the object are ignored - only the
    * structure is required).
    * @return true if the outputFormat may be collected immediately
    * @exception Exception if the input format can't be set successfully
    */
  /*
    public boolean setInputFormat(Instances instanceInfo) throws Exception {
        super.setInputFormat(instanceInfo);
        return false;
    }
*/
   
    /**
    * Signify that this batch of input to the filter is finished. If
    * the filter requires all instances prior to filtering, output()
    * may now be called to retrieve the filtered instances. Any
    * subsequent instances filtered should be filtered based on setting
    * obtained from the first batch (unless the inputFormat has been
    * re-assigned or new options have been set). This default
    * implementation assumes all instance processing occurs during
    * inputFormat() and input().
    *
    * @return true if there are instances pending output
    * @throws NullPointerException if no input structure has been defined,
    * @throws Exception if there was a problem finishing the batch.
    */
    /*
    public boolean batchFinished() throws Exception {

        if (this.getInputFormat() == null) {
          throw new NullPointerException("No input instance format defined");
        }
        
        setOutputFormat(new Instances(this.getInputFormat()));

        // Convert pending input instances
        for(int i = 0; i < this.getInputFormat().numInstances(); i++) {
            push(this.getInputFormat().instance(i));
        }
        
        flushInput();
        m_NewBatch = true;
        m_FirstBatchDone = true;
        return (numPendingOutput() != 0);
    }    
    */
  /**
   * Input an instance for filtering. Ordinarily the instance is processed
   * and made available for output immediately. Some filters require all
   * instances be read before producing output.
   *
   * @param instance the input instance
   * @return true if the filtered instance may now be
   * collected with output().
   * @exception IllegalStateException if no input format has been defined.
   */
  public boolean input(Instance instance) throws Exception{

    if (getInputFormat() == null) {
      throw new IllegalStateException("No input instance format defined");
    }
    if (m_NewBatch) {
      resetQueue();
      this.setOutputFormat(this.getInputFormat());
      m_NewBatch = false;
    }
    instance.setDataset(this.getOutputFormat());
    push(instance);
    return true;

  }
    
    /**
    * Creates a new instance of a classifier given it's class name and
    * (optional) arguments to pass to it's setOptions method. If the
    * classifier implements OptionHandler and the options parameter is
    * non-null, the classifier will have it's options set.
    *
    * @param classifierName the fully qualified class name of the classifier
    * @param options an array of options suitable for passing to setOptions. May
    * be null.
    * @return the newly created classifier, ready for use.
    * @exception Exception if the classifier name is invalid, or the options
    * supplied are not acceptable to the classifier
    */
    public static Filter forName(String FilterName, String [] options) throws Exception {
        return (Filter)Utils.forName(Filter.class,FilterName,options);
    }

    
  /** 
   * Returns the Capabilities of this filter.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();

    // attributes
    result.enableAllAttributes();
    result.enable(Capability.MISSING_VALUES);
    
    // class
    result.enable(Capability.NOMINAL_CLASS);
    
    return result;
  }
  
  /**
    * Returns an enumeration of any additional measure names that might be
    * in the classifier
    * @return an enumeration of the measure names
    */
    public Enumeration enumerateMeasures() {
        Vector newVector = new Vector(4);
        newVector.addElement("measureNumVariables");
        newVector.addElement("measureTotalCases");
        newVector.addElement("measureMeanCases");
        newVector.addElement("measureNumOneCaseVariables");
        return newVector.elements();
    }

    /**
    * Returns the value of the named measure
    * @param measureName the name of the measure to query for its value
    * @return the value of the named measure
    * @exception IllegalArgumentException if the named measure is not supported
    */
    public double getMeasure(String additionalMeasureName) {
        if (additionalMeasureName.compareToIgnoreCase("measureNumVariables") == 0) {
          return this.getOutputFormat().numAttributes()-1;
        }else if (additionalMeasureName.compareToIgnoreCase("measureTotalCases") == 0) {
          return this.measureTotalCases();
        } else if (additionalMeasureName.compareToIgnoreCase("measureMeanCases") == 0) {
          return this.measureMeanCases();
        } else if (additionalMeasureName.compareToIgnoreCase("measureNumOneCaseVariables") == 0) {
          return this.measureNumOneCaseVariables();
        } else {
          throw new IllegalArgumentException(additionalMeasureName 
                              + " not supported (j48)");
        }
    }

    public double measureTotalCases() {
        double total=0.0;
        for (int i=0; i<this.getOutputFormat().numAttributes(); i++)
            total+=this.getOutputFormat().attribute(i).numValues();
        return total;
    }
    
    public double measureMeanCases() {
        return this.measureTotalCases()/this.getOutputFormat().numAttributes();
    }

    public double measureNumOneCaseVariables() {
        double total=0.0;
        for (int i=0; i<this.getOutputFormat().numAttributes(); i++)
            if (this.getOutputFormat().attribute(i).numValues()==1)
                total++;
        return total;
    }
 
     
    /**
     * @param args the command line arguments
     */
    public static void main(String[] argv) {
        // TODO code application logic here
        try {
          if (Utils.getFlag('b', argv)) {
            Filter.batchFilterFile(new E_Filter(), argv);
          } else {
            Filter.filterFile(new E_Filter(), argv);
          }
        } catch (Exception ex) {
          System.out.println(ex.getMessage());
        }
    }

}
