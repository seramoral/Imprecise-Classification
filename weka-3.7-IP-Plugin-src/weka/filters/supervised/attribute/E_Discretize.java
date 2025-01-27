package weka.filters.supervised.attribute;

import weka.filters.*;
import java.io.*;
import java.util.*;
import weka.core.*;
import weka.core.Capabilities.Capability;


/**
 * This class inherits from the weka.filters.supervised.attribute.Discretize class of weka. 
 * The added functionality is only the removal of the one-state variables after the supervised
 * discretization process.
 */
public class E_Discretize extends Discretize{
    

    /** for serialization */
    static final long serialVersionUID = 2889730616939923301L;
    
    
    /**
     * Creates a new instance of E_Discretize
     */
    public E_Discretize() {
        super();
    }

  /** 
   * Returns the Capabilities of this filter.
   *
   * @return            the capabilities of this object
   * @see               Capabilities
   */
  public Capabilities getCapabilities() {
    return super.getCapabilities();
  }
    

  /**
   * Set the output format. Takes the currently defined cutpoints and 
   * m_InputFormat and calls setOutputFormat(Instances) appropriately.
   */
  protected void setOutputFormat() {

    if (m_CutPoints == null) {
      setOutputFormat(null);
      return;
    }
    FastVector attributes = new FastVector(getInputFormat().numAttributes());
    int classIndex = getInputFormat().classIndex();
    for(int i = 0; i < getInputFormat().numAttributes(); i++) {
      if ((m_DiscretizeCols.isInRange(i)) 
	  && (getInputFormat().attribute(i).isNumeric())) {
	if (!m_MakeBinary) {
	  FastVector attribValues = new FastVector(1);
	  if (m_CutPoints[i] == null) {
	    //attribValues.addElement("'All'");
	  } else {
	    for(int j = 0; j <= m_CutPoints[i].length; j++) {
	      if (j == 0) {
		attribValues.addElement("'(-inf-"
			+ Utils.doubleToString(m_CutPoints[i][j], 6) + "]'");
	      } else if (j == m_CutPoints[i].length) {
		attribValues.addElement("'("
			+ Utils.doubleToString(m_CutPoints[i][j - 1], 6) 
					+ "-inf)'");
	      } else {
		attribValues.addElement("'("
			+ Utils.doubleToString(m_CutPoints[i][j - 1], 6) + "-"
			+ Utils.doubleToString(m_CutPoints[i][j], 6) + "]'");
	      }
	    }
	  attributes.addElement(new Attribute(getInputFormat().
					      attribute(i).name(),
					      attribValues));
          }
	} else {
	  if (m_CutPoints[i] == null) {
	    FastVector attribValues = new FastVector(1);
	    attribValues.addElement("'All'");
	    attributes.addElement(new Attribute(getInputFormat().
						attribute(i).name(),
						attribValues));
	  } else {
	    if (i < getInputFormat().classIndex()) {
	      classIndex += m_CutPoints[i].length - 1;
	    }
	    for(int j = 0; j < m_CutPoints[i].length; j++) {
	      FastVector attribValues = new FastVector(2);
	      attribValues.addElement("'(-inf-"
		      + Utils.doubleToString(m_CutPoints[i][j], 6) + "]'");
	      attribValues.addElement("'("
		      + Utils.doubleToString(m_CutPoints[i][j], 6) + "-inf)'");
	      attributes.addElement(new Attribute(getInputFormat().
						  attribute(i).name(),
						  attribValues));
	    }
	  }
	}
      } else {
	attributes.addElement(getInputFormat().attribute(i).copy());
      }
    }
    
    Instances outputFormat = 
      new Instances(getInputFormat().relationName(), attributes, 0);
    outputFormat.setClassIndex(-1);
    Enumeration enu=outputFormat.enumerateAttributes();
    while(enu.hasMoreElements()){
        Attribute att=(Attribute)enu.nextElement();
        if (att.equals(this.getInputFormat().classAttribute())){
            outputFormat.setClass(att);
            break;
        }
    }
    setOutputFormat(outputFormat);
  }
    
  /**
   * Convert a single instance over. The converted instance is added to 
   * the end of the output queue.
   *
   * @param instance the instance to convert
   */
  protected void convertInstance(Instance instance) {

    int index = 0;
    double [] vals = new double [outputFormatPeek().numAttributes()];
    // Copy and convert the values
    for(int i = 0; i < getInputFormat().numAttributes(); i++) {
      if (m_DiscretizeCols.isInRange(i) && 
	  getInputFormat().attribute(i).isNumeric()) {
	int j;
	double currentVal = instance.value(i);
	if (m_CutPoints[i] == null) {
	  /*
          if (instance.isMissing(i)) {
	    vals[index] = Instance.missingValue();
	  } else {
	    vals[index] = 0;
	  }
	  index++;
           */
	} else {
	  if (!m_MakeBinary) {
	    if (instance.isMissing(i)) {
	      vals[index] = Instance.missingValue();
	    } else {
	      for (j = 0; j < m_CutPoints[i].length; j++) {
		if (currentVal <= m_CutPoints[i][j]) {
		  break;
		}
	      }
              vals[index] = j;
	    }
	    index++;
	  } else {
	    for (j = 0; j < m_CutPoints[i].length; j++) {
	      if (instance.isMissing(i)) {
                vals[index] = Instance.missingValue();
	      } else if (currentVal <= m_CutPoints[i][j]) {
                vals[index] = 0;
	      } else {
                vals[index] = 1;
	      }
	      index++;
	    }
	  }   
	}
      } else {
        vals[index] = instance.value(i);
	index++;
      }
    }
    
    Instance inst = null;
    if (instance instanceof SparseInstance) {
      inst = new SparseInstance(instance.weight(), vals);
    } else {
      inst = new Instance(instance.weight(), vals);
    }
    copyValues(inst, false, instance.dataset(), getOutputFormat());    inst.setDataset(getOutputFormat());
    push(inst);
  }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] argv) {
        // TODO code application logic here
        try {
          if (Utils.getFlag('b', argv)) {
            Filter.batchFilterFile(new E_Discretize(), argv);
          } else {
            Filter.filterFile(new E_Discretize(), argv);
          }
        } catch (Exception ex) {
          System.out.println(ex.getMessage());
        }
    }
    
}
