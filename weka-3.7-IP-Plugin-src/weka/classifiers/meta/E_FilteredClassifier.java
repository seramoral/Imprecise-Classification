package weka.classifiers.meta;

import weka.core.Capabilities.Capability;
import weka.core.Capabilities;
import weka.classifiers.Evaluation;
import weka.core.AdditionalMeasureProducer;
import weka.core.Instances;
import weka.filters.Filter;

import java.util.Enumeration;
import java.util.Vector;

/**
 * This class inherits from the FileteredClassifier class of weka. The only added functionality is the
 * implementation of AdditionalMeasureProducer interface.
 * 
 * @author Andres Masegosa, andrew@decsai.ugr.es
 */
public class E_FilteredClassifier extends FilteredClassifier implements  AdditionalMeasureProducer {

    /** for serialization */
    static final long serialVersionUID = 2889730616939923301L;
    
    
    double E_FilterTime;
    
    double E_ClassifierTime;
    
    /**
    * Default constructor.
    */
    public E_FilteredClassifier() {
        super();
    }

    /**
    * Returns default capabilities of the classifier.
    *
    * @return   the capabilities of this classifier
    */
    public Capabilities getCapabilities() {
        return super.getCapabilities();
    }

    /**
    * Returns an enumeration of any additional measure names that might be
    * in the classifier
    * @return an enumeration of the measure names
    */
    public Enumeration enumerateMeasures() {
        Vector newVector = new Vector();

        if (this.m_Classifier instanceof AdditionalMeasureProducer) {
          Enumeration en = ((AdditionalMeasureProducer)this.m_Classifier).
            enumerateMeasures();
          while (en.hasMoreElements()) {
            String mname = (String)en.nextElement();
            newVector.addElement(mname);
          }
        }

        if (this.m_Filter instanceof AdditionalMeasureProducer) {
          Enumeration en = ((AdditionalMeasureProducer)this.m_Filter).
            enumerateMeasures();
          while (en.hasMoreElements()) {
            String mname = (String)en.nextElement();
            newVector.addElement(mname);
          }
        }
        
        newVector.addElement("measureE_FilterTime");
        newVector.addElement("measureE_ClassifierTime");
        
        return newVector.elements();
    }
    
    
    /**
    * Returns the value of the named measure
    * @param measureName the name of the measure to query for its value
    * @return the value of the named measure
    * @exception IllegalArgumentException if the named measure is not supported
    */
    public double getMeasure(String additionalMeasureName) {

        if (this.m_Classifier instanceof AdditionalMeasureProducer) {
          try{
            return ((AdditionalMeasureProducer)m_Classifier).getMeasure(additionalMeasureName);
          }catch (Exception e1){
              if (this.m_Filter instanceof AdditionalMeasureProducer) {
                try{
                  return ((AdditionalMeasureProducer)m_Filter).getMeasure(additionalMeasureName);
                }catch (Exception e2){
                    if (additionalMeasureName.compareToIgnoreCase("measureE_FilterTime") == 0) {
                      return this.measureE_FilterTime();
                    }else if (additionalMeasureName.compareToIgnoreCase("measureE_ClassifierTime") == 0) {
                      return this.measureE_ClassifierTime();
                    }else{
                      throw new IllegalArgumentException("E1FilterClassifer: "
                                          +"Can't return value for : "+additionalMeasureName
                                          +". "+m_Classifier.getClass().getName()+" "
                                          +"is not an AdditionalMeasureProducer");
                    }                
                }
              }else {
                    if (additionalMeasureName.compareToIgnoreCase("measureE_FilterTime") == 0) {
                      return this.measureE_FilterTime();
                    }else if (additionalMeasureName.compareToIgnoreCase("measureE_ClassifierTime") == 0) {
                      return this.measureE_ClassifierTime();
                    }else{
                      throw new IllegalArgumentException("E2FilterClassifer: "
                                          +"Can't return value for : "+additionalMeasureName
                                          +". "+m_Classifier.getClass().getName()+" "
                                          +"is not an AdditionalMeasureProducer");
                    }                
              }
          }
        }else if (this.m_Filter instanceof AdditionalMeasureProducer) {
            try{    
                return ((AdditionalMeasureProducer)m_Filter).getMeasure(additionalMeasureName);
            }catch (Exception e){
                if (additionalMeasureName.compareToIgnoreCase("measureE_FilterTime") == 0) {
                  return this.measureE_FilterTime();
                }else if (additionalMeasureName.compareToIgnoreCase("measureE_ClassifierTime") == 0) {
                  return this.measureE_ClassifierTime();
                }else{
                  throw new IllegalArgumentException("E3FilterClassifer: "
                                      +"Can't return value for : "+additionalMeasureName
                                      +". "+m_Classifier.getClass().getName()+" "
                                      +"is not an AdditionalMeasureProducer");
                }                
            }
        }else {
            if (additionalMeasureName.compareToIgnoreCase("measureE_FilterTime") == 0) {
              return this.measureE_FilterTime();
            }else if (additionalMeasureName.compareToIgnoreCase("measureE_ClassifierTime") == 0) {
              return this.measureE_ClassifierTime();
            }else{
                throw new IllegalArgumentException("E4FilterClassifer: "
                                  +"Can't return value for : "+additionalMeasureName
                                  +". "+m_Classifier.getClass().getName()+" "
                                  +"is not an AdditionalMeasureProducer");
            }
        }

    }

    public double measureE_FilterTime() {
        return this.E_FilterTime;
    }
 
    public double measureE_ClassifierTime() {
        return this.E_ClassifierTime;
    }
     
    
  /**
   * Build the classifier on the filtered data.
   *
   * @param data the training data
   * @throws Exception if the classifier could not be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {

    if (m_Classifier == null) {
      throw new Exception("No base classifiers have been set!");
    }

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    
    
    m_Filter.setInputFormat(data);  // filter capabilities are checked here
    
    double start = System.currentTimeMillis();
    
    data = Filter.useFilter(data, m_Filter);
    
    double end = System.currentTimeMillis();
    this.E_FilterTime=(end-start)/1000.0;
    
    // can classifier handle the data?
    getClassifier().getCapabilities().testWithFail(data);

    m_FilteredInstances = data.stringFreeStructure();
    
    start = System.currentTimeMillis();
    
    m_Classifier.buildClassifier(data);
    
    end = System.currentTimeMillis();
    this.E_ClassifierTime=(end-start)/1000.0;
    
  }

  
  /**
   * Main method for testing this class.
   *
   * @param argv should contain the following arguments:
   * -t training file [-T test file] [-c class index]
   */
  public static void main(String [] argv) {

    try {
      System.out.println(Evaluation.evaluateModel(new E_FilteredClassifier(),
						  argv));
    } catch (Exception e) {
      System.err.println(e.getMessage());
    }
  }

}
