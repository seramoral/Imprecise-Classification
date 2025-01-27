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

package weka.classifiers.credalClassifiers;

import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.Classifier;
import utils.CredalClassificationStatistics;
import weka.core.AdditionalMeasureProducer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;

/**
 * 
 * This is an abstract class as parent for any credal classifier
 *
 *
 * @author Andres Masegosa, andrew@decsai.ugr.es
 */
public abstract class CredalClassifier extends Classifier implements AdditionalMeasureProducer, OptionHandler {

    /** for serialization */
    static final long serialVersionUID = -1669920769968646583L;

    /** The instances used for training. */
    protected Instances m_Instances;

    /** The path of the folder where the predictiosn are stored**/
    protected String m_outputFolder = "./results/";

    /** Flag to indicates if outputs are or not stored**/
    protected boolean m_OutputFlag = false;

    /** Field to manage the Credal Classification statistics**/
    protected CredalClassificationStatistics stats;

    /** Parameter for the IDM**/
    protected double m_SValue = 1.0;


    /**
     * Set the path of the folder where the predictiosn are stored
     * @param name, a folder path endindin in '/'. The folder must be
     * previously created.
     */
    public void setOutputFolder(String name) {
        this.m_outputFolder = name;
    }

    /**
     * Get the path of the folder where the predictiosn are stored
     * @return
     */
    public String getOutputFolder() {
        return this.m_outputFolder;
    }

    /**
     * Flag to indicates if outputs are or not stored
     * @param flag
     */
    public void setOutputFlag(boolean flag) {
        this.m_OutputFlag = flag;
    }

    /**
     * Return the state of the flag that indicates if outputs are or not stored
     * @return
     */
    public boolean getOutputFlag() {
        return this.m_OutputFlag;
    }

    /**
     * It should be called by the credal classifiers to initialize the statistics.
     */
    public void initStats() {

        if (this.stats == null) {
            this.stats = new CredalClassificationStatistics(this);
            this.stats.setOutputFlag(this.getOutputFlag());
            this.stats.setOutputFolder(this.getOutputFolder());
            this.stats.setEnumerateMeasures(true);
            this.stats.updateStatistics(null, null);
            this.stats.setEnumerateMeasures(false);
        }

        this.stats.resetValues();

    }

    /**
     * This method is called for each test instance.
     * @param nonDominatedClasses, array with contain the index of those classes
     * that are non dominated
     * @param instance, the instance to be classified.
     */
    public void updateStatistics(int[] nonDominatedClasses, Instance instance) {
        this.stats.updateStatistics(nonDominatedClasses, instance);
    }

    /**
     * Return the value of the parameter 's' of the IDM model.
     * @return
     */
    public double getSValue() {
        return this.m_SValue;
    }

    /**
     * Set the value of the parameter 's' of the IDM model.
     * @param
     */
    public void setSValue(double value) {
        this.m_SValue = value;
    }

    /**
     * Gets the current settings of the classifier.
     *
     * @return an array of strings suitable for passing to setOptions
     */
    public String[] getOptions() {
        String[] options = new String[5];
        int current = 0;

        options[current++] = "-S";
        options[current++] = "" + getSValue();

        options[current++] = "-O";
        options[current++] = "" + this.getOutputFolder();

        if (this.m_OutputFlag)
            options[current++] = "-OFlag";

        while (current < options.length) {
            options[current++] = "";
        }

        return options;

    }

    /**
     * Parses the options for this object.
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    public void setOptions(String[] options) throws Exception {

        String convertList = Utils.getOption("S", options);
        if (convertList.length() != 0) {
            this.setSValue(Double.parseDouble(convertList));
        } else {
            this.setSValue(1);
        }

        convertList = Utils.getOption("O", options);
        if (convertList.length() != 0) {
            this.setOutputFolder(convertList);
        } else {
            this.setOutputFolder("./results/");
        }

        if (Utils.getFlag("OFlag", options))
            this.setOutputFlag(true);
        else
            this.setOutputFlag(false);

    }

    /**
     * Gets an enumeration describing the available options.
     *
     * @return an enumeration of all the available options.
     */
    public Enumeration listOptions() {

        Vector newVector = new Vector(3);

        newVector.addElement(new Option(
               "The number of ''hidden'' instances which controls the strength of the prior (for credal classifiers, higher s implies higher indeterminacy)",
               "S", 1, "-S svalue"));

        newVector.addElement(new Option(
               "\tSpecifies the output folder where the credal predictions (necessary for the Friedman test) " +
               "are stored. This folder should exist and be completely empty.",
               "O", 1, "-O folder-path"));

        newVector.addElement(new Option(
               "\tSet this flag to true is you need to make a comparison of the credal predictions using the Friedman test " +
               "(as detailed in Section 5 of the user manual). After the experiment is nished, proceed as indicated in " +
               "Section 7 of the user manual to carry out this comparison.",
               "OFlag", 0, "-OFlag"));

        return newVector.elements();
    }

    /**
     * Returns an enumeration of the additional measure names
     * @return an enumeration of the measure names
     */
    public Enumeration enumerateMeasures() {
        Vector newVector = new Vector();

        this.stats = new CredalClassificationStatistics(this);
        this.stats.setOutputFlag(this.getOutputFlag());
        this.stats.setOutputFolder(this.getOutputFolder());
        this.stats.setEnumerateMeasures(true);
        this.stats.updateStatistics(null, null);
        newVector.addAll(this.stats.enumerateMeasures());
        this.stats.setEnumerateMeasures(false);

        return newVector.elements();
    }

    /**
     * Returns the value of the named measure
     * @param measureName the name of the measure to query for its value
     * @return the value of the named measure
     * @exception IllegalArgumentException if the named measure is not supported
     */
    public double getMeasure(String additionalMeasureName) {

        return this.stats.getMeasure(additionalMeasureName);
    }

    /**
   * Returns a description of the classifier.
   *
   * @return a description of the classifier as a string.
   */
  public String toString() {

      return "Credal Classifer: Information can not be displayed.";
  }
  
  /**
   * It returns the indices of the non-dominated class values
   * @param non_dominated_states the array indicating which states are dominated
   * @return the non-dominated index set
   */
  
    public int[] getNonDominatedIndexSet(boolean[] non_dominated_states){
        int[] non_dominated_index_set;
        int num_non_dominated_states = 0;
        boolean non_dominated;
        int num_class_values = non_dominated_states.length;
        int cont;
        
        //Calculate the number of non-dominated states
        
        for (int i=0; i< num_class_values; i++){
            non_dominated = non_dominated_states[i];  
        
            if (non_dominated)
                num_non_dominated_states++;
        }
        
        non_dominated_index_set = new int[num_non_dominated_states];
        
        cont = 0;
        
        for (int i=0; i< num_class_values; i++){
            non_dominated = non_dominated_states[i];  
        
            if (non_dominated){
                non_dominated_index_set[cont]=i;
                cont++;
            }
        }
        
        return non_dominated_index_set;
    }

}
