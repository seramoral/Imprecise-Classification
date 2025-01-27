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

import java.util.ArrayList;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.RevisionUtils;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformationHandler;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.classifiers.credalClassifiers.credalNB.Cma;

import java.util.Enumeration;
import weka.core.AdditionalMeasureProducer;
import weka.core.Option;

/**
 * Credal Model Averaging (CMA)
 * 
 * About
 * Credal averaging over many naive Bayes classifiers, each characterized by a different feature set.
 * 
 * SYNOPSIS
 * Given a data set with k features, CMA performs exact averaging over 2^k naive Bayes classifiers,  each characterized by a different feature set.
 * The prior probability the classifiers is modeled  by a credal set rather than by a single prior distribution; therefore, CMA generalizes Bayesian model averaging to imprecise probabilities.
 * CMA returns more classes on the instances which are prior-dependent, namely when the most probable class varies when the prior over the models is varied.
 *
 */
public class CredalAveraging
        extends CredalClassifier
        implements TechnicalInformationHandler, AdditionalMeasureProducer {

    /** for serialization */
    static final long serialVersionUID = -1478242251770381214L;
    Cma credalNB = null;

    /**
     * Returns a string describing this classifier
     * @return a description of the classifier suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {
        String output = "Credal Model Averaging (CMA)" + "\n\nAbout:"
                + "\nCredal averaging over many naive Bayes classifiers, each characterized by a different feature set."
                + "\n\nSYNOPSIS:"
                + "\nGiven a data set with k features, CMA performs exact averaging over 2^k naive Bayes classifiers,  each characterized by a different feature set."
                + "The prior probability the classifiers is modeled  by a credal set rather than by a single prior distribution; therefore, CMA generalizes Bayesian model averaging to imprecise probabilities."
                + "CMA returns more classes on the instances which are prior-dependent, namely when the most probable class varies when the prior over the models is varied."
                + "\n\nFor more information, see\n\n"
                + getTechnicalInformation().toString()
                + "\n\nOptions:\n\n";

        Enumeration enu = this.listOptions();
        while (enu.hasMoreElements()) {
            Option option = (Option) enu.nextElement();
            output=output.concat(option.synopsis() + ' ');
            output=output.concat(option.description() + "\n");
        }


        return output;

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

        result = new TechnicalInformation(Type.INPROCEEDINGS);
        result.setValue(Field.AUTHOR, "G. Corani and M. Zaffalon");
        result.setValue(Field.YEAR, "2008");
        result.setValue(Field.TITLE, "Credal Model Averaging: an Extension of Bayesian Model Averaging to Imprecise Probabilities");
        result.setValue(Field.BOOKTITLE, "Proc. of the 2008 European Conference on Machine Learning and Knowledge Discovery in Databases (ECML-PKDD 2008)");
        result.setValue(Field.PAGES, "257--271");

        return result;
    }

    /**
     * Returns default capabilities of the classifier.
     *
     * @return      the capabilities of this classifier
     */
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capability.NOMINAL_ATTRIBUTES);

        // class
        result.enable(Capability.NOMINAL_CLASS);

        result.enable(Capability.MISSING_VALUES);
        return result;
    }

    /**
     * Generates the classifier.
     *
     * @param instances set of instances serving as training data
     * @exception Exception if the classifier has not been generated successfully
     */
    public void buildClassifier(Instances instances) throws Exception {

        initStats();
        int attIndex = 0;
        double sum;

        // can classifier handle the data?
        getCapabilities().testWithFail(instances);

        // remove instances with missing class
        instances = new Instances(instances);
        instances.deleteWithMissingClass();

        m_Instances = new Instances(instances, 0);

        ArrayList<int[]> TrainingSet = new ArrayList<int[]>();
        ArrayList<String> FeatureNames = new ArrayList<String>();
        ArrayList<String> classNames = new ArrayList<String>();
        ArrayList<Integer> numClassForEachFeature = new ArrayList<Integer>();
        ArrayList<Integer> SuppliedNonMarInTraining = new ArrayList<Integer>();
        ArrayList<Integer> SuppliedNonMarInTesting = new ArrayList<Integer>();
        ArrayList<Integer> SuppliedNumClassesNonMarTesting = new ArrayList<Integer>();


        //int[][] numClasses=new int[instances.numAttributes()][instances.classAttribute().numValues()];



        for (int i = 0; i < instances.numInstances(); i++) {
            Instance instance = instances.instance(i);
            int[] values = new int[instances.numAttributes()];
            for (int j = 0; j < instances.numAttributes(); j++) {
                if (Double.isNaN(instance.value(j))) {
                    values[j] = -9999;
                } else {
                    values[j] = (int) instance.value(j);
                }
            }
            //values[instances.numAttributes()]=instance.classIndex();

            TrainingSet.add(values);
        }

        Enumeration enume = instances.enumerateAttributes();
        while (enume.hasMoreElements()) {
            FeatureNames.add(((Attribute) enume.nextElement()).name());
        }

        for (int k = 0; k < instances.classAttribute().numValues(); k++) {
            classNames.add(instances.classAttribute().value(k));
        }


        enume = instances.enumerateAttributes();
        while (enume.hasMoreElements()) {
            Attribute att = ((Attribute) enume.nextElement());
            numClassForEachFeature.add(att.numValues());
        }

        //this.credalNB=new NaiveCredalClassifier2(TrainingSet, FeatureNames, classNames,
        //        numClassForEachFeature,SuppliedNonMarInTraining,SuppliedNonMarInTesting,SuppliedNumClassesNonMarTesting);
        //this.credalNB.setSValue(this.getSValue());

        this.credalNB = new Cma(100, instances.classAttribute().numValues());

        this.credalNB.train(TrainingSet, FeatureNames, classNames, numClassForEachFeature, (int) this.getSValue());

    }

    /**
     * Calculates the class membership probabilities for the given test instance.
     *
     * @param instance the instance to be classified
     * @return predicted class probability distribution
     * @exception Exception if distribution can't be computed
     */
    public double[] distributionForInstance(Instance instance) throws Exception {



        int[] values = new int[instance.numAttributes()];
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (Double.isNaN(instance.value(i))) {
                values[i] = -9999;
            } else {
                values[i] = (int) instance.value(i);
            }
        }

        int[] predictions = this.credalNB.classifyInstance(values);

        this.stats.updateStatistics(predictions, instance);

        double[] probs = new double[instance.numClasses()];
        for (int i = 0; i < probs.length; i++) {
            probs[i] = Double.NaN;
        }

        return probs;
    }

    public double classifyInstance(Instance instance) throws Exception {
        return Double.NaN;
    }

    /**
     * Returns the revision string.
     *
     * @return		the revision
     */
    public String getRevision() {
        return RevisionUtils.extract("$Revision: 0 $");
    }

    /**
     * Main method for testing this class.
     *
     * @param argv the options
     */
    public static void main(String[] argv) {
        runClassifier(new CredalAveraging(), argv);
    }
}
