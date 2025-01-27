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
 *    PairedTTester.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */


package weka.experiment;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Range;
import weka.core.RevisionHandler;
import weka.core.RevisionUtils;
import weka.core.Utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Serializable;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Enumeration;
import java.util.Vector;

/**
 * Calculates T-Test statistics on data stored in a set of instances. <p/>
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D &lt;index,index2-index4,...&gt;
 *  Specify list of columns that specify a unique
 *  dataset.
 *  First and last are valid indexes. (default none)</pre>
 * 
 * <pre> -R &lt;index&gt;
 *  Set the index of the column containing the run number</pre>
 * 
 * <pre> -F &lt;index&gt;
 *  Set the index of the column containing the fold number</pre>
 * 
 * <pre> -G &lt;index1,index2-index4,...&gt;
 *  Specify list of columns that specify a unique
 *  'result generator' (eg: classifier name and options).
 *  First and last are valid indexes. (default none)</pre>
 * 
 * <pre> -S &lt;significance level&gt;
 *  Set the significance level for comparisons (default 0.05)</pre>
 * 
 * <pre> -V
 *  Show standard deviations</pre>
 * 
 * <pre> -L
 *  Produce table comparisons in Latex table format</pre>
 * 
 * <pre> -csv
 *  Produce table comparisons in CSV table format</pre>
 * 
 * <pre> -html
 *  Produce table comparisons in HTML table format</pre>
 * 
 * <pre> -significance
 *  Produce table comparisons with only the significance values</pre>
 * 
 * <pre> -gnuplot
 *  Produce table comparisons output suitable for GNUPlot</pre>
 * 
 <!-- options-end -->
 *
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision: 5415 $
 */
public class ShowMeanNoTester extends PairedTTester  {
  
  /** for serialization */
  static final long serialVersionUID = 8370014624008728610L;


  /**
   * Computes a paired t-test comparison for a specified dataset between
   * two resultsets.
   *
   * @param datasetSpecifier the dataset specifier
   * @param resultset1Index the index of the first resultset
   * @param resultset2Index the index of the second resultset
   * @param comparisonColumn the column containing values to compare
   * @return the results of the paired comparison
   * @throws Exception if an error occurs
   */
  public PairedStats calculateStatistics(Instance datasetSpecifier,
					 int resultset1Index,
					 int resultset2Index,
					 int comparisonColumn) throws Exception {

    if (m_Instances.attribute(comparisonColumn).type()
	!= Attribute.NUMERIC) {
      throw new Exception("Comparison column " + (comparisonColumn + 1)
			  + " ("
			  + m_Instances.attribute(comparisonColumn).name()
			  + ") is not numeric");
    }
    if (!m_ResultsetsValid) {
      prepareData();
    }

    Resultset resultset1 = (Resultset) m_Resultsets.elementAt(resultset1Index);
    Resultset resultset2 = (Resultset) m_Resultsets.elementAt(resultset2Index);
    FastVector dataset1 = resultset1.dataset(datasetSpecifier);
    FastVector dataset2 = resultset2.dataset(datasetSpecifier);
    String datasetName = templateString(datasetSpecifier);
    if (dataset1 == null) {
      throw new Exception("No results for dataset=" + datasetName
			 + " for resultset=" + resultset1.templateString());
    } else if (dataset2 == null) {
      throw new Exception("No results for dataset=" + datasetName
			 + " for resultset=" + resultset2.templateString());
    } else if (dataset1.size() != dataset2.size()) {
      throw new Exception("Results for dataset=" + datasetName
			  + " differ in size for resultset="
			  + resultset1.templateString()
			  + " and resultset="
			  + resultset2.templateString()
			  );
    }

    PairedStats pairedStats = new PairedStats(m_SignificanceLevel);

    for (int k = 0; k < dataset1.size(); k ++) {
      Instance current1 = (Instance) dataset1.elementAt(k);
      Instance current2 = (Instance) dataset2.elementAt(k);
      if (current1.isMissing(comparisonColumn)) {
	//System.err.println("Instance has missing value in comparison "
	//		   + "column!\n" + current1);
	continue;
      }
      if (current2.isMissing(comparisonColumn)) {
	//System.err.println("Instance has missing value in comparison "
	//		   + "column!\n" + current2);
	continue;
      }
      if (current1.value(m_RunColumn) != current2.value(m_RunColumn)) {
	System.err.println("Run numbers do not match!\n"
			    + current1 + current2);
      }
      if (m_FoldColumn != -1) {
	if (current1.value(m_FoldColumn) != current2.value(m_FoldColumn)) {
	  System.err.println("Fold numbers do not match!\n"
			     + current1 + current2);
	}
      }
      double value1 = current1.value(comparisonColumn);
      double value2 = current2.value(comparisonColumn);
      pairedStats.add(value1, value2);
    }
    pairedStats.calculateDerived();
    //System.err.println("Differences stats:\n" + pairedStats.differencesStats);
    return pairedStats;

  }

  /**
   * Creates a comparison table where a base resultset is compared to the
   * other resultsets. Results are presented for every dataset.
   *
   * @param baseResultset the index of the base resultset
   * @param comparisonColumn the index of the column to compare over
   * @return the comparison table string
   * @throws Exception if an error occurs
   */
  public String multiResultsetFull(int baseResultset,
				   int comparisonColumn) throws Exception {

    int maxWidthMean = 2;
    int maxWidthStdDev = 2;
    
    double[] sortValues = new double[getNumDatasets()];
      
    // determine max field width
    for (int i = 0; i < getNumDatasets(); i++) {
      sortValues[i] = Double.POSITIVE_INFINITY;  // sorts skipped cols to end
      
      for (int j = 0; j < getNumResultsets(); j++) {
        if (!displayResultset(j))
          continue;
	try {
	  PairedStats pairedStats = 
	    calculateStatistics(m_DatasetSpecifiers.specifier(i), 
				j, j, comparisonColumn);
          if (!Double.isInfinite(pairedStats.yStats.mean) &&
              !Double.isNaN(pairedStats.yStats.mean)) {
            double width = ((Math.log(Math.abs(pairedStats.yStats.mean)) / 
                             Math.log(10))+1);
            if (width > maxWidthMean) {
              maxWidthMean = (int)width;
            }
          }

          if (j == baseResultset) {
            if (getSortColumn() != -1)
              sortValues[i] = calculateStatistics(
                                m_DatasetSpecifiers.specifier(i), 
                                baseResultset, j, getSortColumn()).xStats.mean;
            else
              sortValues[i] = i;
          }
	  
	  if (m_ShowStdDevs &&
              !Double.isInfinite(pairedStats.yStats.stdDev) &&
              !Double.isNaN(pairedStats.yStats.stdDev)) {
	    double width = ((Math.log(Math.abs(pairedStats.yStats.stdDev)) / 
                             Math.log(10))+1);
	    if (width > maxWidthStdDev) {
	      maxWidthStdDev = (int)width;
	    }
	  }
	}  catch (Exception ex) {
	  //ex.printStackTrace();
          System.err.println(ex);
	}
      }
    }

    // sort rows according to sort column
    m_SortOrder = Utils.sort(sortValues);

    // determine column order
    m_ColOrder = new int[getNumResultsets()];
    m_ColOrder[0] = baseResultset;
    int index = 1;
    for (int i = 0; i < getNumResultsets(); i++) {
      if (i == baseResultset)
        continue;
      m_ColOrder[index] = i;
      index++;
    }

    // setup matrix
    initResultMatrix();    
    m_ResultMatrix.setRowOrder(m_SortOrder);
    m_ResultMatrix.setColOrder(m_ColOrder);
    m_ResultMatrix.setMeanWidth(maxWidthMean);
    m_ResultMatrix.setStdDevWidth(maxWidthStdDev);
    m_ResultMatrix.setSignificanceWidth(1);

    // make sure that test base is displayed, even though it might not be
    // selected
    for (int i = 0; i < m_ResultMatrix.getColCount(); i++) {
      if (    (i == baseResultset)
           && (m_ResultMatrix.getColHidden(i)) ) {
        m_ResultMatrix.setColHidden(i, false);
        System.err.println("Note: test base was hidden - set visible!");
      }
    }
    
    // the data
    for (int i = 0; i < getNumDatasets(); i++) {
      m_ResultMatrix.setRowName(i, 
          templateString(m_DatasetSpecifiers.specifier(i)));

      for (int j = 0; j < getNumResultsets(); j++) {
        try {
          // calc stats
          PairedStats pairedStats = 
            calculateStatistics(m_DatasetSpecifiers.specifier(i), 
                j, j, comparisonColumn);

          // count
          m_ResultMatrix.setCount(i, pairedStats.count);

          // mean
          m_ResultMatrix.setMean(j, i, pairedStats.yStats.mean);
          
          // std dev
          m_ResultMatrix.setStdDev(j, i, pairedStats.yStats.stdDev);

          // significance
          if (pairedStats.differencesSignificance < 0)
            m_ResultMatrix.setSignificance(j, i, ResultMatrix.SIGNIFICANCE_WIN);
          else if (pairedStats.differencesSignificance > 0)
            m_ResultMatrix.setSignificance(j, i, ResultMatrix.SIGNIFICANCE_LOSS);
          else
            m_ResultMatrix.setSignificance(j, i, ResultMatrix.SIGNIFICANCE_TIE);
        }
        catch (Exception e) {
          //e.printStackTrace();
          System.err.println(e);
        }
      }
    }

    // generate output
    StringBuffer result = new StringBuffer(1000);
    try {
      result.append(m_ResultMatrix.toStringMatrix());
    }
    catch (Exception e) {
      e.printStackTrace();
    }
    
    // append a key so that we can tell the difference between long
    // scheme+option names
    result.append("\n\n" + m_ResultMatrix.toStringKey());

    return result.toString();
  }


  /**
   * returns a string that is displayed as tooltip on the "perform test"
   * button in the experimenter
   * 
   * @return	the tool tip
   */
  public String getToolTipText() {
    return "Performs test using t-test statistic";
  }

  /**
   * returns the name of the tester
   * 
   * @return	the display name
   */
  public String getDisplayName() {
    return "ShowMean No-Tester";
  }
  
  /**
   * Returns the revision string.
   * 
   * @return		the revision
   */
  public String getRevision() {
    return RevisionUtils.extract("$Revision: 5415 $");
  }
  
  /**
   * Test the class from the command line.
   *
   * @param args contains options for the instance ttests
   */
  public static void main(String args[]) {

    try {
      ShowMeanNoTester tt = new ShowMeanNoTester();
      String datasetName = Utils.getOption('t', args);
      String compareColStr = Utils.getOption('c', args);
      String baseColStr = Utils.getOption('b', args);
      boolean summaryOnly = Utils.getFlag('s', args);
      boolean rankingOnly = Utils.getFlag('r', args);
      try {
	if ((datasetName.length() == 0)
	    || (compareColStr.length() == 0)) {
	  throw new Exception("-t and -c options are required");
	}
	tt.setOptions(args);
	Utils.checkForRemainingOptions(args);
      } catch (Exception ex) {
	String result = "";
	Enumeration enu = tt.listOptions();
	while (enu.hasMoreElements()) {
	  Option option = (Option) enu.nextElement();
	  result += option.synopsis() + '\n'
	    + option.description() + '\n';
	}
	throw new Exception(
	      "Usage:\n\n"
	      + "-t <file>\n"
	      + "\tSet the dataset containing data to evaluate\n"
	      + "-b <index>\n"
	      + "\tSet the resultset to base comparisons against (optional)\n"
	      + "-c <index>\n"
	      + "\tSet the column to perform a comparison on\n"
	      + "-s\n"
	      + "\tSummarize wins over all resultset pairs\n\n"
	      + "-r\n"
	      + "\tGenerate a resultset ranking\n\n"
	      + result);
      }
      Instances data = new Instances(new BufferedReader(
				  new FileReader(datasetName)));
      tt.setInstances(data);
      //      tt.prepareData();
      int compareCol = Integer.parseInt(compareColStr) - 1;
      System.out.println(tt.header(compareCol));
      if (rankingOnly) {
	System.out.println(tt.multiResultsetRanking(compareCol));
      } else if (summaryOnly) {
	System.out.println(tt.multiResultsetSummary(compareCol));
      } else {
	System.out.println(tt.resultsetKey());
	if (baseColStr.length() == 0) {
	  for (int i = 0; i < tt.getNumResultsets(); i++) {
            if (!tt.displayResultset(i))
              continue;
	    System.out.println(tt.multiResultsetFull(i, compareCol));
	  }
	} else {
	  int baseCol = Integer.parseInt(baseColStr) - 1;
	  System.out.println(tt.multiResultsetFull(baseCol, compareCol));
	}
      }
    } catch(Exception e) {
      e.printStackTrace();
      System.err.println(e.getMessage());
    }
  }
}
