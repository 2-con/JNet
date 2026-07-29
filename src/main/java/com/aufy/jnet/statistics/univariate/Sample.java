package com.aufy.jnet.statistics.univariate;

import java.util.Arrays;

import com.aufy.jnet.core.backend.arrayops.Tools;
import com.aufy.jnet.core.backend.exceptions.statistics.General;
import com.aufy.jnet.core.backend.exceptions.statistics.Parameter;

/**
 * Represents a data sample with descriptive and distribution functions.
 */
public class Sample {
  private double[] data = null;
  private double[] sortedData = null;
  private double[] reverseSortedData = null;

  /**
   * Constructs a new Sample and prepares sorted variations of the data.
   *
   * @param data the raw input data array
   */
  public Sample(double[] data) {
    this.data = data;
    this.sortedData = data.clone();
    Arrays.sort(sortedData);
    this.reverseSortedData = Tools.reverse(sortedData.clone());
  }

  /**
   * Returns a copy of the original raw data.
   *
   * @return a cloned array of raw values
   */
  public double[] dump() {
    return data.clone();
  }

  /**
   * Computes the quantile value for a given probability using the instance data. PPF (Percent Point Function) is another term for quantiles.
   *
   * @param p the probability threshold between 0.0 and 1.0
   * @return the calculated quantile value
   */
  public double PPF(double p) {
    General.isProbability(p);

    int index = (int) Math.ceil(p / this.sortedData.length) - 1;

    return this.sortedData[Math.max(0, index)];
  }

  /**
   * Computes the quantile value for a given probability using an arbitrary data array. PPF (Percent Point Function) is another term for quantiles.
   *
   * @param array the input data array
   * @param p the probability threshold between 0.0 and 1.0
   * @return the calculated quantile value
   */
  public static double PPF(double[] array, double p) {
    General.isProbability(p);

    double[] sortedData = array.clone();
    Arrays.sort(sortedData);

    int index = (int) Math.ceil(p / sortedData.length) - 1;
    return sortedData[Math.max(0, index)];
  }

  /**
   * Computes the Cumulative Distribution Function value for a given threshold.
   *
   * @param val the upper limit threshold
   * @return the proportion of values less than or equal to val
   */
  public double CDF(double val) {
    int count = 0;
    for (double x : this.sortedData) {
      if (x <= val) {
        count++;
      } else {
        break;
      }
    }

    return (double) count / this.sortedData.length;
  }

  /**
   * Computes the Cumulative Distribution Function value for an arbitrary array.
   *
   * @param array the input data array
   * @param val the upper limit threshold
   * @return the proportion of values less than or equal to val
   */
  public static double CDF(double[] array, double val) {
    double[] sortedData = array.clone();
    Arrays.sort(sortedData);

    int count = 0;
    for (double x : sortedData) {
      if (x <= val) {
        count++;
      } else {
        break;
      }
    }

    return (double) count / sortedData.length;
  }

  /**
   * Computes the Survival Function value (probability of exceeding a threshold).
   *
   * @param val the lower limit threshold
   * @return the proportion of values greater than or equal to val
   */
  public double SF(double val) {
    int count = 0;
    for (double x : this.reverseSortedData) {
      if (x >= val) {
        count++;
      } else {
        break;
      }
    }

    return (double) count / this.reverseSortedData.length;
  }

  /**
   * Computes the Survival Function value for an arbitrary array.
   *
   * @param array the input data array
   * @param val the lower limit threshold
   * @return the proportion of values greater than or equal to val
   */
  public static double SF(double[] array, double val) {
    double[] reverseSortedData = array.clone();
    Arrays.sort(reverseSortedData);
    reverseSortedData = Tools.reverse(reverseSortedData);

    int count = 0;
    for (double x : reverseSortedData) {
      if (x >= val) {
        count++;
      } else {
        break;
      }
    }

    return (double) count / reverseSortedData.length;
  }

  /**
   * Computes the probability of data falling within an interval.
   *
   * @param low the exclusive or inclusive lower bound
   * @param upper the exclusive or inclusive upper bound
   * @return the calculated interval probability
   */
  public double P(double low, double upper) {
    Parameter.isAboveValue(upper, low, "P (Probability Function)");
    
    return CDF(upper) - CDF(low);
  }

  /**
   * Computes the probability of data falling within an interval for an arbitrary array.
   *
   * @param array the input data array
   * @param low the exclusive or inclusive lower bound
   * @param upper the exclusive or inclusive upper bound
   * @return the calculated interval probability
   */
  public static double P(double[] array, double low, double upper) {
    Parameter.isAboveValue(upper, low, "P (Probability Function)");

    return CDF(array, upper) - CDF(array, low);
  }
  
}
