package com.aufy.jnet.statistics.univariate;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.interfaces.UnaryOp;
/**
 * Utility class for calculating statistical moments of data arrays.
 */
public class Moments {
  
  /**
   * Calculates the arithmetic mean of the array elements.
   *
   * @param array the input data
   * @return the average value
   */
  public static double mean(double[] array) {
    return Reductions.sum(array) / array.length;
  }
  
  /**
   * Calculates the population variance of the array elements.
   *
   * @param array the input data
   * @return the variance measure
   */
  public static double variance(double[] array) {
    double mu = mean(array);
    double[] shifted = UnaryOp.apply(new double[array.length], n -> Math.pow(n - mu, 2));

    return mean(shifted);
  }

  /**
   * Calculates the population standard deviation of the array elements. This is a wrapper for the variance function.
   *
   * @param array the input data
   * @return the standard deviation measure
   */
  public static double standardDeviation(double[] array) {
    return Math.sqrt(variance(array));
  }

  /**
   * Calculates the Fisher-Pearson skewness to measure dataset asymmetry.
   *
   * @param array the input data
   * @return the skewness value
   */
  public static double skew(double[] array) {
    double mu = mean(array);
    double stdeviation = standardDeviation(array);

    double[] shifted = UnaryOp.apply(new double[array.length], n -> Math.pow(n - mu, 3));

    return mean(shifted) / Math.pow(stdeviation, 3);
  }
  
  /**
   * Calculates the population kurtosis to measure dataset tailedness.
   *
   * @param array the input data
   * @return the kurtosis value
   */
  public static double kurtosis(double[] array) {
    double mu = mean(array);
    double stdeviation = standardDeviation(array);

    double[] shifted = UnaryOp.apply(new double[array.length], n -> Math.pow(n - mu, 4));

    return mean(shifted) / Math.pow(stdeviation, 4);
  }
}
