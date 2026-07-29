package com.aufy.jnet.statistics.bivariate;

import com.aufy.jnet.core.backend.arrayops.Elementwise;
import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Uniform;
import com.aufy.jnet.statistics.univariate.Moments;

/**
 * Utility class for calculating statistical dependencies and joint relationships between two data streams.
 */
public class Correlation {
  
  /**
   * Computes the population cross-covariance between two equal-length arrays.
   *
   * @param a the first data series
   * @param b the second data series
   * @return the joint variance measure
   */
  public static double crossCovariance(double[] a, double[] b) {
    double meanX = Moments.mean(a);
    double meanY = Moments.mean(b);

    double[] shiftedX = Uniform.subtract(a, meanX);
    double[] shiftedY = Uniform.subtract(b, meanY);
    return Reductions.sum(Elementwise.multiply(shiftedX, shiftedY)) / a.length;
  }

  /**
   * Computes the Pearson cross-correlation coefficient between two arrays.
   *
   * @param a the first data series
   * @param b the second data series
   * @return the correlation coefficient scaled between -1.0 and 1.0, or 0.0 if variance is zero
   */
  public static double crossCorrelation(double[] a, double[] b) {
    double cov = crossCovariance(a, b);

    double stdX = Moments.variance(a);
    double stdY = Moments.standardDeviation(b);
    if (stdX == 0 || stdY == 0) return 0.0;

    return cov / (stdX * stdY);
  }

}
