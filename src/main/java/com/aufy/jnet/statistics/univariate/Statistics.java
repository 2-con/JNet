package com.aufy.jnet.statistics.univariate;

import java.util.Arrays;
import java.util.HashMap;

/**
 * Utility class for calculating order statistics, central tendency, and vector norms.
 */
public class Statistics {

  /**
   * Finds the minimum value in the array.
   *
   * @param array the input data
   * @return the smallest value
   */
  public static double min(double[] array) {
    double ans = Double.MAX_VALUE;
    for (double n : array) ans = (n < ans) ? n : ans;
    return ans;
  }

  /**
   * Finds the maximum value in the array.
   *
   * @param array the input data
   * @return the largest value
   */
  public static double max(double[] array) {
    double ans = Double.MIN_VALUE;
    for (double n : array) ans = (n > ans) ? n : ans;
    return ans;
  }
  
  /**
   * Computes the median value of the array elements.
   *
   * @param array the input data
   * @return the middle element of the sorted data
   */
  public static double median(double[] array) {
    double[] copy = array.clone();
    Arrays.sort(copy);
    return copy[array.length/2];
  }

  /**
   * Identifies the most frequently occurring value in the array.
   *
   * @param array the input data
   * @return the mode value
   */
  public static double mode(double[] array) {
    HashMap<Double, Integer> dict = new HashMap<>();
    
    // count
    for (double n : array) {
      if (dict.containsKey(n)) {
        dict.put(n, dict.get(n) + 1);
      } else {
        dict.put(n, 1);
      }
    }

    // find max
    int max = 0;
    double ans = 0;
    for (HashMap.Entry<Double, Integer> entry : dict.entrySet()) {
      if (entry.getValue() > max) {
        max = entry.getValue();
        ans = entry.getKey();
      }
    }

    return ans;
  }
  
  /**
   * Calculates the statistical range (spread) of the data.
   *
   * @param array the input data
   * @return the difference between the max and min values
   */
  public static double range(double[] array) {
    return max(array) - min(array);
  }
  
  /**
   * Computes the first quartile (25th percentile) value.
   *
   * @param array the input data
   * @return the value at the first quarter index
   */
  public static double quartile1(double[] array) {
    double[] copy = array.clone();
    Arrays.sort(copy);
    return copy[array.length/4];
  }
  
  /**
   * Computes the third quartile (75th percentile) value.
   *
   * @param array the input data
   * @return the value at the third quarter index
   */
  public static double quartile3(double[] array) {
    double[] copy = array.clone();
    Arrays.sort(copy);
    return copy[(array.length/4) * 3];
  }

  /**
   * Computes the Interquartile Range (IQR).
   *
   * @param array the input data
   * @return the difference between the third and first quartiles
   */
  public static double iqr(double[] array) {
    return quartile3(array) - quartile1(array);
  }

  /**
   * Computes the L1 norm (Manhattan distance/absolute sum) of the array.
   *
   * @param array the input data
   * @return the sum of absolute values
   */
  public static double l1Norm(double[] array) {
    double ans = 0;
    for (double n : array) ans += Math.abs(n);
    return ans;
  }

  /**
   * Computes the L2 norm (Euclidean norm/magnitude) of the array.
   *
   * @param array the input data
   * @return the square root of the sum of squared values
   */
  public static double l2Norm(double[] array) {
    double ans = 0;
    for (double n : array) ans += n * n;
    return Math.sqrt(ans);
  }
}
