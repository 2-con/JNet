package com.aufy.jnet.core.backend.arrayops;

import com.aufy.jnet.core.backend.interfaces.Accumulator;

/**
 * Manages raw mathematical operations for both integer and double arrays.
 * 
 * <p>
 * This class provide simple bare-bones mathematical operations for integer and double arrays. All double implementations rely on
 * {@link com.aufy.jnet.core.backend.interfaces.Accumulator} as a baseline tool.
 * <p>
 * 
 */
public class Reductions {
  
  /**
   * Sums the entire array.
   * 
   * @param array array.
   * @return sum.
   */
  public static double sum(double[] array) {
    return Accumulator.apply(array, 0, (a, b) -> a + b);
  }

  /**
   * Sums the entire array.
   * 
   * @param array array.
   * @return sum.
   */
  public static int sum(int[] array) {
    int output = 0;
    for (int n : array) output += n;
    return output;
  }

  /**
   * multiplies the entire array.
   * 
   * @param array array.
   * @return product.
   */
  public static double prod(double[] array) {
    return Accumulator.apply(array, 1, (a, b) -> a * b);
  }

  /**
   * multiplies the entire array.
   * 
   * @param array array.
   * @return product.
   */
  public static int prod(int[] array) {
    int output = 1;
    for (int n : array) output *= n;
    return output;
  }

  /**
   * Find the minimum value in an array.
   * 
   * @param array array.
   * @return minimum value.
   */
  public static double min(double[] array) {
    return Accumulator.apply(array, Double.MAX_VALUE, (a, b) -> (b < a) ? b : a);
  }

  /**
   * Find the minimum value in an array.
   * 
   * @param array array.
   * @return minimum value.
   */
  public static double min(int[] array) {
    int output = Integer.MAX_VALUE;
    for (int n : array) output = (n < output) ? n : output;
    return output;
  }

  /**
   * Find the maximum value in an array.
   * 
   * @param array array.
   * @return maximum value.
   */
  public static double max(double[] array) {
    return Accumulator.apply(array, Double.MIN_VALUE, (a, b) -> (b > a) ? b : a);
  }

  /**
   * Find the maximum value in an array.
   * 
   * @param array array.
   * @return maximum value.
   */
  public static int max(int[] array) {
    int output = Integer.MIN_VALUE;
    for (int n : array) output = (n > output) ? n : output;
    return output;
  }

  /**
   * Find the dot product of two arrays.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return dot product.
   */
  public static double dot(double[] array1, double[] array2) {
    return sum(Elementwise.multiply(array1, array2));
  }

  /**
   * Find the dot product of two arrays.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return dot product.
   */
  public static int dot(int[] array1, int[] array2) {
    return sum(Elementwise.multiply(array1, array2));
  }


}
