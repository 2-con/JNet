package com.aufy.jnet.core.backend.arrayops;

import com.aufy.jnet.core.backend.interfaces.BinaryOp;

/**
 * Manages parallel operations for both integer and double arrays. Both arrays must be the same size; this class will not check nor ensure compatibility
 * 
 * <p>
 * This class provide simple bare-bones mathematical operations for integer and double arrays. All double implementations rely on
 * {@link com.aufy.jnet.core.backend.interfaces.BinaryOp} as a baseline tool.
 * <p>
 * 
 */
public class Elementwise {
  /**
   * Multiplies two arrays elementwise into a new array of the same size.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static double[] multiply(double[] array1, double[] array2) {
    return BinaryOp.apply(array1, array2, (a, b) -> a * b);
  }

  /**
   * Multiplies two arrays elementwise into a new array of the same size.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static int[] multiply(int[] array1, int[] array2) {
    int[] result = new int[array1.length];
    for (int i = 0; i < array1.length; i++) result[i] = array1[i] * array2[i];
    return result;
  }

  /**
   * Adds two arrays elementwise into a new array of the same size.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static double[] add(double[] array1, double[] array2) {
    return BinaryOp.apply(array1, array2, (a, b) -> a + b);
  }

  /**
   * Adds two arrays elementwise into a new array of the same size.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static int[] add(int[] array1, int[] array2) {
    int[] result = new int[array1.length];
    for (int i = 0; i < array1.length; i++) result[i] = array1[i] + array2[i];
    return result;
  }

  /**
   * Subtracts two arrays elementwise into a new array of the same size.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static double[] subtract(double[] array1, double[] array2) {
    return BinaryOp.apply(array1, array2, (a, b) -> a - b);
  }

  /**
   * Subtracts two arrays elementwise into a new array of the same size.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static int[] subtract(int[] array1, int[] array2) {
    int[] result = new int[array1.length];
    for (int i = 0; i < array1.length; i++) result[i] = array1[i] - array2[i];
    return result;
  }

  /**
   * Divides two arrays elementwise into a new array of the same size.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static double[] divide(double[] array1, double[] array2) {
    return BinaryOp.apply(array1, array2, (a, b) -> a / b);
  }

  /**
   * Divides two arrays elementwise into a new array of the same size.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static int[] divide(int[] array1, int[] array2) {
    int[] result = new int[array1.length];
    for (int i = 0; i < array1.length; i++) result[i] = array1[i] / array2[i];
    return result;
  }

  /**
   * Raises each element of a first array to the power of the corresponding element in a second array.
   * 
   * @param array1 the base array.
   * @param array2 the exponent array.
   * @return a new array containing the powered results.
   */
  public static double[] pow(double[] array1, double[] array2) {
    return BinaryOp.apply(array1, array2, (a, b) -> Math.pow(a, b));
  }

  /**
   * Raises each element of a first array to the power of the corresponding element in a second array.
   * 
   * @param array1 the base array.
   * @param array2 the exponent array.
   * @return a new array containing the powered results.
   */
  public static int[] pow(int[] array1, int[] array2) {
    int[] result = new int[array1.length];
    for (int i = 0; i < array1.length; i++) result[i] = (int) Math.pow(array1[i], array2[i]);
    return result;
  }

  /**
   * Computes the logarithm of each element in an array using a dynamic array of bases into a new array.
   * 
   * @param array1 the input array.
   * @param array2 the array of bases.
   * @return a new array containing the logarithms.
   */
  public static double[] log(double[] array1, double[] array2) {
    return BinaryOp.apply(array1, array2, (a, b) -> Math.log(a) / Math.log(b));
  }

  /**
   * Computes the logarithm of each element in an array using a dynamic array of bases into a new array.
   * 
   * @param array1 the input array.
   * @param array2 the array of bases.
   * @return a new array containing the logarithms.
   */
  public static int[] log(int[] array1, int[] array2) {
    int[] result = new int[array1.length];
    for (int i = 0; i < array1.length; i++) result[i] = (int) (Math.log(array1[i]) / Math.log(array2[i]));
    return result;
  }

  /**
   * Compares two arrays elementwise into a new array of the same size. If the values from each array are equal, the result is 1, otherwise 0.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static double[] equal(double[] array1, double[] array2) {
    return BinaryOp.apply(array1, array2, (a, b) -> a == b ? 1 : 0);
  }
  
  /**
   * Compares two arrays elementwise into a new array of the same size. If the values from each array are equal, the result is 1, otherwise 0.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static int[] equal(int[] array1, int[] array2) {
    int[] result = new int[array1.length];
    for (int i = 0; i < array1.length; i++) result[i] = array1[i] == array2[i] ? 1 : 0;
    return result;
  }

}
