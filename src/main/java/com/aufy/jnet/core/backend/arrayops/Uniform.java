package com.aufy.jnet.core.backend.arrayops;

import com.aufy.jnet.core.backend.interfaces.UnaryOp;

/**
 * Manages unary operations for both integer and double arrays and transforms each by a scalar.
 * 
 * <p>
 * This class provide simple bare-bones mathematical operations for integer and double arrays. All double implementations rely on
 * {@link com.aufy.jnet.core.backend.interfaces.UnaryOp} as a baseline tool.
 * <p>
 * 
 */
public class Uniform {
  /**
   * Adds a scalar value to each element of an array into a new array.
   * 
   * @param array the input array
   * @param scalar the value to add
   * @return a new array with the scalar added to each element
   */
  public static double[] add(double[] array, double scalar) {
    return UnaryOp.apply(array, n -> n + scalar);
  }

  /**
   * Adds a scalar value to each element of an array into a new array.
   * 
   * @param array the input array
   * @param scalar the value to add
   * @return a new array with the scalar added to each element
   */
  public static int[] add(int[] array, int scalar) {
    int[] result = new int[array.length];
    for (int i = 0; i < array.length; i++) result[i] = array[i] + scalar;
    return result;
  }

  /**
   * Subtracts a scalar value from each element of an array into a new array.
   * 
   * @param array the input array
   * @param scalar the value to subtract
   * @return a new array with the scalar subtracted from each element
   */
  public static double[] subtract(double[] array, double scalar) {
    return add(array, -scalar);
  }

  /**
   * Subtracts a scalar value from each element of an array into a new array.
   * 
   * @param array the input array
   * @param scalar the value to subtract
   * @return a new array with the scalar subtracted from each element
   */
  public static int[] subtract(int[] array, int scalar) {
    return add(array, -scalar);
  }

  /**
   * Multiplies each element of an array by a scalar value into a new array.
   * 
   * @param array the input array
   * @param scalar the value to multiply by
   * @return a new array with each element multiplied by the scalar
   */
  public static double[] multiply(double[] array, double scalar) {
    return UnaryOp.apply(array, n -> n * scalar);
  }

  /**
   * Multiplies each element of an array by a scalar value into a new array.
   * 
   * @param array the input array
   * @param scalar the value to multiply by
   * @return a new array with each element multiplied by the scalar
   */
  public static int[] multiply(int[] array, int scalar) {
    int[] result = new int[array.length];
    for (int i = 0; i < array.length; i++) result[i] = array[i] * scalar;
    return result;
  }

  /**
   * Divides each element of an array by a scalar value into a new array.
   * 
   * @param array the input array
   * @param scalar the value to divide by
   * @return a new array with each element divided by the scalar
   */
  public static double[] divide(double[] array, double scalar) {
    return multiply(array, 1 / scalar);
  }

  /**
   * Divides each element of an array by a scalar value into a new array.
   * 
   * @param array the input array
   * @param scalar the value to divide by
   * @return a new array with each element divided by the scalar
   */
  public static int[] divide(int[] array, int scalar) {
    return multiply(array, 1 / scalar);
  }

  /**
   * Raises each element of an array to the power of a scalar exponent into a new array. This method does not support int[] arrays and is not overloaded.
   * 
   * @param array the input array
   * @param exponent the power to raise each element to
   * @return a new array with elements raised to the given power
   */
  public static double[] pow(double[] array, double exponent) {
    return UnaryOp.apply(array, n -> Math.pow(n, exponent));
  }

  /**
   * Computes the logarithm of each element in an array using a specified scalar base into a new array.
   * This method does not support int[] arrays and is not overloaded.
   * 
   * @param array the input array
   * @param base the base of the logarithm
   * @return a new array containing the logarithms
   */
  public static double[] log(double[] array, double base) {
    return UnaryOp.apply(array, n -> Math.log(n) / Math.log(base));
  }

    /**
   * Raises a scalar base to the power of each element in an array into a new array.
   * This method does not support int[] arrays and is not overloaded.
   * 
   * @param base the base value
   * @param array the exponent array
   * @return a new array with the base raised to each element's power
   */
  public static double[] exp(double base, double[] array) {
    return UnaryOp.apply(array, n -> Math.pow(base, n));
  }

  /**
   * Computes the absolute value of each element in an array into a new array.
   * 
   * @param array the input array
   * @return a new array containing the absolute values
   */
  public static double[] abs(double[] array) {
    return UnaryOp.apply(array, n -> Math.abs(n));
  }

  /**
   * Computes the absolute value of each element in an array into a new array.
   * 
   * @param array the input array
   * @return a new array containing the absolute values
   */
  public static int[] abs(int[] array) {
    int[] result = new int[array.length];
    for (int i = 0; i < array.length; i++) result[i] = Math.abs(array[i]);
    return result;
  }

    /**
   * Negates each element of an array into a new array. This is a wrapper for {@link #multiply(double[], double)}.
   * 
   * @param array the input array
   * @return a new array with all elements negated
   */
  public static double[] negate(double[] array) {
    return multiply(array, -1);
  }

  /**
   * Negates each element of an array into a new array. This is a wrapper for {@link #multiply(double[], double)}.
   * 
   * @param array the input array
   * @return a new array with all elements negated
   */
  public static int[] negate(int[] array) {
    return multiply(array, -1);
  }

}
