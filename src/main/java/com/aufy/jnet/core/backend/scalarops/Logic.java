package com.aufy.jnet.core.backend.scalarops;

/**
 * Manages raw double processing for only scalar doubles. These are not array operations and are simply helper functions.
 * 
 * <p>
 * This class provide simple bare-bones operations that modify doubles not based on mathematics. This class contains methods for processing doubles
 * over mathematical tools; for that, see {@link com.aufy.jnet.core.backend.scalarops.Functions}
 * <p>
 * 
 */
public class Logic {
  /**
   * Compares two values. If the values from each array are equal, the result is 1, otherwise 0.
   * 
   * @param array1 the first array.
   * @param array2 the second array.
   * @return a new array of the same size.
   */
  public static double equal(double input1, double input2) {
    return input1 == input2 ? 1 : 0;
  }

}