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
public class Unary {
  /**
   * Heavyside step function. Returns 1 if the value is greater than 0, and 0 otherwise.
   * 
   * @param value the double value to evaluate.
   * @return a new double value.
   */
  public static double step(double value) {
    return (value > 0) ? 1 : 0;
  }

  /**
   * Sigmoid function.
   * 
   * @param value the double value to evaluate.
   * @return a new double value.
   */
  public static double sigmoid(double value) {
    return 1.0/(1.0 + Math.exp(-value));
  }

  /**
   * Returns the identity of the value.
   * 
   * @param value the double value to evaluate.
   * @return the same value.
   */
  public static double identity(double value) {
    return value;
  }

}