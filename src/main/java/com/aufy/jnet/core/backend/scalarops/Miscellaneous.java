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
public class Miscellaneous {
  /**
   * Rounds a value to a specific scale.
   * 
   * @param value the double value to round.
   * @param scale up to how many decimal places to round.
   * @return a new rounded double.
   */
  public static double round(double value, int scale) {
    return Math.round(value * Math.pow(10, scale)) / Math.pow(10, scale);
  }

  /**
   * Truncates a value to the nearest integer.
   * 
   * @param value the double value to truncate.
   * @return a new truncated double.
   */
  public static double truncate(double value) {
    return Math.min(Math.floor(Math.max(0,value)),Math.ceil(value));
  }

  /**
   * Truncates a value to a specific scale.
   * 
   * @param value the double value to truncate.
   * @param scale up to how many decimal places to truncate.
   * @return a new truncated double.
   */
  public static double truncate(double value, int scale) {
    return truncate(value * Math.pow(10, scale)) / Math.pow(10, scale);
  }

  /**
   * Gets the fractional part of a value.
   * 
   * @param value the double value to get the fractional part of.
   * @param scale up to how many decimal places to truncate.
   * @return a new truncated double.
   */
  public static double fractional(double value, int scale) {
    return value - (truncate(value * Math.pow(10, scale)) / Math.pow(10, scale));
  }
}