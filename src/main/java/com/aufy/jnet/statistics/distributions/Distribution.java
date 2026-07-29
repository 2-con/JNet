package com.aufy.jnet.statistics.distributions;

/**
 * Abstract class representing a distribution. Do not use this as it will return zeros for all methods; this class is meant to be inherited and 
 * implemented.
 */
public class Distribution {
  /**
   * Returns the arithmetic mean of the distribution.
   * 
   * @return the arithmetic mean of the distribution as a double.
   */
  public double mean() {
    return 0;
  }

  /**
   * Returns the variance of the distribution.
   * 
   * @return the variance of the distribution as a double.
   */
  public double variance() {
    return 0;
  }

  /**
   * Returns the standardDeviation of the distribution.
   * 
   * @return the variance of the distribution as a double.
   */
  public double standardDeviation() {
    return Math.sqrt(variance());
  }

  /**
   * Returns one sample from the distribution.
   * 
   * @return one sample from the distribution.
   */
  public double sample() {
    return 0;
  }
}
