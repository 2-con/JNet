package com.aufy.jnet.statistics;

import java.util.Arrays;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.statistics.distributions.Distribution;

/**
 * Generate a sample of datapoints based off a distribution. Can either be used as a static method or as an instance method, but both do the exact same thing.
 */
public class Generator {
  /**
   * Generate a sample of datapoints based off a distribution.
   * 
   * @param distribution the distribution.
   * @param shape the shape of the sample, it determines how many datapoints to generate.
   * @return the sample.
   */
  public static double[] sample(Distribution distribution, int... shape) {
    int size = Reductions.prod(shape);
    double[] data = new double[size];

    for (int i = 0; i < size; i++) {
      data[i] = distribution.sample();
    }
    return data;
  }

  /**
   * Generate a sample of datapoints based off a distribution.
   * 
   * @param distribution the distribution.
   * @param size the amount of datapoints to generate.
   * @return the sample.
   */
  public static double[] sample(Distribution distribution, int size) {
    double[] data = new double[size];

    for (int i = 0; i < size; i++) {
      data[i] = distribution.sample();
    }
    return data;
  }

  /**
   * Generates a raw memory block filled with zeros.
   * 
   * @param shape the shape of the memory block.
   * @return double array of zeros.
   */
  public static double[] zeros(int... shape) {
    int size = Reductions.prod(shape);
    return new double[size];
  }

  /**
   * Generates a raw memory block filled with ones.
   * 
   * @param shape the shape of the memory block.
   * @return double array of ones.
   */
  public static double[] ones(int... shape) {
    int size = Reductions.prod(shape);
    double[] data = new double[size];
    Arrays.fill(data, 1.0);
    return data;
  }
}
