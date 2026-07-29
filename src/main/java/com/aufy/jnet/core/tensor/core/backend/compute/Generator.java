package com.aufy.jnet.core.tensor.core.backend.compute;

import java.util.Arrays;
import java.util.stream.IntStream;

import com.aufy.jnet.core.backend.arrayops.Reductions;

/**
 * Generates a raw array of data accoding to a shape. Not to be confused with advanced initalizers or implementation-specific generators.
 *
 */
public class Generator {
  /*
  complex intializers go in math or nn
  */
  
  /**
   * Fills a raw memory block with a specific value.
   * 
   * @param value double value
   * @param shape shape of the tensor. Must have a valid shape
   * @return raw memory block
   */
  public static double[] fill(double value, int... shape) {
    int size = Reductions.prod(shape);
    double[] data = new double[size];
    Arrays.fill(data, value);

    return data;
  }

  /**
   * Fills a raw memory block with zeros.
   * 
   * @param shape shape of the tensor. Must have a valid shape
   * @return raw memory block
   */
  public static double[] zeros(int... shape) {
    int size = Reductions.prod(shape);
    
    return new double[size];
  }

  /**
   * Fills a raw memory block with ones.
   * 
   * @param shape shape of the tensor. Must have a valid shape
   * @return raw memory block
   */
  public static double[] ones(int... shape) {
    int size = Reductions.prod(shape);
    double[] data = new double[size];
    Arrays.fill(data, 1.0);

    return data;
  }

  /**
   * Fills a raw memory block with ordered numbers naively.
   * 
   * @param shape shape of the tensor. Must have a valid shape
   * @return raw memory block of ordered numbers
   */
  public static int[] arrange(int... shape) {
    return IntStream.range(0, Reductions.prod(shape) + 1).toArray();
  }
}
