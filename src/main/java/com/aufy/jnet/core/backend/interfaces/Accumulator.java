package com.aufy.jnet.core.backend.interfaces;

@FunctionalInterface
public interface Accumulator {
  double _apply(double accumulator, double newValue);

  /**
   * Applies a binary operation to all elements of an array, but updates an accumulator based on a starting value.
   * 
   * {@snippet :
   * double result = initial;
   * for (int i = 0; i < data.length; i++) {
   *   result = operation(result, data[i]);
   * }
   * return result;
   * }
   * 
   * @param data an array of double values.
   * @param operation an accumulator operation to apply to parse all elements, can either be a lambda expression or a reference to a method.
   * @return a double of parsing all the elements and applying the operation to an output accumulator.
   */
  public static double apply(double[] data, double initial, Accumulator operation) {
    double result = initial;
    for (int i = 0; i < data.length; i++) {
      result = operation._apply(result, data[i]);
    }
    return result;
  }
}
