package com.aufy.jnet.tensor.core.backend.func;

@FunctionalInterface
public interface Unary {
  double _apply(double x);

  /**
   * Applies a unary operation to all elements of an array.
   * 
   * @param data an array of double values
   * @param operation a unary operation (accepts one double and returns a double) to apply to each element, can either be a lambda expression or a reference to a method
   * @return a new array containing the results of applying the operation to each element of the input array
   */
  static double[] apply(double[] data, Unary operation) {
    double[] result = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      result[i] = operation._apply(data[i]);
    }
    return result;
  }
}
