package com.aufy.jnet.core.backend.interfaces;

@FunctionalInterface
public interface UnaryOp {
  double _apply(double data);

  /** 
   * Applies a unary operation to all elements of an array.
   * 
   * A java equivalent of the method is provided below:
   * 
   * {@snippet :
   * double[] result = new double[data.length];
   * for (int i = 0; i < data.length; i++) {
   *   result[i] = operation(data[i]);
   * }
   * return result;
   * }
   * 
   * @param data an array of double values
   * @param operation a unary operation (accepts one double and returns a double) to apply to each element, can either be a lambda expression or a reference to a method
   * @return a new array containing the results of applying the operation to each element of the input array
   */
  public static double[] apply(double[] data, UnaryOp operation) {
    double[] result = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      result[i] = operation._apply(data[i]);
    }
    return result;
  }
}
