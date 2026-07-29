package com.aufy.jnet.core.backend.interfaces;

@FunctionalInterface
public interface BinaryOp {
  double _apply(double dataA, double dataB);

  /**
   * Applies a binary operation to all elements of both arrays simultaneously.
   * 
   * {@snippet :
   * double[] result = new double[a.length];
   * for (int i = 0; i < a.length; i++) {
   *   result[i] = operation(a[i], b[i]);
   * }
   * return result;
   * }
   * 
   * @param data an array of double values
   * @param operation a binary operation to apply to each element, can either be a lambda expression or a reference to a method
   * @return a new array containing the results of applying the operation to each element of the input array
   */
  public static double[] apply(double[] dataA, double[] dataB, BinaryOp operation) {
    double[] result = new double[dataA.length];
    for (int i = 0; i < dataA.length; i++) {
      result[i] = operation._apply(dataA[i], dataB[i]);
    }
    return result;
  }
}
