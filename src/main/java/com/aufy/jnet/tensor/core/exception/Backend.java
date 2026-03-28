package com.aufy.jnet.tensor.core.exception;

public class Backend {
  /*
  all errors here should deal with the double[] data directly. the user would most likely the clearer errors set in graph
  or shapes anyways, so be clear and punctual here
  */

  public static void nullCheck(double[] data) {
    for (double n : data) if (Double.isNaN(n)) throw new NullPointerException("Bad data: null value (NaN) detected inside tensor data");
  }

  public static void infCheck(double[] data) {
    for (double n : data) if (Double.isInfinite(n)) throw new NullPointerException("Bad data: infinity detected inside tensor data");
  }

  public static void sameMemorySize(double[] heapA, double[] heapB) {
    if (heapA.length == 0 || heapB.length == 0) throw new NullPointerException("Bad data: empty heap as tensor data");
    if (heapA.length != heapB.length) throw new NullPointerException("Mismatching heap sizes for operation: unable to perform operation because of differing memory buffer sizes");
  }
}
