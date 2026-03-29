package com.aufy.jnet.tensor.core.exception;

import java.awt.IllegalComponentStateException;

public class Internal {
  /*
  all errors here should deal with the double[] data directly. the user would most likely clearer and friendlier errors set in graph
  up the chain anyways, so errors here should be ominous and look like they are coming from the depths of hell
  */

  private static String namingMessage(String operationName) {
    return (operationName == null || operationName.isBlank()) ? "" : " for " + operationName;
  }


  /**
   * Check if a double[] data contains any invalid values (NaN or Infinity).
   * This is a safety check to prevent errors in the backend.
   * @param data the double[] data to check
   * @throws IllegalComponentStateException if the data contains any invalid values
   */
  public static void verifyData(double[] data) throws IllegalComponentStateException {
    if (data == null) throw new IllegalComponentStateException("Compromised data: null memory buffer as tensor data");
    if (data.length == 0) throw new IllegalComponentStateException("Compromised data: empty memory buffer as tensor data");
    for (double n : data) {
      if (Double.isNaN(n)) throw new IllegalComponentStateException("Bad data: NaN detected inside tensor data");
      if (Double.isInfinite(n)) throw new IllegalComponentStateException("Bad data: infinity detected inside tensor data");
    }
  }

  public static void verifyDataType(Object[] data) throws IllegalStateException {
    for (Object n : data) {
      if (!(n instanceof Number)) {
        throw new IllegalStateException("Bad data: data must be a numeric type");
      }
    }
  }

  /**
   * Check if two heaps have the same memory size.
   * 
   * @param dataA first heap to check
   * @param dataB second heap to check
   * @throws NullPointerException if either heap is empty, or if they have different sizes
   */
  public static void verifyEqualHeapSizes(String operationName, double[] dataA, double[] dataB) throws NullPointerException {
    if (dataA.length != dataB.length) throw new NullPointerException("Mismatching data sizes" + namingMessage(operationName) + ": unable to perform operation because of differing memory buffer sizes");
  }
}
