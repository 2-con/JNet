package com.aufy.jnet.core.backend.exceptions.internal;

import java.awt.IllegalComponentStateException;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Checks for invalid data in data (array or scalar).
 * 
 */
public class Data {
  /*
  all errors here should deal with the double[] data directly. the user would most likely clearer and friendlier errors set in graph
  up the chain anyways, so errors here should be ominous and look like they are coming from the depths of hell
  */

  /**
   * Check if a double[] data contains any invalid values (NaN or Infinity).
   * 
   * @param data the double[] data to check
   * @throws IllegalComponentStateException if the data contains any invalid values
   */
  public static void verifyData(double[] data) throws IllegalComponentStateException {
    if (data == null) throw new IllegalComponentStateException(Message.crash("internal", "Compromised data", null, "data is null"));
    if (data.length == 0) throw new IllegalComponentStateException(Message.crash("internal", "Compromised data", null, "data is empty"));
    for (double n : data) {
      if (Double.isNaN(n)) throw new IllegalComponentStateException(Message.crash("internal", "Bad data", null, "data contains NaN"));
      if (Double.isInfinite(n)) throw new IllegalComponentStateException(Message.crash("internal", "Bad data", null, "data contains Infinity"));
    }
  }

  /**
   * Check if an array data contains any invalid values (other data types, NaN, or Infinity).
   * 
   * @param data the double[] data to check
   * @throws IllegalComponentStateException if the data contains any invalid values
   */
  public static void verifyDataType(Object[] data) throws IllegalStateException {
    for (Object n : data) {
      if (!(n instanceof Number)) {
        throw new IllegalStateException(Message.crash("internal", "Compromised data", null, "data contains non-numeric value(s)"));
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
  public static void verifyEqualSizes(String operationName, double[] dataA, double[] dataB) throws NullPointerException {
    if (dataA.length != dataB.length) throw new NullPointerException(Message.crash("internal", "Incompatible data", operationName, "binary operation halted because of differing memory buffer sizes (" + dataA.length + " and " + dataB.length + ")"));
  }

  
}
