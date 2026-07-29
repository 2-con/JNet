package com.aufy.jnet.core.backend.exceptions.statistics;

import java.awt.IllegalComponentStateException;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Checks if a value is within bounds.
 */
public class General {
  /**
   * Checks if a double is a proper probability.
   * 
   * @param data the data to check
   * @throws IllegalComponentStateException if the double is less than 0 or more than 1
   */
  public static void isProbability(double data) throws IllegalComponentStateException {
    if (data < 0 || data > 1) throw new IllegalComponentStateException(Message.crash("statistical", "Illegal probability", null, "a proper probability must be between 0 and 1 inclusive"));
  }

}
