package com.aufy.jnet.core.backend.exceptions.statistics;

import java.awt.IllegalComponentStateException;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Checks if a value is within bounds.
 */
public class Parameter {
  /**
   * Checks if a double is positive non-zero or not.
   * 
   * @param data the data to check
   * @throws IllegalComponentStateException if the double is zero or negative
   */
  public static void isPositive(double data) throws IllegalComponentStateException {
    if (data <= 0) throw new IllegalComponentStateException(Message.crash("statistical initialization", "Illegal parameter", null, "Parameter cannot be negative or zero"));
  }

  /**
   * Checks if a double is above a cutoff.
   * 
   * @param cutoff the cutoff to compare if data is above or not
   * @param data the data to check
   * @throws IllegalComponentStateException if the double is less than cutoff
   */
  public static void isAboveValue(double cutoff, double data, String distribution) throws IllegalComponentStateException {
    if (data < cutoff) throw new IllegalComponentStateException(Message.crash("statistical initialization", "Illegal parameter", distribution , "parameter must be above " + cutoff));
  }

}
