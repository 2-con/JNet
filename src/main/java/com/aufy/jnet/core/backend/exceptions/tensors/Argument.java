package com.aufy.jnet.core.backend.exceptions.tensors;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Methods for checking arguments in the methods of tensors.
 */
public class Argument {

  /**
   * Checks if a double is above a limit. If its not, then it will throw an exception.
   * 
   * @param cutoff the cutoff to compare if data is above or not.
   * @param data the scalar to check.
   * @throws IllegalArgumentException
   */
  public static void aboveValue(double cutoff, double data, String operation) throws IllegalArgumentException {
    if (data < cutoff) throw new IllegalArgumentException(Message.crash("tensor argument", "Illegal parameter", operation , "parameter must be at least " + cutoff + " but got " + data));
  }

  /**
   * Checks if a double is below a limit. If its not, then it  will throw an exception.
   * 
   * @param cutoff the cutoff to compare if data is below or not.
   * @param data the scalar to check.
   * @throws IllegalArgumentException
   */
  public static void belowValue(double cutoff, double data, String operation) throws IllegalArgumentException {
    if (data > cutoff) throw new IllegalArgumentException(Message.crash("tensor argument", "Illegal parameter", operation , "parameter cannot exceed " + cutoff + " but got " + data));
  }

  /**
   * Checks if a double is between two values. If not, then it will throw an exception.
   * 
   * @param lower the lower inclusive bound.
   * @param upper the upper inclusive bound.
   * @param data the scalar to check.
   * @throws IllegalArgumentException
   */
  public static void betweenValue(double lower, double upper, double data, String operation) throws IllegalArgumentException {
    if (data > upper || data < lower) throw new IllegalArgumentException(Message.crash("tensor argument", "Illegal parameter", operation , "parameter must be between " + lower + " and " + upper + " but got " + data));
  }
}
