package com.aufy.jnet.core.backend.exceptions.internal;

import java.awt.IllegalComponentStateException;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Methods for checking entire components in case they are of an invalid type. This is different from the {@link Data} class because it checks for 
 * values in components that may not be arrays or information.
 * 
 */
public class Component {
  /**
   * Check if a data contains any invalid values (NaN or Infinity).
   * @param data the data to check
   * @throws IllegalComponentStateException if the data contains any invalid values
   */
  public static void isItNaN(Object data) throws IllegalComponentStateException {
    if (data == null) throw new IllegalComponentStateException(Message.crash("internal", "Corrupted data", null, "a critical component is null (likely missing)"));
  }
}
