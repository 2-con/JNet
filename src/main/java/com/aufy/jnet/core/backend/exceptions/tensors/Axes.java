package com.aufy.jnet.core.backend.exceptions.tensors;

import java.util.HashSet;
import java.util.Set;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Methods for checking axes in tensors. This deals with operations relating to the axes themselves and not shaping operations, for that see {@link Geometry}.
 */
public class Axes {

  /**
   * Throws an exception if the axis is out of bounds.
   * 
   * @param rank rank of the tensor
   * @param axis axis to verify
   * @throws IndexOutOfBoundsException
   */
  public static void verifyAxis(String operationName, int rank, int axis) throws IndexOutOfBoundsException {
    if (axis < 0 || axis >= rank) {
      throw new IndexOutOfBoundsException(Message.crash("tensor indexing", "Axis out of bounds", operationName, axis + " is out of bounds for a tensor of rank " + rank));
    }
  }

  /**
   * Throws an exception if the axes are not unique.
   * 
   * @param axes axes to verify
   * @throws IllegalArgumentException
   */
  public static void verifyUniqueList(String operationName, int[] axes) throws IllegalArgumentException {
    Set<Integer> seen = new HashSet<>();
    
    for (int axis : axes) {
      if (!seen.add(axis)) {
        throw new IllegalArgumentException(Message.crash("tensor indexing", "Duplicate axis", operationName, "order must be unique, but found two or more instances of " + axis));
      }
    }
  }

  /**
   * Throws an exception if the axes is empty.
   * 
   * @param axes axes to verify
   * @throws IllegalArgumentException
   */
  public static void verifyNotEmpty(String operationName, int[] axes) throws IllegalArgumentException {
    if (axes.length == 0) {
      throw new IllegalArgumentException(Message.crash("tensor indexing", "Empty axes", operationName, "axes cannot be empty"));
    }
  }

}
