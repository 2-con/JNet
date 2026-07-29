package com.aufy.jnet.core.backend.exceptions.tensors;

import java.util.Arrays;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Methods for checking the geometry of tensors.
 */
public class Geometry {

  /**
   * Checks if a shape is neither null nor empty.
   *
   * @param operationName the name of the calling operation.
   * @param shape the shape dimensions to check.
   * @throws IllegalArgumentException if the shape is null or empty.
   */
  public static void verifyNotEmpty(String operationName, int... shape) throws IllegalArgumentException {
    if (shape == null || shape.length == 0) {
      throw new IllegalArgumentException( Message.crash("tensor shaping", "Illegal shape", operationName, "shapes cannot be empty or null"));
    }
  }

  /**
   * Validates that the rank matches the expected rank.
   *
   * @param operationName the name of the calling operation.
   * @param expected the required rank.
   * @param actual the received rank.
   * @throws IllegalArgumentException if the ranks do not match.
   */
  public static void verifyRank(String operationName, int expected, int actual) throws IllegalArgumentException {
    if (expected != actual) {
      throw new IllegalArgumentException(Message.crash("tensor shaping", "Mismatching rank", operationName, "expected rank " + expected + " but got " + actual));
    }
  }

  /**
   * Validates that the rank matches the minimum expected rank with a message.
   *
   * @param operationName the name of the calling operation.
   * @param expected the required minimum rank.
   * @param actual the received rank.
   * @throws IllegalArgumentException if the ranks do not match.
   */
  public static void verifyMinimumRank(String operationName, int expected, int actual) throws IllegalArgumentException {
    if (expected > actual) {
      throw new IllegalArgumentException(Message.crash("tensor shaping", "Rank too big", operationName, "minimum rank must be " + expected + " but got " + actual));
    }
  }

  /**
   * Validates that the total data size matches the volume of the target shape.
   *
   * @param operationName the name of the calling operation.
   * @param size the total number of data elements.
   * @param shape the target shape dimensions.
   * @throws IllegalArgumentException if the data size does not match the shape volume.
   */
  public static void verifyDataShape(String operationName, int size, int[] shape) throws IllegalArgumentException {
    if (size != Reductions.prod(shape)) {
      throw new IllegalArgumentException(Message.crash("tensor shaping", "Mismatching data size", operationName, "tensor of volume " + size + " cannot be packaged into " + Arrays.toString(shape) + " (size " + Reductions.prod(shape) + ")"));
    }
  }

  /**
   * Validates that two shapes resolve to the same total number of elements.
   *
   * @param operationName the name of the calling operation.
   * @param shapeA the first shape array.
   * @param shapeB the second shape array.
   * @throws IllegalArgumentException if the shapes have different total volumes.
   */
  public static void verifyEqualSize(String operationName, int[] shapeA, int[] shapeB) throws IllegalArgumentException {
    int sizeA = Reductions.prod(shapeA);
    int sizeB = Reductions.prod(shapeB);
    
    if (sizeA != sizeB) {
      throw new IllegalArgumentException(Message.crash("tensor shaping", "Mismatching sizes", operationName, "tensor of shape " + Arrays.toString(shapeA) + "(size " + sizeA + ") is not equal to  " + Arrays.toString(shapeB) + "(size " + sizeB + ")"));
    }
  }

}

