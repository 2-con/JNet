package com.aufy.jnet.core.backend.exceptions.tensors;

import java.awt.IllegalComponentStateException;
import java.util.Arrays;

import com.aufy.jnet.core.backend.exceptions.Message;

/**
 * Utility class for validating structural and dimension requirements of tensor operations.
 */
public class Operation {
  /*
   * specific implementations go here; VIPs only
   */

  /**
   * Checks if two tensor shapes are compatible for matrix multiplication.
   *
   * @param shapeA the shape of the left tensor
   * @param shapeB the shape of the right tensor
   * @throws IllegalArgumentException if ranks are under 1 or inner dimensions mismatch
   */
  public static void verifyMatMul(int[] shapeA, int[] shapeB) throws IllegalArgumentException {
    if (shapeA.length < 1 || shapeB.length < 1) {
      throw new IllegalArgumentException(Message.crash("tensor operation", "Incompatible rank", "matmul", "tensors must have a rank of at least 1"));
    }

    //                     v           v
    // matmul  = A[..., i, j] * B[..., j, k] = C[..., i, k]
    int indexA = shapeA[shapeA.length - 1]; // last
    int indexB = (shapeB.length == 1) ? shapeB[0] : shapeB[shapeB.length - 2]; // 2nd dim if there is one, else remaining one

    if (indexA != indexB) {
      throw new IllegalArgumentException(Message.crash("tensor operation", "Incompatible dimensions for matrix multiplication", "matmul", "index " + indexA + " from tensor A and index " + indexB + " from tensor B"));
    }
  }

  /**
   * Checks if multiple shapes match exactly on all dimensions except the concatenation axis.
   *
   * @param axis the dimension index along which tensors are joined
   * @param shapes the collection of shapes to combine
   * @throws IllegalArgumentException if ranks vary or non-axis dimensions do not match
   */
  public static void verifyConcat(int axis, int[]... shapes) throws IllegalArgumentException {
    int rankRef = shapes[0].length;
    Axes.verifyAxis("concatenation", rankRef, axis);

    for (int i = 1; i < shapes.length; i++) {
      if (shapes[i].length != rankRef) {
        throw new IllegalArgumentException(
            Message.crash("tensor operation", "Mismatching rank at index " + i + " for concatenation", "concatenation", "expected " + rankRef + " but got " + shapes[i].length));
      }
      for (int d = 0; d < rankRef; d++) {
        if (d != axis && shapes[i][d] != shapes[0][d]) {
          throw new IllegalArgumentException(Message.crash("tensor operation", "Dimension mismatch at axis " + d, "concatenation", shapes[0][d] + " vs " + shapes[i][d]));
        }
      }
    }
  }

  /**
   * Checks if all provided shapes are completely identical for stacking.
   *
   * @param shapes the collection of shapes to stack
   * @throws IllegalArgumentException if any shape differs from the first one
   */
  public static void verifyStack(int[]... shapes) {
    for (int i = 1; i < shapes.length; i++) {
      if (!Arrays.equals(shapes[0], shapes[i])) {
        throw new IllegalArgumentException(Message.crash("tensor operation", "Stack requires identical shapes", "stack", "Mismatch at index " + i + ": " + Arrays.toString(shapes[0]) + " vs " + Arrays.toString(shapes[i])));
      }
    }
  }

  /**
   * Checks if two shapes are compatible for element-wise broadcasting rules.
   *
   * @param operationName the name of the calling operation
   * @param shapeA the shape of the first tensor
   * @param shapeB the shape of the second tensor
   * @throws IllegalArgumentException if dimensions cannot stretch to match each other
   */
  public static void verifyBroadcast(String operationName, int[] shapeA, int[] shapeB) throws IllegalArgumentException {
    int lenA = shapeA.length;
    int lenB = shapeB.length;
    int maxLen = Math.max(lenA, lenB);

    for (int i = 1; i <= maxLen; i++) {
      int dimA = (lenA - i >= 0) ? shapeA[lenA - i] : 1;
      int dimB = (lenB - i >= 0) ? shapeB[lenB - i] : 1;

      if (dimA != dimB && dimA != 1 && dimB != 1) {
        throw new IllegalArgumentException(Message.crash("tensor operation", "Unable to broadcast shapes", operationName, "broadcasting only works if either dimensions are 1 or match, but axis " + i + " has a dimension of " + dimA + " in tensor A (" + Arrays.toString(shapeA) + ") while the same axis corresponds to a dimension of " + dimB + " in tensor B (" + Arrays.toString(shapeB) + ")"));
      }
    }
  }

  /**
   * Checks if a reshaping array contains at most one inferred (-1) dimension.
   *
   * @param operationName the name of the calling operation
   * @param shape the target shape layout containing dimensions
   * @throws IllegalArgumentException if any dimension is less than -1, or if multiple dimensions are -1
   */
  public static void verifyInference(String operationName, int... shape) throws IllegalArgumentException {
    int countNegatives = 0;
    for (int dim : shape) {
      if (dim < -1) {
        throw new IllegalArgumentException( Message.crash("tensor operation", "Illegal dimension", operationName, String.valueOf(dim)));
      }
      if (dim == -1)
        countNegatives++;
    }

    if (countNegatives > 1) {
      throw new IllegalArgumentException(Message.crash("tensor operation", "Ambiguous inference", operationName, "only one inference allowed but got " + countNegatives + " dimensions to infer"));
    }
  }

  
  /**
   * Checks if all elements in a double array are above a limit.
   *
   * @param cutoff the cutoff to compare if data is above or not.
   * @param data the array to check.
   * @throws IllegalArgumentException if any element is below the cutoff.
   */
  public static void aboveValue(double cutoff, double[] data, String operation) throws IllegalArgumentException {
    for (int i = 0; i < data.length; i++) {
      if (data[i] < cutoff) {
        throw new IllegalArgumentException(Message.crash("tensor argument", "Bad data", operation , "tensor data must be at least " + cutoff + " but one instance got " + data[i]));
      }
    }
  }

  /**
   * Checks if all elements in a double array are below a limit.
   *
   * @param cutoff the cutoff to compare if data is below or not.
   * @param data the array to check.
   * @throws IllegalArgumentException if any element exceeds the cutoff.
   */
  public static void belowValue(double cutoff, double[] data, String operation) throws IllegalArgumentException {
    for (int i = 0; i < data.length; i++) {
      if (data[i] > cutoff) {
        throw new IllegalArgumentException(Message.crash("tensor argument", "Bad data", operation , "tensor data cannot exceed " + cutoff + " but one instance got " + data[i]));
      }
    }
  }

  /**
   * Checks if all elements in a double array are between two values.
   *
   * @param lower the lower inclusive bound.
   * @param upper the upper inclusive bound.
   * @param data the array to check.
   * @throws IllegalArgumentException if any element is out of bounds.
   */
  public static void betweenValue(double lower, double upper, double[] data, String operation) throws IllegalArgumentException {
    for (int i = 0; i < data.length; i++) {
      if (data[i] > upper || data[i] < lower) {
        throw new IllegalArgumentException(Message.crash("tensor argument", "Bad data", operation , "tensor data must be within " + lower + " and " + upper + " but one instance got " + data[i]));
      }
    }
  }

  /**
   * Check if a data contains any invalid values (NaN or Infinity) for gradients.
   * @param data the data to check
   * @throws IllegalComponentStateException if the data contains any invalid values
   */
  public static void hasGrad(Object data) throws IllegalComponentStateException {
    if (data == null) throw new IllegalComponentStateException(Message.crash("internal", "Missing data", null, "tensor gradient is null; try calling backward()"));
  }

}
