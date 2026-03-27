package com.aufy.jnet.tensor.core.exception;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import com.aufy.jnet.tensor.core.backend.util.ArrayTools;

public class Shape {
  private static String namingMessage(String operationName) {
    return (operationName == null || operationName.isBlank()) ? "" : " for " + operationName;
  }

  public static void verifyNotEmpty(String operationName, int... shape) {
    if (shape == null || shape.length == 0) {
      throw new IllegalArgumentException("Illegal shape" + namingMessage(operationName) + ": shape cannot be empty.");
    }
  }

  public static void verifyRank(String operationName, int expected, int actual) {
    if (expected != actual) {
      throw new IllegalArgumentException("Mismatching rank" + namingMessage(operationName) + ": expected rank " + expected + " but got " + actual);
    }
  }

  public static void verifyInference(String operationName, int... shape) {
    int countNegatives = 0;
    for (int dim : shape) {
      if (dim < -1) throw new IllegalArgumentException("Illegal dimension" + namingMessage(operationName) + ": " + dim);
      if (dim == -1) countNegatives++;
    }
    if (countNegatives > 1) {
      throw new IllegalArgumentException("Ambiguous inference" + namingMessage(operationName) + ": only one inference allowed but got " + countNegatives + " dimensions to infer");
    }
  }

  public static void verifySizeMatch(String operationName, int[] shapeA, int[] shapeB) {
    int sizeA = ArrayTools.prod(shapeA);
    int sizeB = ArrayTools.prod(shapeB);
    if (sizeA != sizeB) {
      throw new IllegalArgumentException("Mismatching sizes" + namingMessage(operationName) + ": cannot reshape " + Arrays.toString(shapeA) + "(size "+ sizeA + ") into shape " + Arrays.toString(shapeB) + "(size " + sizeB + ")");
    }
  }

  public static void verifyPermutation(int[] originalShape, int[] axes) {
    verifyRank("permutation", originalShape.length, axes.length);

    Set<Integer> seen = new HashSet<>();
    for (int axis : axes) {
      verifyAxis(axis, originalShape.length);

      if (!seen.add(axis)) {
        throw new IllegalArgumentException("Duplicate axis in permutation: permutation order must be unique, but found two or more instances of axis " + axis);
      }
    }
  }

  public static void verifyBroadcast(int[] shapeA, int[] shapeB) {
    int lenA = shapeA.length;
    int lenB = shapeB.length;
    int maxLen = Math.max(lenA, lenB);

    for (int i = 1; i <= maxLen; i++) {
      int dimA = (lenA - i >= 0) ? shapeA[lenA - i] : 1;
      int dimB = (lenB - i >= 0) ? shapeB[lenB - i] : 1;

      if (dimA != dimB && dimA != 1 && dimB != 1) {
        throw new IllegalArgumentException("Incompatible shapes for broadcasting: " + Arrays.toString(shapeA) + " and " + Arrays.toString(shapeB));
      }
    }
  }

  public static void verifyAxis(int rank, int axis) {
    if (axis < 0 || axis >= rank) {
      throw new IndexOutOfBoundsException("Axis out of bounds: " + axis + " is out of bounds for a tensor of rank " + rank);
    }
  }

  public static void verifyConcat(int axis, int[]... shapes) {
    int rankRef = shapes[0].length;
    verifyAxis(rankRef, axis);

    for (int i = 1; i < shapes.length; i++) {
      if (shapes[i].length != rankRef) {
        throw new IllegalArgumentException("Mismatching rank at index " + i + " for concatenation.");
      }
      for (int d = 0; d < rankRef; d++) {
        if (d != axis && shapes[i][d] != shapes[0][d]) {
          throw new IllegalArgumentException("Dimension mismatch at axis " + d + ": " + shapes[0][d] + " vs " + shapes[i][d]);
        }
      }
    }
  }

  public static void verifyStack(int[]... shapes) {
    for (int i = 1; i < shapes.length; i++) {
      if (!Arrays.equals(shapes[0], shapes[i])) {
        throw new IllegalArgumentException("Stack requires identical shapes. Mismatch at index " + i + ": " + Arrays.toString(shapes[0]) + " vs " + Arrays.toString(shapes[i]));
      }
    }
  }
}

