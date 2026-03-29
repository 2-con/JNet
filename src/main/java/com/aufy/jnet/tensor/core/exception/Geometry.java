package com.aufy.jnet.tensor.core.exception;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import com.aufy.jnet.tensor.core.backend.util.ArrayOps;

public class Geometry {
  /*
  make these the most readable since shaping is the primary way people mess up tensor ops
  */

  private static String namingMessage(String operationName) {
    return (operationName == null || operationName.isBlank()) ? "" : " for " + operationName;
  }

  // ==============================================================================================
  // GENERAL CASES
  // ==============================================================================================

  public static void verifyNotEmpty(String operationName, int... shape) throws IllegalArgumentException {
    if (shape == null || shape.length == 0) throw new IllegalArgumentException("Illegal shape" + namingMessage(operationName) + ": shape cannot be empty.");
  }

  public static void verifyRank(String operationName, int expected, int actual) throws IllegalArgumentException {
    if (expected != actual) throw new IllegalArgumentException("Mismatching rank" + namingMessage(operationName) + ": expected rank " + expected + " but got " + actual);
  }

  public static void verifyDataShape(String operationName, int size, int[] shape) throws IllegalArgumentException {
    if (size != ArrayOps.prod(shape)) throw new IllegalArgumentException("Mismatching data size" + namingMessage(operationName) + ": data of size "+ size + " cannot be packaged into " + Arrays.toString(shape) + " (size " + ArrayOps.prod(shape) + ")");
  }

  public static void verifyAxis(String operationName, int rank, int axis) throws IndexOutOfBoundsException {
    if (axis < 0 || axis >= rank) throw new IndexOutOfBoundsException("Axis out of bounds" + namingMessage(operationName) + ": " + axis + " is out of bounds for a tensor of rank " + rank);
  }

  public static void verifyInference(String operationName, int... shape) throws IllegalArgumentException {
    int countNegatives = 0;
    for (int dim : shape) {
      if (dim < -1) throw new IllegalArgumentException("Illegal dimension" + namingMessage(operationName) + ": " + dim);
      if (dim == -1) countNegatives++;
    }
    if (countNegatives > 1) {
      throw new IllegalArgumentException("Ambiguous inference" + namingMessage(operationName) + ": only one inference allowed but got " + countNegatives + " dimensions to infer");
    }
  }

  public static void verifySizeMatch(String operationName, int[] shapeA, int[] shapeB) throws IllegalArgumentException {
    int sizeA = ArrayOps.prod(shapeA);
    int sizeB = ArrayOps.prod(shapeB);
    if (sizeA != sizeB) {
      throw new IllegalArgumentException("Mismatching sizes" + namingMessage(operationName) + ": cannot reshape " + Arrays.toString(shapeA) + "(size "+ sizeA + ") into shape " + Arrays.toString(shapeB) + "(size " + sizeB + ")");
    }
  }

  public static void verifyUniqueList(String operationName, int[] axes) throws IllegalArgumentException {
    Set<Integer> seen = new HashSet<>();
    for (int axis : axes) {
      if (!seen.add(axis)) {
        throw new IllegalArgumentException("Duplicate axis" + namingMessage(operationName) + ": order must be unique, but found two or more instances of " + axis);
      }
    }
  }

  public static void verifyBroadcast(String operationName, int[] shapeA, int[] shapeB) throws IllegalArgumentException {
    int lenA = shapeA.length;
    int lenB = shapeB.length;
    int maxLen = Math.max(lenA, lenB);

    for (int i = 1; i <= maxLen; i++) {
      int dimA = (lenA - i >= 0) ? shapeA[lenA - i] : 1;
      int dimB = (lenB - i >= 0) ? shapeB[lenB - i] : 1;

      if (dimA != dimB && dimA != 1 && dimB != 1) {
        throw new IllegalArgumentException("Unable to broadcast shapes" + namingMessage(operationName) + ": broadcasting only works if either dimensions are 1 or match, but axis " + i + " has a dimension of " + dimA + " in tensor A (" + Arrays.toString(shapeA) + ") while the same axis corresponds to a dimension of " + dimB + " in tensor B (" + Arrays.toString(shapeB) + ")");
      }
    }
  }

  // ==============================================================================================
  // SPESIFIC CASES
  // ==============================================================================================
  
  public static void verifyMatMul(int[] shapeA, int[] shapeB) throws IllegalArgumentException{
    if (shapeA.length < 1 || shapeB.length < 1) throw new IllegalArgumentException("Incompatible rank: tensors must have a rank of at least 1");
    
    int indexA = shapeA[shapeA.length - 1];
    int indexB = (shapeB.length == 1) ? shapeB[0] : shapeB[shapeB.length - 2];
    
    if (indexA != indexB) {
      throw new IllegalArgumentException("Incompatible dimensions for matrix multiplication: " + indexA + " and " + indexB);
    }
  }

  public static void verifyConcat(int axis, int[]... shapes) throws IllegalArgumentException {
    int rankRef = shapes[0].length;
    verifyAxis("concatenation", rankRef, axis);

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

