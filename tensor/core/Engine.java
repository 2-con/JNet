package tensor.core;

public class Engine {
  /**
   * Calculates the strides of a tensor given its shape.
   * 
   * The strides is an array of length equal to the rank of the tensor, where each element is the number of elements in memory between two consecutive elements in the specified axis.
   * 
   * @param shape the shape of the tensor
   * @return the strides of the tensor
   */
  public static int[] calculateStrides(int[] shape) {
    int[] strides = new int[shape.length];
    int st = 1;
    for (int i = shape.length - 1; i >= 0; i--) {
      strides[i] = st;
      st *= shape[i];
    }
    return strides;
  }

  /**
   * Pretty prints a tensor given its flat representation, shape, strides, rank, offset, and width.
   * The tensor is represented as a nested array, with each rank of the tensor represented by a pair of square brackets.
   * Each element of the tensor is represented by a float formatted to the specified width.
   * If the tensor is of rank greater than 1, then the elements of each sub-array are indented by a single space for each rank greater than 1.
   * 
   * @param flat the flat representation of the tensor
   * @param shape the shape of the tensor
   * @param strides the strides of the tensor
   * @param rank the rank of the tensor
   * @param offset the offset of the tensor
   * @param width the width of each element in the pretty print
   * @return a string representation of the tensor
   */
  public static String prettyPrint(double[] flat, int[] shape, int[] strides, int rank, int offset, int width) {
    StringBuilder sb = new StringBuilder("[");

    for (int i = 0; i < shape[rank]; i++) {
      int currentOffset = offset + (i * strides[rank]);

      if (rank == shape.length - 1) {
        sb.append(String.format("%" + width + ".4f", flat[currentOffset]));
      } else {
        sb.append(prettyPrint(flat, shape, strides, rank + 1, currentOffset, width));
      }

      if (i < shape[rank] - 1) {
        if (rank == shape.length - 1) {
          sb.append(", ");
        } else {
          int newlineCount = Math.max(1, shape.length - rank - 1);
          sb.append(",").append("\n".repeat(newlineCount));
          sb.append(" ".repeat(rank + 1)); 
        }
      }
    }
    sb.append("]");
    return sb.toString();
  }

  /**
   * Returns the index of the tensor element at the given coordinates.
   * The index is calculated by multiplying each coordinate by the corresponding stride and summing the results.
   * 
   * @param strides the strides of the tensor
   * @param indices the coordinates of the element
   * @return the index of the element at the given coordinates
   */
  public static int getIndex(int[] strides, int... indices) {
    int index = 0;
    for (int i = 0; i < indices.length; i++) {
      index += indices[i] * strides[i];
    }
    return index;
  }

  /**
   * Reverses the given array.
   * 
   * @param arr the array to reverse
   * @return a new array with the elements of the given array in reverse order
   */
  public static int[] reverse(int[] arr) {
    int[] res = new int[arr.length];
    for (int i = 0; i < arr.length; i++) {
        res[i] = arr[arr.length - 1 - i];
    }
    return res;
  }

  /**
   * Returns the offset of the tensor element at the given coordinates.
   * The offset is calculated by multiplying each coordinate by the corresponding stride and summing the results.
   * 
   * @param strides the strides of the tensor
   * @param coords the coordinates of the element
   * @return the offset of the element at the given coordinates
   */
  public static int getOffset(int[] strides, int[] coords) {
    int offset = 0;
    for (int i = 0; i < coords.length; i++) {
      offset += coords[i] * strides[i];
    }
    return offset;
  }

  /**
   * Calculates the shape of the tensor resulting from the contraction of two tensors.
   * The contraction is performed by summing the elements of the two tensors along the given axes.
   * The resulting shape is the concatenation of the two arrays of surviving axes.
   * 
   * @param shapeA the shape of the first tensor
   * @param shapeB the shape of the second tensor
   * @param axesA the axes of the first tensor to contract along
   * @param axesB the axes of the second tensor to contract along
   * @return the shape of the resulting tensor
   */
  public static int[] calculateResShape(int[] shapeA, int[] shapeB, int[] axesA, int[] axesB) {
    int[] survivorsA = getSurvivors(shapeA, axesA);
    int[] survivorsB = getSurvivors(shapeB, axesB);
    
    int[] resShape = new int[survivorsA.length + survivorsB.length];

    System.arraycopy(survivorsA, 0, resShape, 0, survivorsA.length);
    System.arraycopy(survivorsB, 0, resShape, survivorsA.length, survivorsB.length);
    
    return resShape;
  }

  /**
   * Returns the axes of the resulting tensor after a contraction operation.
   * The axes of the resulting tensor are the concatenation of the surviving axes of the two input tensors.
   * 
   * @param shapeA the shape of the first tensor
   * @param shapeB the shape of the second tensor
   * @param axesA the axes of the first tensor to contract along
   * @param axesB the axes of the second tensor to contract along
   * @return an array of two arrays, where the first array contains the surviving axes of the first tensor and the second array contains the surviving axes of the second tensor
   */
  public static int[][] getResultAxes(int[] shapeA, int[] shapeB, int[] axesA, int[] axesB) {
    int[] survivorsA = getSurvivors(shapeA, axesA);
    int[] survivorsB = getSurvivors(shapeB, axesB);

    int[] axesFromA = new int[survivorsA.length];
    int[] axesFromB = new int[survivorsB.length];

    for (int i = 0; i < axesFromA.length; i++) {
      axesFromA[i] = i;
    }

    for (int i = 0; i < axesFromB.length; i++) {
      axesFromB[i] = survivorsA.length + i;
    }

    return new int[][] {axesFromA, axesFromB};
  }

  /**
   * Broadcasts the gradient of a tensor to a larger shape.
   * This is used when computing the gradient of a contraction operation.
   * The gradient is broadcasted to the larger shape by repeating the values along the dimensions that were reduced.
   * 
   * @param gradOutput the gradient of the resulting tensor
   * @param gradShape the shape of the resulting tensor
   * @param inputShape the shape of the original tensor
   * @param axes the axes that were reduced in the contraction operation
   * @return a new array containing the broadcasted gradient
   */
  public static double[] broadcastGrad(double[] gradOutput, int[] gradShape, int[] inputShape, int[] axes) {
    double[] result = new double[Engine.sizeOf(inputShape)];

    int[] inputStrides = Engine.calculateStrides(inputShape);
    int[] gradStrides  = Engine.calculateStrides(gradShape);

    for (int i = 0; i < result.length; i++) {
      int remaining = i;
      int gradIndex = 0;

      for (int dim = 0, g = 0; dim < inputShape.length; dim++) {
        int coord = remaining / inputStrides[dim];
        remaining %= inputStrides[dim];

        // if this dimension was reduced, skip it
        boolean reduced = false;
        for (int ax : axes) {
          if (ax == dim) {
            reduced = true;
            break;
          }
        }

        if (!reduced) {
          gradIndex += coord * gradStrides[g];
          g++;
        }
      }

      result[i] = gradOutput[gradIndex];
    }

    return result;
  }

  /**
   * Returns an array of dimensions that are not contracted along the given axes.
   * The surviving dimensions are the dimensions of the tensor that are not reduced in a contraction operation.
   * 
   * @param shape the shape of the tensor
   * @param axes the axes of the tensor to contract along
   * @return an array of the surviving dimensions of the tensor
   */
  public static int[] getSurvivors(int[] shape, int[] axes) {
    int[] survivors = new int[shape.length - axes.length];

    int ptr = 0;
    for (int i = 0; i < shape.length; i++) {
      if (!contains(axes, i)) {
        survivors[ptr++] = shape[i];
      }
    }

    return survivors;
  }

  /**
   * Contracts two tensors along the specified axes.
   * 
   * @param dataA the data of the first tensor
   * @param stridesA the strides of the first tensor
   * @param dataB the data of the second tensor
   * @param stridesB the strides of the second tensor
   * @param shapeA the shape of the first tensor
   * @param axesA the axes of the first tensor to contract along
   * @param shapeB the shape of the second tensor
   * @param axesB the axes of the second tensor to contract along
   * @param resShape the shape of the resulting tensor
   * @return a new array containing the result of contracting the two input tensors along the specified axes
   */
  public static double[] contract(double[] dataA, int[] stridesA, double[] dataB, int[] stridesB, int[] shapeA, int[] axesA, int[] shapeB, int[] axesB, int[] resShape) {
    double[] resData = new double[sizeOf(resShape)];
    int[] resCoords = new int[resShape.length];

    int[] subShapeA = getSubShape(shapeA, axesA);
    int contractVolume = sizeOf(subShapeA);

    int resIdx = 0;
    do {
      double sum = 0;
      for (int k = 0; k < contractVolume; k++) {
        int[] kCoords = unravel(k, subShapeA);
        
        int offsetA = mapToOffset(resCoords, kCoords, axesA, shapeA, stridesA, true);
        int offsetB = mapToOffset(resCoords, kCoords, axesB, shapeB, stridesB, false);
        
        sum += dataA[offsetA] * dataB[offsetB];
      }
      resData[resIdx++] = sum;
    } while (nextCoordinate(resCoords, resShape));

    return resData;
  }

  /**
   * Maps the given coordinates to an offset in the flattened tensor.
   * The given coordinates are split into two parts: the coordinates of the contraction axes and the coordinates of the remaining axes.
   * The coordinates of the contraction axes are set in the full coordinates array at the corresponding contraction axes.
   * The coordinates of the remaining axes are copied from the result coordinates array, starting at the given result pointer.
   * The result pointer is incremented by the number of remaining axes after the copying is done.
   * Finally, the full coordinates array is used to calculate the offset in the flattened tensor.
   * 
   * @param resCoords the coordinates of the remaining axes
   * @param kCoords the coordinates of the contraction axes
   * @param axes the contraction axes
   * @param fullShape the shape of the flattened tensor
   * @param strides the strides of the flattened tensor
   * @param isA whether the result coordinates array should be used from the beginning or from the end
   * @return the offset in the flattened tensor corresponding to the given coordinates
   */
  public static int mapToOffset(int[] resCoords, int[] kCoords, int[] axes, int[] fullShape, int[] strides, boolean isA) {
    int[] fullCoords = new int[fullShape.length];
    
    for (int i = 0; i < axes.length; i++) {
      fullCoords[axes[i]] = kCoords[i];
    }

    int resPtr = isA ? 0 : (resCoords.length - (fullShape.length - axes.length));
    for (int i = 0; i < fullShape.length; i++) {
      if (!contains(axes, i)) {
        fullCoords[i] = resCoords[resPtr++];
      }
    }
    
    return getOffset(strides, fullCoords);
  }

  /**
   * Unravels a given index into a set of coordinates based on the given shape.
   * The coordinates are calculated by repeatedly dividing the index by the size of the corresponding dimension and taking the remainder as the coordinate.
   * 
   * @param index the index to unravel
   * @param shape the shape of the tensor to unravel
   * @return the unraveled coordinates
   */
  public static int[] unravel(int index, int[] shape) {
    int[] coords = new int[shape.length];
    for (int i = shape.length - 1; i >= 0; i--) {
      coords[i] = index % shape[i];
      index /= shape[i];
    }
    return coords;
  }

  /**
   * Advances the given coordinates by one in the given shape.
   * The coordinates are incremented from right to left (i.e. the last coordinate is incremented first).
   * If the last coordinate reaches the end of its dimension, it wraps around to zero and the next coordinate to the left is incremented.
   * If the first coordinate reaches the end of its dimension, the function returns false.
   * 
   * @param coords the coordinates to advance
   * @param shape the shape of the tensor
   * @return true if the coordinates were advanced, false if the first coordinate reached the end of its dimension
   */
  public static boolean nextCoordinate(int[] coords, int[] shape) {
    for (int i = shape.length - 1; i >= 0; i--) {
      if (++coords[i] < shape[i]) return true;
      coords[i] = 0;
    }
    return false;
  }

  /**
   * Returns the total number of elements in a tensor of the given shape.
   * 
   * @param shape the shape of the tensor
   * @return the total number of elements in the tensor
   */
  public static int sizeOf(int[] shape) {
    int vol = 1;
    for (int d : shape) vol *= d;
    return vol;
  }

  /**
   * Returns a new array containing the dimensions of the given shape at the given axes.
   * 
   * @param fullShape the full shape of the tensor
   * @param axes the axes of the tensor to get the dimensions from
   * @return a new array containing the dimensions of the given shape at the given axes
   */
  public static int[] getSubShape(int[] fullShape, int[] axes) {
    int[] subShape = new int[axes.length];
    for (int i = 0; i < axes.length; i++) subShape[i] = fullShape[axes[i]];
    return subShape;
  }

  /**
   * Returns true if the given array contains the given value, false otherwise.
   * 
   * @param arr the array to search in
   * @param val the value to search for
   * @return true if the given array contains the given value, false otherwise
   */
  public static boolean contains(int[] arr, int val) {
    for (int i : arr) if (i == val) return true;
    return false;
  }
}
