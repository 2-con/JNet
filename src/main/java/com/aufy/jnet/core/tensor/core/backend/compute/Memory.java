package com.aufy.jnet.core.tensor.core.backend.compute;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;

/**
 * Main class for handling memory and index logic. This is different from shaping (which also handle memory and stride logic) in the sense
 * that it handles operation that dierectly work with tensor memory. No safeguards are present as they have been removed to the tensor class.
 *
 */
public class Memory {
  /*
  for operations focused on the index logic itself. shapes and higher order descriptions are handled under shaping.
  for now just keep these as tools shaping or engine uses
  */
  
  /**
   * Calculates the raw memory blocks to skip to move one slot in a dim.
   *
   * @param shape The shape of the tensor.
   * @return The strides array.
   */
  public static int[] calculateStrides(int[] shape) {
    int[] strides = new int[shape.length];
    int st = 1;
    for (int i = shape.length - 1; i >= 0; i--) { // starting from the last, go backwards
      strides[i] = st;
      st *= shape[i]; // the next dimension will be placed after the length of the axis since that is when the tensor would be stacked
    }
    return strides;
  }

  /**
   * Calculates the index based on the strides and indices.
   *
   * @param strides The array of strides.
   * @param indices The variable number of indices.
   * @return The calculated index.
   */
  public static int getIndex(int[] strides, int... indices) {
    return Reductions.dot(strides, indices);
  }

  /**
   * Converts a raw index in the 1D heap into a proper coordinate in the tensor. It does not have any safety so
   * make sure the index is within bounds (product of shape).
   * 
   * @param index Index in the 1D heap.
   * @param shape Shape of the tensor.
   * @return Coordinates in the tensor.
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
   * Calculates the raw memory address based on the strides and the coordinates in the tensor.
   *
   * @param strides The array of strides.
   * @param coords The specific location in a tensor.
   * @return The calculated raw index.
   */
  public static int getOffset(int[] strides, int[] coords) {
    int offset = 0;
    for (int i = 0; i < coords.length; i++) {
      offset += coords[i] * strides[i];
    }
    return offset;
  }

  /**
   * It starts at the rightmost dimension and adds 1 until it overflows and resets that dimension to 0, carrying the addition over to the left dimension. 
   * It returns true as long is more to visit and false when the is no more to visit.
   *
   * @param coords The location of a spot.
   * @param shape The shape of the tensor.
   * @return Whether there is still more to visit.
   */
  public static boolean nextCoordinate(int[] coords, int[] shape) {
    for (int i = shape.length - 1; i >= 0; i--) {
      if (++coords[i] < shape[i]) return true;
      coords[i] = 0;
    }
    return false;
  }

  /**
   * Calculates the raw memory address based on the strides and the coordinates in the tensor after reducing it along axes and coordinates in the
   * extracted subspace. Its essentially converting indices from the full tensor to the reduced tensor based on specifications axes.
   * 
   * @param resCoords The coordinates in the reduced tensor.
   * @param kCoords The coordinates in an extracted subspace of the original tensor.
   * @param axes The specific dimension indices that are being collapsed or calculated over.
   * @param fullShape The complete, original shape of the input tensor before any reduction happened.
   * @param strides The memory strides of that original shape
   * @param isA A positional flag. For multi-tensor operations (like C = A @ B), the surviving coordinates might line up with the front or the back of the shape. isA controls whether to read resCoords from the beginning (0) or offset from the end
   * @return The resulting raw memory address in this new tensor.
   */
  public static int mapToOffset(int[] resCoords, int[] kCoords, int[] axes, int[] fullShape, int[] strides, boolean isA) {
    int[] fullCoords = new int[fullShape.length];
    
    for (int i = 0; i < axes.length; i++) { // puts the kcoord (wherever it may be) in the location relative to the original shape
      fullCoords[axes[i]] = kCoords[i];
    }

    int resPtr = isA ? 0 : (resCoords.length - (fullShape.length - axes.length));

    for (int i = 0; i < fullShape.length; i++) { // fills the remaining coordinates with resCoords. if theres already a spot filled then it moves
      if (!Tools.contains(axes, i)) { // only put stuff in if its not in axes of reduction
        fullCoords[i] = resCoords[resPtr++];
      }
    }
    
    return getOffset(strides, fullCoords); // find the location of the strides if it is on the full shape
  }

}
