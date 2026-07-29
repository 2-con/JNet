package com.aufy.jnet.core.tensor.core.backend.compute;

import java.util.ArrayList;

import com.aufy.jnet.core.backend.arrayops.Reductions;
import com.aufy.jnet.core.backend.arrayops.Tools;

/**
 * Main class for handling shaping or geometric processing. This is different from memory because this class handles with parts indirectly related with
 * the memory
 */
public class Shaping {
  /*
  operates on the shape metadata. actual computation goes to engine.
  */
  
  /**
   * Calculates the shape of the result tensor after contraction along axes.
   * 
   * @param shapeA Shape of tensor A.
   * @param shapeB Shape of tensor B.
   * @param axesA Axes of contraction in tensor A.
   * @param axesB Corresponding axes of contraction in tensor B.
   * @return The shape of the result tensor after contraction.
   */
  public static int[] calculateResShape(int[] shapeA, int[] shapeB, int[] axesA, int[] axesB) {
    // contracted axes get removed
    int[] survivorsA = removeAxes(shapeA, axesA);
    int[] survivorsB = removeAxes(shapeB, axesB);
    
    int[] resShape = new int[survivorsA.length + survivorsB.length];

    // copy the survivors over to reshape
    System.arraycopy(survivorsA, 0, resShape, 0       , survivorsA.length);
    System.arraycopy(survivorsB, 0, resShape, survivorsA.length, survivorsB.length);
    
    return resShape;
  }

  /**
   * Returns the shape of a tensor with selected axes removed.
   * 
   * @param shape Shape of the tensor.
   * @param axes Axes to remove.
   * @return The shape of the tensor with selected axes removed.
   */
  public static int[] removeAxes(int[] shape, int[] axes) {
    return Tools.remove(shape, axes);
  }

  /**
   * Returns the shape of a tensor with selected axes turned to a singleton.
   * 
   * @param shape Shape of the tensor.
   * @param axes Axes to turn to singletons.
   * @return The shape of the tensor with selected axes turned to singletons.
   */
  public static int[] reduceAxes(int[] shape, int[] axes) {
    int[] reduced = shape.clone();
    for (int axis : axes) {
      reduced[axis] = 1;
    }
    return reduced;
  }

  /**
   * Extracts a subset of dimensions from a full shape array based on specified axes. It basically re-arranges a shape array in an order. No safety present so duplicates and missing dims are ok and not restricted.
   *
   * @param shape The shape of the tensor.
   * @param order The new order of the shape.
   * @return A new shape array containing only the sizes of the selected axes.
   */
  public static int[] reorder(int[] shape, int[] order) {
    int[] out = new int[order.length];
    for (int i = 0; i < order.length; i++) out[i] = shape[order[i]];
    return out;
  }

  /**
   * Computes the resulting shape when broadcasting two tensor shapes together.
   *
   * @param shapeA The shape array of the first tensor.
   * @param shapeB The shape array of the second tensor.
   * @return The broadcasted shape array.
   */
  public static int[] broadcastedShape(int[] shapeA, int[] shapeB) {
    int rank = Math.max(shapeA.length, shapeB.length);
    int[] result = new int[rank];

    for (int i = 0; i < rank; i++) {
      int dimA = (i >= rank - shapeA.length) ? shapeA[i - (rank - shapeA.length)] : 1;
      int dimB = (i >= rank - shapeB.length) ? shapeB[i - (rank - shapeB.length)] : 1;

      result[i] = Math.max(dimA, dimB);
    }
    return result;
  }
  
  /**
   * Removes all singleton dimensions (dimensions of size 1) from the tensor shape. If all dimensions are 1, returns a shape of [1].
   *
   * @param shape The original shape array.
   * @return The squeezed shape array with size-1 dimensions removed.
   */
  public static int[] squeeze(int[] shape) {
    ArrayList<Integer> newShape = new ArrayList<>();
    
    for (int axis : shape) {
      if (axis != 1) {
        newShape.add(axis);
      }
    }

    if (newShape.isEmpty()) newShape.add(1);
    
    return newShape.stream().mapToInt(Integer::intValue).toArray();
  }
  
  /**
   * Inserts singleton dimensions (dimensions of size 1) into a shape at the specified axis indices.
   *
   * @param shape The original shape array.
   * @param axes The target indices where singleton dimensions should be inserted.
   * @return The unsqueezed shape array containing the new singleton dimensions.
   */
  public static int[] unsqueeze(int[] shape, int[] axes) {
    int newRank = shape.length + axes.length;
    int[] result = new int[newRank];

    int shapePtr = 0;
    for (int i = 0; i < newRank; i++) result[i] = (Tools.contains(axes, i)) ? 1 : shape[shapePtr++];
    
    return result;
  }

  /**
   * Calculates the memory offsets for elements in a subshape view based on target axes and strides.
   * 
   * @param subShape The dimensions of the subview.
   * @param axes The underlying tensor axes that map to the sub-view dimensions.
   * @param strides The stride array of the underlying tensor.
   * @return An array containing the calculated flat memory offsets for every element in the sub-view.
   */
  public static int[] findOffset(int[] subShape, int[] axes, int[] strides) {
    int volume = Reductions.prod(subShape);
    int[] offsets = new int[volume];
    for (int k = 0; k < volume; k++) {
      int[] kCoords = Memory.unravel(k, subShape);
      int offset = 0;
      for (int i = 0; i < axes.length; i++) {
        offset += kCoords[i] * strides[axes[i]];
      }
      offsets[k] = offset;
    }
    return offsets;
  }

  /**
   * Infers the missing dimension size (denoted by -1) in a target reshape configuration based on the total volume of the original shape.
   *
   * @param oldShape The original shape of the tensor.
   * @param newShape The desired new shape, containing at most one placeholder dimension of -1.
   * @return A completed shape array where the -1 placeholder is replaced by the calculated dimension.
   */
  public static int[] inferShape(int[] oldShape, int[] newShape) {
    int[] workingShape = newShape.clone();

    for (int i = 0; i < workingShape.length; i++) {
      if (workingShape[i] == -1) {
        workingShape[i] = Reductions.prod(oldShape) / (-1 * Reductions.prod(newShape));
      }
    }

    return workingShape;
  }
  
}
