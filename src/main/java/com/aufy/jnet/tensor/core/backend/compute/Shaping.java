package com.aufy.jnet.tensor.core.backend.compute;

import java.util.ArrayList;

import com.aufy.jnet.stats.primitive.Statistics;
import com.aufy.jnet.tensor.core.backend.util.ArrayTools;

public class Shaping {
  /*
  operates on the shape metadata. actual computation goes to engine.
  */
  
  public static int[] calculateResShape(int[] shapeA, int[] shapeB, int[] axesA, int[] axesB) {
    int[] survivorsA = getSurvivors(shapeA, axesA);
    int[] survivorsB = getSurvivors(shapeB, axesB);
    
    int[] resShape = new int[survivorsA.length + survivorsB.length];

    System.arraycopy(survivorsA, 0, resShape, 0, survivorsA.length);
    System.arraycopy(survivorsB, 0, resShape, survivorsA.length, survivorsB.length);
    
    return resShape;
  }

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

  public static int[] getSurvivors(int[] shape, int[] axes) {
    int[] reduced = shape.clone();
    for (int axis : axes) {
      reduced[axis] = 1;
    }
    return reduced;
  }

  public static int[] unravel(int index, int[] shape) {
    int[] coords = new int[shape.length];
    for (int i = shape.length - 1; i >= 0; i--) {
      coords[i] = index % shape[i];
      index /= shape[i];
    }
    return coords;
  }

  public static int[] getSubShape(int[] fullShape, int[] axes) {
    int[] subShape = new int[axes.length];
    for (int i = 0; i < axes.length; i++) subShape[i] = fullShape[axes[i]];
    return subShape;
  }

  public static int[] broadcastedShape(int[] a, int[] b) {
    int rank = Math.max(a.length, b.length);
    int[] result = new int[rank];

    for (int i = 0; i < rank; i++) {
      int dimA = (i >= rank - a.length) ? a[i - (rank - a.length)] : 1;
      int dimB = (i >= rank - b.length) ? b[i - (rank - b.length)] : 1;

      result[i] = Math.max(dimA, dimB);
    }
    return result;
  }
  
  public static int[] squeezeShape(int[] shape) {
    ArrayList<Integer> newShape = new ArrayList<>();
    
    for (int axis : shape) {
      if (axis != 1) {
        newShape.add(axis);
      }
    }

    if (newShape.isEmpty()) newShape.add(1);
    
    return newShape.stream().mapToInt(Integer::intValue).toArray();
  }
  
  public static int[] unsqueezeShape(int[] shape, int[] axes) {
    int newRank = shape.length + axes.length;
    int[] result = new int[newRank];

    int shapePtr = 0;
    for (int i = 0; i < newRank; i++) result[i] = (ArrayTools.contains(axes, i)) ? 1 : shape[shapePtr++];
    
    return result;
  }

  public static int[] findOffset(int[] subShape, int[] axes, int[] strides) {
    int volume = Statistics.prod(subShape);
    int[] offsets = new int[volume];
    for (int k = 0; k < volume; k++) {
      int[] kCoords = Shaping.unravel(k, subShape);
      int offset = 0;
      for (int i = 0; i < axes.length; i++) {
        offset += kCoords[i] * strides[axes[i]];
      }
      offsets[k] = offset;
    }
    return offsets;
  }

  public static int[] inferShape(int[] oldShape, int[] newShape) {
    int[] workingShape = newShape.clone();

    for (int i = 0; i < workingShape.length; i++) {
      if (workingShape[i] == -1) {
        workingShape[i] = Statistics.prod(oldShape) / (-1 * Statistics.prod(newShape));
      }
    }

    return workingShape;
  }
}
