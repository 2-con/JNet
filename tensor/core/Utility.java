package tensor.core;

import java.util.ArrayList;

public class Utility {
  public static int[] calculateStrides(int[] shape) {
    int[] strides = new int[shape.length];
    int st = 1;
    for (int i = shape.length - 1; i >= 0; i--) {
      strides[i] = st;
      st *= shape[i];
    }
    return strides;
  }

  public static int getIndex(int[] strides, int... indices) {
    int index = 0;
    for (int i = 0; i < indices.length; i++) {
      index += indices[i] * strides[i];
    }
    return index;
  }

  public static int[] reverse(int[] arr) {
    int[] res = new int[arr.length];
    for (int i = 0; i < arr.length; i++) {
      res[i] = arr[arr.length - 1 - i];
    }
    return res;
  }

  public static int getOffset(int[] strides, int[] coords) {
    int offset = 0;
    for (int i = 0; i < coords.length; i++) {
      offset += coords[i] * strides[i];
    }
    return offset;
  }

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

  public static int[] getSurvivors(int[] shape, int[] axes, boolean keepDims) {
    if (keepDims) {
        int[] reduced = shape.clone();
        for (int axis : axes) {
          reduced[axis] = 1; // Collapse to 1 instead of deleting
        }
        return reduced;
    } else {
        // Your existing logic for when you actually want to drop dims
        int[] survivors = new int[shape.length - axes.length];
        int ptr = 0;
        for (int i = 0; i < shape.length; i++) {
          if (!contains(axes, i)) survivors[ptr++] = shape[i];
        }
        return survivors;
    }
  }

  /**
   * ## This overloaded version removes dimensions
   */
  public static int[] getSurvivors(int[] shape, int[] axes) {
    return getSurvivors(shape, axes, false);
}

  public static int[] unravel(int index, int[] shape) {
    int[] coords = new int[shape.length];
    for (int i = shape.length - 1; i >= 0; i--) {
      coords[i] = index % shape[i];
      index /= shape[i];
    }
    return coords;
  }

  public static boolean nextCoordinate(int[] coords, int[] shape) {
    for (int i = shape.length - 1; i >= 0; i--) {
      if (++coords[i] < shape[i]) return true;
      coords[i] = 0;
    }
    return false;
  }

  public static int sizeOf(int[] shape) {
    int vol = 1;
    for (int d : shape) vol *= d;
    return vol;
  }

  public static int[] getSubShape(int[] fullShape, int[] axes) {
    int[] subShape = new int[axes.length];
    for (int i = 0; i < axes.length; i++) subShape[i] = fullShape[axes[i]];
    return subShape;
  }

  public static boolean contains(int[] arr, int val) {
    for (int i : arr) if (i == val) return true;
    return false;
  }

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

  public static int mapToOffset(int[] resCoords, int[] kCoords, int[] axes, int[] fullShape, int[] strides, boolean isA, boolean keepDims) {
    int offset = 0;
    int kPtr = 0;
    int resPtr = 0;

    for (int i = 0; i < fullShape.length; i++) {
      if (contains(axes, i)) {
        offset += kCoords[kPtr++] * strides[i];
      } else {
        offset += resCoords[resPtr] * strides[i];
        resPtr++; 
      }
    }
    return offset;
  }

  public static String print(double[] flat, int[] shape, int[] strides, int rank, int offset, int shift) {
    int width = 0;
    for (double d : flat) {
      width = Math.max(width, String.format("%.4f", d).length());
    }

    if (shape.length == 0) return String.format("%.4f", flat[0]);
    
    StringBuilder sb = new StringBuilder("");
    
    if (rank == 0) {
      sb.append(" ".repeat(shift));
    }

    sb.append("[");
    
    for (int i = 0; i < shape[rank]; i++) {
      int currentOffset = offset + (i * strides[rank]);

      if (rank == shape.length - 1) {
        sb.append(String.format("%" + width + ".4f", flat[currentOffset]).replace(',','.'));
      } else {
        sb.append(print(flat, shape, strides, rank + 1, currentOffset, shift));
      }
      
      if (i < shape[rank] - 1) {
        if (rank == shape.length - 1) {
          sb.append(" ");
        } else {
          int newlineCount = Math.max(1, shape.length - rank - 1);
          sb.append(" ").append("\n".repeat(newlineCount));
          sb.append(" ".repeat(shift));
          sb.append(" ".repeat(rank + 1)); 
        }
      }
    }
    sb.append("]");
    return sb.toString();
  }

  public static int[] broadcastedShape(int[] a, int[] b) {
    int rank = Math.max(a.length, b.length);
    int[] result = new int[rank];

    for (int i = 0; i < rank; i++) {
      int dimA = (i >= rank - a.length) ? a[i - (rank - a.length)] : 1;
      int dimB = (i >= rank - b.length) ? b[i - (rank - b.length)] : 1;

      if (dimA != dimB && dimA != 1 && dimB != 1) throw new IllegalArgumentException("Mismatching shapes for broadcasting");

      result[i] = Math.max(dimA, dimB);
    }
    return result;
  }
  
  public static int[] squeezeShape(int[] shape) {
    if (shape.length == 0) {
      throw new IllegalArgumentException("Insufficient rank to trim: got Tensor of rank " + shape.length);
    }

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
    for (int i = 0; i < newRank; i++) result[i] = (contains(axes, i)) ? 1 : shape[shapePtr++];
    
    return result;
  }
}
