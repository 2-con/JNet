package tensor.core;

public class Engine {
  public static int[] calculateStrides(int[] shape) {
    int[] strides = new int[shape.length];
    int st = 1;
    for (int i = shape.length - 1; i >= 0; i--) {
      strides[i] = st;
      st *= shape[i];
    }
    return strides;
  }

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

    return new int[][] { axesFromA, axesFromB };
  }

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
}
