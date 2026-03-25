package tensor.core.backend.compute;

import tensor.core.backend.util.ArrayOps;

public class PointerLogic {
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

  public static int getOffset(int[] strides, int[] coords) {
    int offset = 0;
    for (int i = 0; i < coords.length; i++) {
      offset += coords[i] * strides[i];
    }
    return offset;
  }

  public static boolean nextCoordinate(int[] coords, int[] shape) {
    for (int i = shape.length - 1; i >= 0; i--) {
      if (++coords[i] < shape[i]) return true;
      coords[i] = 0;
    }
    return false;
  }

  public static int mapToOffset(int[] resCoords, int[] kCoords, int[] axes, int[] fullShape, int[] strides, boolean isA) {
    int[] fullCoords = new int[fullShape.length];
    
    for (int i = 0; i < axes.length; i++) {
      fullCoords[axes[i]] = kCoords[i];
    }

    int resPtr = isA ? 0 : (resCoords.length - (fullShape.length - axes.length));
    for (int i = 0; i < fullShape.length; i++) {
      if (!ArrayOps.contains(axes, i)) {
        fullCoords[i] = resCoords[resPtr++];
      }
    }
    
    return getOffset(strides, fullCoords);
  }
}
