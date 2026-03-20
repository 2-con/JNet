package tensor.core;

public class Engine {
  public static int[] calculateStrides(int[] dims) {
    int[] strides = new int[dims.length];
    int st = 1;
    for (int i = dims.length - 1; i >= 0; i--) {
      strides[i] = st;
      st *= dims[i];
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
}
