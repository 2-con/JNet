package tensor.core.backend.util;

import java.util.function.DoubleUnaryOperator;

// for generic array operations
public class ArrayOps {

  // utilities

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

  public static double[] foreach(double[] array, DoubleUnaryOperator operation) {
    double[] copy = array.clone();

    for (int i = 0; i < copy.length; i++) {
      copy[i] = operation.applyAsDouble(copy[i]);
    }

    return copy;
  }

  public static double[] reverse(double[] arr) {
    double[] res = new double[arr.length];
    for (int i = 0; i < arr.length; i++) {
      res[i] = arr[arr.length - 1 - i];
    }
    return res;
  }

  public static boolean contains(int[] arr, int val) {
    for (int i : arr) if (i == val) return true;
    return false;
  }
}