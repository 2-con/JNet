package tensor.tools;

import java.util.Arrays;
import tensor.TensorCore;
import tensor.core.Memory;
import tensor.core.Traceable;

public class Debug {
  public static void print(double[] array, int[] shape) {
    System.out.println(ArrayTools.print(array, shape, Memory.calculateStrides(shape), 0, 0, 0));
  }

  public static void print(Traceable tensor) {
    System.out.println(tensor);
  }

  public static void print(TensorCore tensor) {
    System.out.println(tensor);
  }

  public static void trace(Traceable tensor) {
    printGraph(tensor, "", true, true);
  }
  
  private static void printGraph(Traceable node, String indent, boolean isLast, boolean firstLayer) {
    String operation = (node.getGradFunc() != null) ? " [Product]" : " [Leaf]";
    String shape = Arrays.toString(node.getShape());
    
    System.out.print(indent);
    System.out.print(isLast ? "└─ " : "├─ ");
    
    if (firstLayer) {
      System.out.println("\r▣  Tensor" + shape + " [ROOT]");
    } else {
      System.out.println("Tensor" + shape + operation);
    }

    // 2. Recurse into parents
    if (node.getParents() != null) {
      for (int i = 0; i < node.getParents().size(); i++) {
        boolean lastChild = (i == node.getParents().size() - 1);
        printGraph(node.getParents().get(i), indent + (isLast ? "   " : "│  "), lastChild, false);
      }
    }
  }

}
