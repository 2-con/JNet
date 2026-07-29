package com.aufy.jnet.core.tensor.tools;

import java.util.Arrays;

import com.aufy.jnet.core.tensor.core.backend.compute.Memory;
import com.aufy.jnet.core.tensor.core.backend.util.ArrayTools;
import com.aufy.jnet.core.tensor.core.implementations.CoreTensor;

public class Visual {
  /*
  actually make it look nice. note that the essentials (printing raw arrays) should be moved to arrayops bc this should be
  completely isolated. 
  */

  public static void print(double[] array, int[] shape) { // don't ever use this for internal printing: use ArraOps in core/backend/util instead 
    System.out.println(ArrayTools.print(array, shape, Memory.calculateStrides(shape), 0, 0, 0));
  }

  public static void trace(CoreTensor tensor) {
    printGraph(tensor, "", true, true);
  }
  
  private static void printGraph(CoreTensor node, String indent, boolean isLast, boolean firstLayer) {
    String operation = (node.derivative != null) ? " [Product]" : " [Leaf]";
    String shape = Arrays.toString(node.shape);
    
    System.out.print(indent);
    System.out.print(isLast ? "└─ " : "├─ ");
    
    if (firstLayer) {
      System.out.println("\r#  Tensor" + shape + " [ROOT]");
    } else {
      System.out.println("Tensor" + shape + operation);
    }

    // BFS over parents
    if (node.parents != null) {
      for (int i = 0; i < node.parents.size(); i++) {
        boolean lastChild = (i == node.parents.size() - 1);
        printGraph(node.parents.get(i), indent + (isLast ? "   " : "│  "), lastChild, false);
      }
    }
  }
}
