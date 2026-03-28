package com.aufy.jnet.tensor.tools;

import java.util.Arrays;

import com.aufy.jnet.stats.primitive.Statistics;
import com.aufy.jnet.tensor.core.backend.compute.PointerLogic;
import com.aufy.jnet.tensor.core.backend.util.ArrayTools;
import com.aufy.jnet.tensor.core.impl.CoreTensor;

public class Visual {
  /*
  actually make it look nice. note that the essentials (printing raw arrays) should be moved to arrayops bc this should be
  completely isolated. 
  */

  public static void print(double[] array, int[] shape) { // don't ever use this for internal printing: use ArraOps in core/backend/util instead 
    System.out.println(ArrayTools.print(array, shape, PointerLogic.calculateStrides(shape), 0, 0, 0));
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
      System.out.println("\r▣  Tensor" + shape + " [ROOT]");
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

  public static void statisticalBreakdown(CoreTensor tensor) {
    System.out.println("TensorCore \n"
      + Arrays.toString(tensor.shape)
      + "\n requiresGrad = " + tensor.requiresGrad
      + "\n size = " + tensor.size
      + "\n rank = " + tensor.rank
      + "\n Statistical Breakdown"
      + "\n mean = " + Statistics.mean(tensor.dump())
      + "\n stdev = " + Statistics.stdev(tensor.dump())
      + "\n min = " + Statistics.min(tensor.dump())
      + "\n max = " + Statistics.max(tensor.dump())
    );
  }

}
