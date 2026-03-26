package com.aufy.jnet.tensor.tools;

import java.util.Arrays;

import com.aufy.jnet.stats.Statistics;
import com.aufy.jnet.tensor.core.backend.compute.PointerLogic;
import com.aufy.jnet.tensor.core.backend.util.ArrayOps;
import com.aufy.jnet.tensor.core.impl.TensorCore;
import com.aufy.jnet.tensor.core.impl.Traceable;

public class Visual {
  public static void print(double[] array, int[] shape) { // don't ever use this for internal printing: use ArraOps in core/backend/util instead 
    System.out.println(ArrayOps.print(array, shape, PointerLogic.calculateStrides(shape), 0, 0, 0));
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

  public static void statisticalBreakdown(TensorCore tensor) {
    System.out.println("TensorCore \n"
      + Arrays.toString(tensor.getShape())
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
