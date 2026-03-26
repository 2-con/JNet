package com.aufy.jnet.tensor.graph.main;

import java.util.List;
import com.aufy.jnet.tensor.core.backend.compute.Engine;
import com.aufy.jnet.tensor.core.backend.compute.Shaping;
import com.aufy.jnet.tensor.core.backend.func.Binary;
import com.aufy.jnet.tensor.core.impl.DataContainer;
import com.aufy.jnet.tensor.core.impl.TensorCore;
import com.aufy.jnet.tensor.functional.main.CoreBinaryOps;
import com.aufy.jnet.tensor.functional.main.CoreShapeOps;

public class BinaryOps {
    // ==============================================================================================
    // GENERIC
    // ==============================================================================================
    
    public static TensorCore elementwise(TensorCore tensorA, TensorCore tensorB, Binary op, Binary dA, Binary dB) {
      if (tensorA.rank != tensorB.rank) {
        throw new IllegalArgumentException("Mismatching rank for binary elementwise operation. Got tensors of rank " + tensorA.rank + " and " + tensorB.rank);
      }
      
      int[] broadcastShapeTarget = Shaping.broadcastedShape(tensorA.shape, tensorB.shape);
      
      DataContainer coreA = CoreShapeOps.broadcast(tensorA.core, broadcastShapeTarget);
      DataContainer coreB = CoreShapeOps.broadcast(tensorB.core, broadcastShapeTarget);
      
      TensorCore out = new TensorCore(CoreBinaryOps.elementwise(coreA, coreB, op));
      
      out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;
      
      if (out.requiresGrad) {
        out.parents = List.of(tensorA, tensorB);
        
        out.derivative = (grad) -> {
          if (tensorA.requiresGrad) {
            DataContainer dACore = CoreBinaryOps.elementwise(coreA, coreB, dA);
            DataContainer gradA  = CoreBinaryOps.elementwise(grad.core, dACore, (x, y) -> x * y);
            
            double[] reducedData = Engine.reduceSum(gradA.dump(), broadcastShapeTarget, tensorA.shape);
            tensorA.accumulate(new TensorCore(reducedData, tensorA.shape));
          }
          
          if (tensorB.requiresGrad) {
            DataContainer dBCore = CoreBinaryOps.elementwise(coreA, coreB, dB);
            DataContainer gradB  = CoreBinaryOps.elementwise(grad.core, dBCore, (x, y) -> x * y);
            
            double[] reducedData = Engine.reduceSum(gradB.dump(), broadcastShapeTarget, tensorB.shape);
            tensorB.accumulate(new TensorCore(reducedData, tensorB.shape));
          }
        };
      }
      
      return out;
    }
    
    public static TensorCore contract(TensorCore tensorA, TensorCore tensorB, int[] axesA, int[] axesB) {
      TensorCore out = new TensorCore(CoreBinaryOps.contract(tensorA.core, tensorB.core, axesA, axesB));
      out.requiresGrad = tensorA.requiresGrad || tensorB.requiresGrad;
      
      if (out.requiresGrad) {
        out.parents = List.of(tensorA, tensorB);
        
        // returns the grad axes of A and B
        int[][] survivors = Shaping.getResultAxes(tensorA.shape, tensorB.shape, axesA, axesB);
        
        out.derivative = (grad) -> {
          if (tensorA.requiresGrad) {
            DataContainer gradACore = CoreBinaryOps.contract(grad.core, tensorB.core, survivors[1], axesB);
            tensorA.accumulate(new TensorCore(gradACore));
          }
          
          if (tensorB.requiresGrad) {
            DataContainer gradBCore = CoreBinaryOps.contract(tensorA.core, grad.core, axesA, survivors[0]);
            tensorB.accumulate(new TensorCore(gradBCore));
          }
        };
      }
      
      return out;
    }

    // ==============================================================================================
    // IMPLEMENTATION
    // ==============================================================================================

  public static TensorCore add(TensorCore tensorA, TensorCore tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x + y, (x, y) -> 1.0, (x, y) -> 1.0);
  }

  public static TensorCore sub(TensorCore tensorA, TensorCore tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x - y, (x, y) -> 1.0, (x, y) -> -1.0);
  }

  public static TensorCore hadamard(TensorCore tensorA, TensorCore tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x * y, (x, y) -> y, (x, y) -> x);
  }

  public static TensorCore div(TensorCore tensorA, TensorCore tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> x / y, (x, y) -> 1.0 / y, (x, y) -> -x / Math.pow(y, 2));
  }

  public static TensorCore pow(TensorCore tensorA, TensorCore tensorB) {
    return elementwise(tensorA, tensorB, (x, y) -> Math.pow(x, y), (x, y) -> y * Math.pow(x, y - 1), (x, y) -> Math.pow(x, y) * Math.log(x));
  }
}
