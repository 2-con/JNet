package tensor.functional.main;

import java.util.Arrays;
import tensor.core.backend.compute.Engine;
import tensor.core.backend.compute.Shaping;
import tensor.core.backend.func.Binary;
import tensor.core.impl.DataContainer;

public class CoreBinaryOps {
  public static DataContainer elementwise(DataContainer a, DataContainer b, Binary operation) {
    if (a.rank != b.rank) {
      throw new IllegalArgumentException("Mismatching rank for binary elementwise operation. Got tensors of rank " + a.rank + " and " + b.rank);
    }
    if (!Arrays.equals(a.shape, b.shape)) {
      int[] broadcastShapeTarget = Shaping.broadcastedShape(a.shape, b.shape);

      a = CoreShapeOps.broadcast(a, broadcastShapeTarget);
      b = CoreShapeOps.broadcast(b, broadcastShapeTarget);
    }

    double[] resultData = Binary.apply(a.data, b.data, operation);
    return new DataContainer(resultData, a.shape);
  }

  public static DataContainer contract(DataContainer a, DataContainer b, int[] axesA, int[] axesB) {
    // if (axesA.length > a.rank) throw new IllegalArgumentException("Mismatching axes to contract Tensor A by, attempting to contract along " + axesA.length + " axes when Tensor A has a rank of " + a.rank );
    // if (axesB.length > b.rank) throw new IllegalArgumentException("Mismatching axes to contract Tensor B by, attempting to contract along " + axesB.length + " axes when Tensor A has a rank of " + b.rank);

    for (int i = 0; i < axesA.length; i++) {
      if (a.shape[axesA[i]] != b.shape[axesB[i]]) {
        throw new IllegalArgumentException("Mismatching dimensions at contraction axes, tensor A has size " + a.shape[axesA[i]] + " at index " + axesA[i] +  " while tensor B has size " + b.shape[axesB[i]]  + " at index " + axesB[i]);
      }
    }

    int[] resShape = Shaping.calculateResShape(a.shape, b.shape, axesA, axesB);
    double[] resultData = Engine.contract(a.data, a.strides, b.data, b.strides, a.shape, axesA, b.shape, axesB, resShape);

    return new DataContainer(resultData, resShape);
  }
  
}
