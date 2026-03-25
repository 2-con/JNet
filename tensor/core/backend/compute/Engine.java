package tensor.core.backend.compute;

import stats.Statistics;

public class Engine {
  public static double[] transformData(double[] gradOutput, int[] gradShape, int[] inputShape) {
    double[] result = new double[Statistics.prod(inputShape)];
    int[] inputStrides = PointerLogic.calculateStrides(inputShape);
    int[] gradStrides  = PointerLogic.calculateStrides(gradShape);

    for (int i = 0; i < result.length; i++) {
      int remaining = i;
      int gradIndex = 0;

      for (int dim = 0; dim < inputShape.length; dim++) {
        int coord = (remaining / inputStrides[dim]) % inputShape[dim];
        
        if (gradShape[dim] != 1) {
          gradIndex += coord * gradStrides[dim];
        }
      }
      result[i] = gradOutput[gradIndex];
    }
    return result;
  }

  public static double[] broadcast(double[] data, int[] srcShape, int[] targetShape) {
    int targetSize = Statistics.prod(targetShape);
    int[] srcStrides = PointerLogic.calculateStrides(srcShape);
    int[] targetStrides = PointerLogic.calculateStrides(targetShape);
    
    double[] out = new double[targetSize];

    for (int i = 0; i < targetSize; i++) {
      int srcIdx = 0;
      int remaining = i;

      for (int dim = 0; dim < targetShape.length; dim++) {
        int coord = remaining / targetStrides[dim];
        remaining %= targetStrides[dim];

        if (srcShape[dim] != 1) {
          srcIdx += coord * srcStrides[dim];
        }
      }
      out[i] = data[srcIdx];
    }
    return out;
  }

  public static double[] contract(double[] dataA, int[] stridesA, double[] dataB, int[] stridesB, int[] shapeA, int[] axesA, int[] shapeB, int[] axesB, int[] resShape) {
    double[] resData = new double[Statistics.prod(resShape)];
    
    int[] subShape = Shaping.getSubShape(shapeA, axesA);
    int contractVolume = Statistics.prod(subShape);
    
    int[] kOffsetsA = Shaping.precomputeKOffsets(subShape, axesA, stridesA);
    int[] kOffsetsB = Shaping.precomputeKOffsets(subShape, axesB, stridesB);
    
    int[] resCoords = new int[resShape.length];
    int resIdx = 0;
    do {
      int baseOffsetA = PointerLogic.mapToOffset(resCoords, new int[axesA.length], axesA, shapeA, stridesA, true);
      int baseOffsetB = PointerLogic.mapToOffset(resCoords, new int[axesB.length], axesB, shapeB, stridesB, false);

      double sum = 0;

      for (int k = 0; k < contractVolume; k++) {
        sum += dataA[baseOffsetA + kOffsetsA[k]] * dataB[baseOffsetB + kOffsetsB[k]];
      }
      resData[resIdx++] = sum;

    } while (PointerLogic.nextCoordinate(resCoords, resShape));

    return resData;
  }

  public static double[] reduceSum(double[] gradData, int[] gradShape, int[] origShape) {
    double[] reduced = new double[Statistics.prod(origShape)];
    int[] gradStrides = PointerLogic.calculateStrides(gradShape);
    int[] origStrides = PointerLogic.calculateStrides(origShape);

    for (int i = 0; i < gradData.length; i++) {
      int remaining = i;
      int origIdx = 0;

      for (int dim = 0; dim < gradShape.length; dim++) {
        int coord = (remaining / gradStrides[dim]) % gradShape[dim];
        
        if (origShape[dim] != 1) {
          origIdx += coord * origStrides[dim];
        }
      }
      reduced[origIdx] += gradData[i];
    }
    return reduced;
  }
}
