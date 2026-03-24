package tensor.core;

public class Engine {
  public static double[] transformData(double[] gradOutput, int[] gradShape, int[] inputShape) {
    double[] result = new double[Utility.sizeOf(inputShape)];
    int[] inputStrides = Utility.calculateStrides(inputShape);
    int[] gradStrides  = Utility.calculateStrides(gradShape);

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
    int targetSize = Utility.sizeOf(targetShape);
    int[] srcStrides = Utility.calculateStrides(srcShape);
    int[] targetStrides = Utility.calculateStrides(targetShape);
    
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
    double[] resData = new double[Utility.sizeOf(resShape)];
    int[] resCoords = new int[resShape.length];

    int[] subShapeA = Utility.getSubShape(shapeA, axesA);
    int contractVolume = Utility.sizeOf(subShapeA);

    int resIdx = 0;
    do {
      double sum = 0;
      for (int k = 0; k < contractVolume; k++) {
        int[] kCoords = Utility.unravel(k, subShapeA);
        
        int offsetA = Utility.mapToOffset(resCoords, kCoords, axesA, shapeA, stridesA, true);
        int offsetB = Utility.mapToOffset(resCoords, kCoords, axesB, shapeB, stridesB, false);
        
        sum += dataA[offsetA] * dataB[offsetB];
      }
      resData[resIdx++] = sum;
    } while (Utility.nextCoordinate(resCoords, resShape));

    return resData;
  }

  public static double[] reduceSum(double[] gradData, int[] gradShape, int[] origShape) {
    double[] reduced = new double[Utility.sizeOf(origShape)];
    int[] gradStrides = Utility.calculateStrides(gradShape);
    int[] origStrides = Utility.calculateStrides(origShape);

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
