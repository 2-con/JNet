package tensor.operation;
import tensor.core.Engine;

@FunctionalInterface
public interface Reduction {
  double reduce(double accumulator, double nextValue);
  default double identity() {return 0;}
  
  Reduction sum = new Reduction() {
    @Override public double reduce(double acc, double val) {return acc + val;}
    @Override public double identity() {return 0;}
  };
  
  Reduction prod = new Reduction() {
    @Override public double reduce(double acc, double val) {return acc * val;}
    @Override public double identity() {return 1.0;}
  };
  
  Reduction max = new Reduction() {
    @Override public double reduce(double acc, double val) {return Math.max(acc, val);}
    @Override public double identity() {return Double.NEGATIVE_INFINITY;}
  };
  
  Reduction min = new Reduction() {
    @Override public double reduce(double acc, double val) {return Math.min(acc, val);}
    @Override public double identity() {return Double.POSITIVE_INFINITY;}
  };
  
  public static double[] apply(double[] data, int[] shape, int[] strides, int[] axes, Reduction operation) {
    int[] resShape = Engine.getSurvivors(shape, axes);
    double[] resData = new double[Engine.calculateVolume(resShape)];
    
    int[] subShape = Engine.getSubShape(shape, axes);
    int reductionVolume = Engine.calculateVolume(subShape);

    int[] resCoords = new int[resShape.length];
    int resIdx = 0;

    do {
      double acc = operation.identity();
      for (int k = 0; k < reductionVolume; k++) {
        int[] kCoords = Engine.unravel(k, subShape);

        int offset = Engine.mapToOffset(resCoords, kCoords, axes, shape, strides);
        acc = operation.reduce(acc, data[offset]);
      }

      resData[resIdx++] = acc;
    } while (Engine.nextCoordinate(resCoords, resShape));

    return resData;
  }
}
