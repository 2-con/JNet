public class MutabilitySimulation {
  static class DummyTensor {
    final double[] data;
    DummyTensor(double[] data) { this.data = data; }
  }

  public static void main(String[] args) {
    // immutable();
    mutable();
  }

  public static void mutable() {
    long startTime = System.nanoTime();

    DummyTensor weights = new DummyTensor(new double[]{0.5, -0.5, 1.0, -1.0, 0.2});
    double[] gradients = new double[]{0.01, -0.02, 0.05, -0.01, 0.03};
    double learningRate = 0.1;

    for (int epoch = 0; epoch < 1000; epoch++) {
      
      for (int i = 0; i < weights.data.length; i++) {
        weights.data[i] = weights.data[i] - (learningRate * gradients[i]);
      }
    }

    long endTime = System.nanoTime();
    System.out.printf("Mutable Simulation Time: %.4f ms (Final val: %.2f)%n", (endTime - startTime) / 1_000_000.0, weights.data[0]);
  }

  public static void immutable() {
    long startTime = System.nanoTime();

    DummyTensor weights = new DummyTensor(new double[]{0.5, -0.5, 1.0, -1.0, 0.2});
    double[] gradients = new double[]{0.01, -0.02, 0.05, -0.01, 0.03};
    double learningRate = 0.1;

    for (int epoch = 0; epoch < 1000; epoch++) {
      double[] nextData = new double[weights.data.length];
      
      for (int i = 0; i < nextData.length; i++) {
        nextData[i] = weights.data[i] - (learningRate * gradients[i]);
      }
      weights = new DummyTensor(nextData); 
    }

    long endTime = System.nanoTime();
    System.out.printf("Immutable Simulation Time: %.4f ms (Final val: %.2f)%n", (endTime - startTime) / 1_000_000.0, weights.data[0]);
  }
}
