import com.aufy.jnet.Tensor;

public class testTensor {
  public static void main(String[] args) {
    Tensor A = new Tensor(new double[] {
      1, 1, 2,
      1, 3, 1,
      1, 1, 1,
      2, 1, 1
    }, 4, 3).requiresGrad();
    Tensor B = new Tensor(new double[] {
      0, 1, 0,
      1, 0, 0,
      1, 0, 0,
      0, 1, 0
    }, 4, 3);

    int axis = 0;
    Tensor C = A.sum(axis).mul(1.0/A.getShape(axis));
    // Tensor C = A.prod(axis);

    C.backward();
    System.out.println(C);
    System.out.println(A.grad());

  }
}
