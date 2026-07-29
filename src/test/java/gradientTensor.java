import com.aufy.jnet.Tensor;

public class gradientTensor {
  public static void main(String[] args) {
    Tensor A = new Tensor(new double[]{
      1, 2,
      3, 4
    }, 2, 2).requiresGrad();

    Tensor B = new Tensor(new double[]{
      5, 6,
      7, 8
    }, 2, 2).requiresGrad();

    Tensor C = Tensor.apply(
      A,
      B,
      (a, b) -> {
        return a.add(b);
      },
      (a, grad) -> {return grad.onesLike().mul(67);},
      (b, grad) -> {return grad.onesLike();}
    );

    C.backward();
    // System.out.println(C);

    System.out.println(A.grad());
  }
}
