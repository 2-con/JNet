package com.aufy.jnet.statistics.distributions;

import java.util.Random;

import com.aufy.jnet.core.backend.exceptions.statistics.Parameter;

/**
 * Exponential distribution. Note that JNet use "lambda" over "beta" unlike other frameworks.
 */
public class Exponential extends Distribution {
  private static final Random RNG = new Random();

  private final double lambda;

  public Exponential(double lambda) {
    Parameter.isPositive(lambda);

    this.lambda = lambda;
  }

  @Override
  public double sample() {
    return RNG.nextExponential() / lambda;
  }

  @Override
  public double mean() {
    return 1/lambda;
  }

  @Override
  public double variance() {
    return 1/ (lambda * lambda);
  }

  @Override
  public String toString() {
    return "Exponential(lambda = " + lambda + ")";
  }
}
