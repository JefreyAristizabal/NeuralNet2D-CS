using System;

namespace NeuralNet
{
    public static class ActivationFunction
    {
        private const double LeakySlope = 0.01;
        private const double Epsilon = 1e-12;

        public static double Sigmoid(double x)
        {
            // Clipping para evitar overflow en exp()
            x = Math.Max(-60.0, Math.Min(60.0, x));
            double result = 1.0 / (1.0 + Math.Exp(-x));
            return Clamp(result);
        }

        public static double SigmoidDerivative(double x)
        {
            double s = Sigmoid(x);
            return Clamp(s * (1 - s));
        }

        public static double Tanh(double x)
        {
            x = Math.Max(-30.0, Math.Min(30.0, x)); // previene overflow
            double result = Math.Tanh(x);
            return Clamp(result);
        }

        public static double TanhDerivative(double x)
        {
            double t = Tanh(x);
            return Clamp(1 - t * t);
        }

        public static double ReLU(double x)
        {
            return Clamp(x > 0 ? x : 0);
        }

        public static double ReLUDerivative(double x)
        {
            return x > 0 ? 1.0 : 0.0;
        }

        public static double LeakyReLU(double x)
        {
            return Clamp(x > 0 ? x : LeakySlope * x);
        }

        public static double LeakyReLUDerivative(double x)
        {
            return x > 0 ? 1.0 : LeakySlope;
        }

        // Clampea el resultado entre rangos válidos
        private static double Clamp(double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
                return 0.0;

            return Math.Max(-1e10, Math.Min(1e10, value));
        }
    }
}
