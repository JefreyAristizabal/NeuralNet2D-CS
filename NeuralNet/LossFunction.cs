using System;

namespace NeuralNet
{
    public class LossFunction
    {
        public static double MSE(double[] outputs, double[] targets)
        {
            if (outputs.Length != targets.Length)
                throw new ArgumentException("Dimensiones de salida y objetivo no coinciden");

            double sum = 0;
            for (int i = 0; i < outputs.Length; i++)
            {
                double o = outputs[i], t = targets[i];
                if (!double.IsFinite(o) || !double.IsFinite(t))
                    throw new Exception("❌ MSE recibió valores inválidos (NaN o Inf)");
                sum += Math.Pow(t - o, 2);
            }
            return sum / outputs.Length;
        }

        public static double[] MSE_Derivative(double[] outputs, double[] targets)
        {
            if (outputs.Length != targets.Length)
                throw new ArgumentException("Dimensiones de salida y objetivo no coinciden");

            double[] result = new double[outputs.Length];
            for (int i = 0; i < outputs.Length; i++)
            {
                double o = outputs[i], t = targets[i];
                if (!double.IsFinite(o) || !double.IsFinite(t))
                    throw new Exception("❌ MSE_Derivative recibió valores inválidos (NaN o Inf)");
                result[i] = o - t;
            }
            return result;
        }

        public static double BinaryCrossEntropy(double[] predicted, double[] target)
        {
            double sum = 0.0;
            for (int i = 0; i < predicted.Length; i++)
            {
                double y = target[i];
                double p = Math.Max(1e-15, Math.Min(1 - 1e-15, predicted[i])); // evitar log(0)
                sum += -(y * Math.Log(p) + (1 - y) * Math.Log(1 - p));
            }
            return sum / predicted.Length;
        }

        public static double[] BinaryCrossEntropyDerivative(double[] predicted, double[] target)
        {
            double[] gradient = new double[predicted.Length];
            for (int i = 0; i < predicted.Length; i++)
            {
                double y = target[i];
                double p = Math.Max(1e-15, Math.Min(1 - 1e-15, predicted[i])); // estabilidad numérica
                gradient[i] = (p - y) / (p * (1 - p)); // derivada de la función
            }
            return gradient;
        }

    }

    public enum LossType
    {
        MSE,
        BinaryCrossEntropy,
        BinaryCrossEntropyDerivative
    }
}
