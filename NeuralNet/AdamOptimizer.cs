using System;
using System.Collections.Generic;

namespace NeuralNet
{
    public class AdamOptimizer : IOptimizer
    {
        private readonly double lr, b1, b2, eps, clipNorm;
        private int t;
        private readonly Dictionary<Layer, (Matrix mw, Matrix vw, Matrix mb, Matrix vb)> m;

        public AdamOptimizer(double learningRate = 0.0003, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8, double clipNorm = 5.0)
        {
            lr = learningRate; b1 = beta1; b2 = beta2; eps = epsilon; this.clipNorm = clipNorm;
            m = new Dictionary<Layer, (Matrix, Matrix, Matrix, Matrix)>();
            t = 0;
        }

        public void Update(Layer layer)
        {
            t++;
            if (!m.ContainsKey(layer))
            {
                m[layer] = (
                    new Matrix(layer.WeightGradients.Rows, layer.WeightGradients.Cols),
                    new Matrix(layer.WeightGradients.Rows, layer.WeightGradients.Cols),
                    new Matrix(layer.BiasGradients.Rows, layer.BiasGradients.Cols),
                    new Matrix(layer.BiasGradients.Rows, layer.BiasGradients.Cols)
                );
            }

            var (mw, vw, mb, vb) = m[layer];

            // Clip
            layer.WeightGradients.ClipByNorm(clipNorm);
            layer.BiasGradients.ClipByNorm(clipNorm);

            // Update weights momentums
            for (int i = 0; i < layer.WeightGradients.Rows; i++)
                for (int j = 0; j < layer.WeightGradients.Cols; j++)
                {
                    double g = layer.WeightGradients[i, j];
                    mw[i, j] = b1 * mw[i, j] + (1 - b1) * g;
                    vw[i, j] = b2 * vw[i, j] + (1 - b2) * g * g;

                    double mHat = mw[i, j] / (1 - Math.Pow(b1, t));
                    double vHat = vw[i, j] / (1 - Math.Pow(b2, t));
                    double update = lr * mHat / (Math.Sqrt(vHat) + eps);

                    layer.Weights[i, j] -= update;
                }

            // Update biases momentums
            for (int i = 0; i < layer.BiasGradients.Rows; i++)
                for (int j = 0; j < layer.BiasGradients.Cols; j++)
                {
                    double g = layer.BiasGradients[i, j];
                    mb[i, j] = b1 * mb[i, j] + (1 - b1) * g;
                    vb[i, j] = b2 * vb[i, j] + (1 - b2) * g * g;

                    double mHat = mb[i, j] / (1 - Math.Pow(b1, t));
                    double vHat = vb[i, j] / (1 - Math.Pow(b2, t));
                    double update = lr * mHat / (Math.Sqrt(vHat) + eps);

                    layer.Biases[i, j] -= update;
                }

            // Limpieza opcional de gradientes
            layer.WeightGradients.Map(_ => 0);
            layer.BiasGradients.Map(_ => 0);
        }
    }
}
