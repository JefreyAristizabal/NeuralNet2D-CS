using System;

namespace NeuralNet
{
    public static class Extensions
    {
        /// <summary>
        /// Calcula la norma de Frobenius de una matriz.
        /// </summary>
        public static double FrobeniusNorm(this Matrix m)
        {
            double sum = 0;
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Cols; j++)
                    sum += m[i, j] * m[i, j];

            return Math.Sqrt(sum);
        }
    }
}
