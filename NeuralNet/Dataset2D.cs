using System;

namespace NeuralNet
{
    public static class Dataset2D
    {
        private static Random rand = new Random();

        public static (double[][] inputs, double[][] outputs) Generate(int samplesPerClass = 100)
        {
            int total = samplesPerClass * 2;
            double[][] inputs = new double[total][];
            double[][] outputs = new double[total][];

            // Centros para cada clase (dos subclusteres)
            (double x, double y)[] c0 = { (0.2, 0.2), (0.3, 0.6) };
            (double x, double y)[] c1 = { (0.7, 0.7), (0.6, 0.3) };
            double spread = 0.1;

            for (int i = 0; i < samplesPerClass; i++)
            {
                var a = c0[rand.Next(c0.Length)];
                double x0 = Clamp(a.x + (rand.NextDouble() - 0.5) * spread);
                double y0 = Clamp(a.y + (rand.NextDouble() - 0.5) * spread);
                inputs[i] = new double[] { x0, y0 }; outputs[i] = new double[] { 0 };

                var b = c1[rand.Next(c1.Length)];
                double x1 = Clamp(b.x + (rand.NextDouble() - 0.5) * spread);
                double y1 = Clamp(b.y + (rand.NextDouble() - 0.5) * spread);
                inputs[i + samplesPerClass] = new double[] { x1, y1 }; outputs[i + samplesPerClass] = new double[] { 1 };
            }

            return (inputs, outputs);
        }

        private static double Clamp(double v) => Math.Max(0, Math.Min(1, v));
    }
}
