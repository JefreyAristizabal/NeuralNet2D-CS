using System.IO;

namespace NeuralNet
{
    public static class Exporter
    {
        public static void Export2DClassificationGrid(NeuralNetwork net, string filePath, int resolution = 100)
        {
            using var writer = new StreamWriter(filePath);
            writer.WriteLine("x,y,predicted");

            for (int i = 0; i <= resolution; i++)
                for (int j = 0; j <= resolution; j++)
                {
                    double x = i / (double)resolution;
                    double y = j / (double)resolution;
                    double p = net.Predict(new double[] { x, y })[0];
                    writer.WriteLine($"{x:F3},{y:F3},{p:F5}");
                }
        }
    }
}
