namespace NeuralNet
{
    public static class Dataset
    {
        public static double[][] XOR_Inputs = new double[][]
        {
            new double[]{0,0}, new double[]{0,1},
            new double[]{1,0}, new double[]{1,1}
        };
        public static double[][] XOR_Outputs = new double[][]
        {
            new double[]{0}, new double[]{1},
            new double[]{1}, new double[]{0}
        };
    }
}
