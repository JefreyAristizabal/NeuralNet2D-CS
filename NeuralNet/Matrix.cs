using System;

namespace NeuralNet
{
    public class Matrix
    {
        private readonly double[,] data;

        public int Rows => data.GetLength(0);
        public int Cols => data.GetLength(1);
        public int Columns => Cols;

        public Matrix(int rows, int cols)
        {
            if (rows <= 0 || cols <= 0)
                throw new ArgumentException("Rows and Cols must be > 0.");
            data = new double[rows, cols];
        }

        public double this[int i, int j]
        {
            get => data[i, j];
            set
            {
                if (double.IsNaN(value) || double.IsInfinity(value))
                    throw new ArgumentException($"Invalid matrix value at [{i},{j}]: {value}");
                data[i, j] = value;
            }
        }

        public static Matrix FromArray(double[] arr)
        {
            var m = new Matrix(arr.Length, 1);
            for (int i = 0; i < arr.Length; i++) m[i, 0] = arr[i];
            return m;
        }

        public double[] ToArray()
        {
            var result = new double[Rows * Cols];
            int idx = 0;
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    result[idx++] = data[i, j];
            return result;
        }

        public Matrix Copy()
        {
            var m = new Matrix(Rows, Cols);
            Array.Copy(data, m.data, data.Length);
            return m;
        }

        public static Matrix Random(int rows, int cols, double min = -1, double max = 1)
        {
            var rand = new Random();
            var m = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    m[i, j] = min + (max - min) * rand.NextDouble();
            return m;
        }

        public static Matrix Random(int rows, int cols, double stdDev)
            => Random(rows, cols, -stdDev, stdDev);

        public static Matrix Dot(Matrix a, Matrix b)
        {
            if (a.Cols != b.Rows)
                throw new ArgumentException("Incompatible dims for Dot");
            var m = new Matrix(a.Rows, b.Cols);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Cols; j++)
                    for (int k = 0; k < a.Cols; k++)
                        m[i, j] += a[i, k] * b[k, j];
            return m;
        }

        public Matrix Transpose()
            => Transpose(this);

        public static Matrix Transpose(Matrix a)
        {
            var m = new Matrix(a.Cols, a.Rows);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    m[j, i] = a[i, j];
            return m;
        }

        public static Matrix Hadamard(Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols)
                throw new ArgumentException("Dim mismatch for Hadamard");
            var m = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < m.Rows; i++)
                for (int j = 0; j < m.Cols; j++)
                    m[i, j] = a[i, j] * b[i, j];
            return m;
        }

        public void Map(Func<double, double> f)
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    this[i, j] = f(this[i, j]);
        }

        public static Matrix Map(Matrix a, Func<double, double> f)
        {
            var m = a.Copy();
            m.Map(f);
            return m;
        }

        public void Add(Matrix b)
        {
            if (Rows != b.Rows || Cols != b.Cols) throw new ArgumentException("Dim mismatch for Add");
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    this[i, j] += b[i, j];
        }

        public void AddScaled(Matrix b, double scale)
        {
            if (Rows != b.Rows || Cols != b.Cols) throw new ArgumentException("Dim mismatch for AddScaled");
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    this[i, j] += b[i, j] * scale;
        }

        public void ClipByNorm(double clipNorm)
        {
            double sum = 0;
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    sum += data[i, j] * data[i, j];
            double norm = Math.Sqrt(sum);
            if (norm > clipNorm && norm > 1e-8)
            {
                double scale = clipNorm / norm;
                for (int i = 0; i < Rows; i++)
                    for (int j = 0; j < Cols; j++)
                        data[i, j] *= scale;
            }
        }

        public static Matrix ApplyFunction(Matrix a, Matrix b, Func<double, double, double> func)
        {
            if (a.Rows != b.Rows || a.Cols != b.Cols) throw new ArgumentException("Dim mismatch in ApplyFunction");
            var m = new Matrix(a.Rows, a.Cols);
            for (int i = 0; i < a.Rows; i++)
                for (int j = 0; j < a.Cols; j++)
                    m[i, j] = func(a[i, j], b[i, j]);
            return m;
        }

        public double Average()
        {
            double sum = 0;
            int count = Rows * Cols;
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    sum += data[i, j];
            return sum / count;
        }

        public bool HasNaN()
        {
            for (int i = 0; i < Rows; i++)
                for (int j = 0; j < Cols; j++)
                    if (double.IsNaN(data[i, j]) || double.IsInfinity(data[i, j]))
                        return true;
            return false;
        }
    }
}
