using System;
using System.IO;
using System.Drawing;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media.Imaging;

namespace GameOfLife {

    public class BitmapOps {

        public static Color True  = Color.FromArgb(-1);
        public static Color False = Color.FromArgb(-16777216);

        public static bool[,] ToBoolArray(Bitmap source, int factor) {

            bool[,] result = new bool[source.Height / factor, source.Width / factor];

            for (int i = 0; i < source.Height / factor; i++) {
                for (int j = 0; j < source.Width / factor; j++) {
                    result[i, j] = source.GetPixel(j * factor, i * factor) == True;
                }
            }
            return result;
        }

        public static Bitmap ToBitmap(bool[,] source, int factor) {

            Bitmap result = new Bitmap(source.GetLength(0) * factor, source.GetLength(1) * factor);

            for (int i = 0; i < source.GetLength(0) * factor; i++) {
                for (int j = 0; j < source.GetLength(1) * factor; j++) {
                    result.SetPixel(i, j, source[i / factor, j / factor] ? True : False);
                }
            }
            return result;
        }

        public static Bitmap UpScale(Bitmap source, int factor) {

            Bitmap result = new Bitmap(source.Width * factor, source.Height * factor);

            for (int i = 0; i < source.Width * factor; i++) {
                for (int j = 0; j < source.Height * factor; j++) {
                    result.SetPixel(i, j, source.GetPixel(i / factor, j / factor));
                }
            }
            return result;
        }

        public static Bitmap DownScale(Bitmap source, int factor) {

            Bitmap result = new Bitmap(source.Width / factor, source.Height / factor);

            for (int i = 0; i < source.Width / factor; i++) {
                for (int j = 0; j < source.Height / factor; j++) {
                    result.SetPixel(i, j, source.GetPixel(i * factor, j * factor));
                }
            }
            return result;
        }
    }
}