using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Extensions {

    internal static class Extensions {

        public static IEnumerable<T> AsEnumerable<T>(this T[,] array2D) {
            var enumerator = array2D.GetEnumerator();
            while (enumerator.MoveNext()) {
                yield return (T)enumerator.Current;
            }
        }

        public static IEnumerable<(int x, int y, T value)> AsEnumerableIndexed<T>(this T[,] array2D) {
            for (var x = 0; x < array2D.GetLength(0); ++x) {
                for (var y = 0; y < array2D.GetLength(1); ++y) {
                    yield return (x: x, y: y, value: array2D[x, y]);
                }
            }
        }

        // -----

        public static Int32Rect GetFullRect(this WriteableBitmap bitmap)
            => new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight);

        public static uint Bgra32(this Color color) // Little Endian
            => ((uint)color.A << 24) | ((uint)color.R << 16) | ((uint)color.G << 8) | color.B;

        public static uint ToAlpha(this byte value) => ((uint)value) << 24;

        public static Color Argb32(this uint color)
            => Color.FromArgb((byte)(color >> 24), (byte)(color >> 16), (byte)(color >> 8), (byte)(color));

        public static Color Argb32(this int color)
            => Color.FromArgb((byte)(color >> 24), (byte)(color >> 16), (byte)(color >> 8), (byte)(color));

        public static void WritePixels(this WriteableBitmap bitmap, uint[] pixels)
            => bitmap.WritePixels(bitmap.GetFullRect(), pixels, bitmap.BackBufferStride, 0);

        public static void WriteSquares() {

        }

        // -----

        public static int CountX<T>(this T[,] array2D) => array2D.GetLength(0);
        public static int CountY<T>(this T[,] array2D) => array2D.GetLength(1);
    }
}