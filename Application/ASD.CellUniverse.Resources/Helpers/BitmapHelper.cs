﻿using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Resources.Helpers {

    internal static class BitmapHelper {

        public static WriteableBitmap Valid(WriteableBitmap bitmap, Size size)
            => bitmap == null || bitmap.PixelWidth != (int)size.Width || bitmap.PixelHeight != (int)size.Height
            ? CreateWriteable(size) : bitmap;

        public static WriteableBitmap CreateWriteable(Size size)
            => new WriteableBitmap((int)size.Width, (int)size.Height, Default.Dpi, Default.Dpi, Default.Format, null);

        private static class Default {

            public static PixelFormat Format => PixelFormats.Bgra32;
            public static double Dpi => 96.0;
        }
    }
}