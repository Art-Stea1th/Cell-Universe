using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ASD.CellUniverse.Controls.Helpers {

    internal static class BitmapHelper {

        public static WriteableBitmap CreateWriteable(int width, int height)
            => new WriteableBitmap(width, height, Default.Dpi, Default.Dpi, Default.Format, null);

        private static class Default {
            public static PixelFormat Format => PixelFormats.Bgra32;
            public static double Dpi => 96.0;
        }
    }
}