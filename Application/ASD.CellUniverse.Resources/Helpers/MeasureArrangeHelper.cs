using System.Windows;

namespace ASD.CellUniverse.Resources.Helpers {

    internal static class MeasureArrangeHelper {

        public static Size ComputeSize(Size availableSize, Size contentSize) {
            var scaleFactor = ComputeScaleFactor(availableSize, contentSize);
            return new Size(contentSize.Width * scaleFactor, contentSize.Height * scaleFactor);
        }

        public static double ComputeScaleFactor(Size availableSize, Size contentSize) {

            var isConstrainedWidth = !double.IsPositiveInfinity(availableSize.Width);
            var isConstrainedHeight = !double.IsPositiveInfinity(availableSize.Height);

            if (isConstrainedWidth || isConstrainedHeight) {

                var scaleX = contentSize.Width == 0.0 ? 0.0 : availableSize.Width / contentSize.Width;
                var scaleY = contentSize.Height == 0.0 ? 0.0 : availableSize.Height / contentSize.Height;

                if (!isConstrainedWidth) { return scaleY; }
                if (!isConstrainedHeight) { return scaleX; }

                return scaleX < scaleY ? scaleX : scaleY;
            }
            return 1.0;
        }
    }
}